import argparse
import json
import networkx as nx

import torch
import torch.nn.functional as F
from torch import optim

from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx

from model import pi_GNN

size_largest_graph = {'MUTAG': 28, 'DD': 5748, 'NCI1': 111, 'PROTEINS_full': 620, 'ENZYMES': 126, 'IMDB-BINARY': 136, 'IMDB-MULTI': 89, 'REDDIT-BINARY': 3782, 'REDDIT-MULTI-5K': 3648, 'COLLAB': 492}


class Features(object):
    def __init__(self, n_max, cat=True):
        self.n_max = n_max
        self.cat = cat

    def __call__(self, data):
        x = data.x
        G = to_networkx(data, to_undirected=True)
        G.remove_edges_from(nx.selfloop_edges(G))
        degs = [G.degree(v) for v in G.nodes()]
        triangles = [nx.triangles(G,v) for v in G.nodes()]
        degs = torch.FloatTensor(degs).unsqueeze(1)
        triangles = torch.FloatTensor(triangles).unsqueeze(1)
        deg_tri = torch.cat([degs,triangles], dim=-1)

        if x is not None and self.cat:
            x = x.view(-1, 1) if x.dim() == 1 else x
            features = torch.zeros(self.n_max, x.size(1)+2, dtype=x.dtype)
            if x.size(0) <= self.n_max:
                features[:x.size(0),:2] = deg_tri[:x.size(0),:]
                features[:x.size(0),2:] = x
            else:
                features[:,:2] = deg_tri[:self.n_max,:]
                features[:,2:] = x[:self.n_max,:]
        else:
            features = torch.zeros(self.n_max, 2)
            if G.number_of_nodes() <= self.n_max:
                features[:G.number_of_nodes(),:] = deg_tri
            else:
                features = deg_tri[:self.n_max,:]

        data.x = features
        return data


class MyTransform(object):
    def __init__(self, n_max):
        self.n_max = n_max

    def __call__(self, data):
        x, edge_index = data.x, data.edge_index
        self_loops = torch.cat([torch.arange(0, self.n_max).unsqueeze(0), torch.arange(0, self.n_max).unsqueeze(0)], dim=0)
        if data.num_nodes <= self.n_max:
            adj = torch.zeros(self.n_max, self.n_max)
            adj[edge_index[0,:], edge_index[1,:]] = 1
        else:
            G = to_networkx(data, to_undirected=True)
            G.remove_nodes_from(range(self.n_max, G.number_of_nodes()))
            adj = torch.LongTensor(nx.to_numpy_matrix(G))
            edge_index = torch.nonzero(adj).t()

        data.num_nodes = self.n_max
        data.num_edges = edge_index.size(1)
        edge_index = torch.cat([edge_index, self_loops], dim=1)
        data.adj = adj
        data.edge_index = edge_index
        return data

# Argument parser
parser = argparse.ArgumentParser(description='pi_GNN')
parser.add_argument('--dataset', default='IMDB-BINARY', help='Dataset name')
parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train')
parser.add_argument('--hidden-nodes', type=int, default=20, help='Number of latent nodes')
parser.add_argument('--hidden-dim', type=int, default=32, help='Size of hidden layer')
parser.add_argument('--alpha', type=float, default=0.5, help='Trade-off between structural features and node attributes')
parser.add_argument('--use-dustbins', action='store_true', default=True, help='Whether to use dustbins')
parser.add_argument('--features', type=str, default='raw', choices=['gcn', 'gin', 'raw'])
parser.add_argument('--n-layers', type=int, default=2, help='Number of message passing layers')
args = parser.parse_args()

use_node_attr = False
if args.dataset == 'ENZYMES' or args.dataset == 'PROTEINS_full':
    use_node_attr = True

n_max = size_largest_graph[args.dataset]

dataset = TUDataset(root='./datasets/'+args.dataset, name=args.dataset, use_node_attr=use_node_attr, pre_transform=MyTransform(n_max), transform=Features(n_max))

with open('data_splits/'+args.dataset+'_splits.json','rt') as f:
    for line in f:
        splits = json.loads(line)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def train(epoch, loader, optimizer):
    model.train()
    loss_all = 0
    n_samples = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = F.nll_loss(model(data), data.y)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()

        n_samples += data.num_graphs
    return loss_all / n_samples


def val(loader):
    model.eval()
    loss_all = 0
    n_samples = 0
    correct = 0
    for data in loader:
        data = data.to(device)
        output = model(data)
        loss_all += F.nll_loss(output, data.y, reduction='sum').item()
        pred = output.max(1)[1]
        correct += pred.eq(data.y).sum().item()
        
        n_samples += data.num_graphs
    return loss_all / n_samples, correct / n_samples


def test(loader):
    model.eval()
    correct = 0
    n_samples = 0
    for data in loader:
        data = data.to(device)
        pred = model(data).max(1)[1]
        correct += pred.eq(data.y).sum().item()

        n_samples += data.num_graphs
    return correct / n_samples


acc = []
for i in range(10):
    print('---------------- Split {} ----------------'.format(i))
    train_index = splits[i]['model_selection'][0]['train']
    val_index = splits[i]['model_selection'][0]['validation']
    test_index = splits[i]['test']

    test_dataset = dataset[test_index]
    val_dataset = dataset[val_index]
    train_dataset = dataset[train_index]

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    model = pi_GNN(dataset.num_features, args.hidden_dim, n_max, args.hidden_nodes, args.n_layers, args.use_dustbins, args.features, args.dropout, dataset.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0
    for epoch in range(1, args.epochs+1):
        train_loss = train(epoch, train_loader, optimizer)
        val_loss, val_acc = val(val_loader)
        if best_val_acc <= val_acc:
            test_acc = test(test_loader)
            best_val_acc = val_acc
        if epoch % 20 == 0:
            print('Epoch: {:03d}, Train Loss: {:.7f}, '
                    'Val Loss: {:.7f}, Test Acc: {:.7f}'.format(
                epoch, train_loss, val_loss, test_acc))
    
    acc.append(test_acc)
acc = torch.tensor(acc)
print('---------------- Final Result ----------------')
print('Mean: {:7f}, Std: {:7f}'.format(acc.mean(), acc.std()))
