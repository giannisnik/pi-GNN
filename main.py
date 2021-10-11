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
    def __init__(self, cat=True):
        self.cat = cat

    def __call__(self, data):
        x = data.x
        G = to_networkx(data, to_undirected=True)
        degs = [G.degree(v) for v in G.nodes()]
        triangles = [nx.triangles(G,v) for v in G.nodes()]
        degs = torch.FloatTensor(degs).unsqueeze(1)
        triangles = torch.FloatTensor(triangles).unsqueeze(1)
        feats = torch.cat([degs,triangles], dim=-1)

        if x is not None and self.cat:
            x = x.view(-1, 1) if x.dim() == 1 else x
            data.x = torch.cat([x, feats.to(x.dtype)], dim=-1)
        else:
            data.x = feats

        return data


class MyTransform(object):
    def __init__(self, n_max):
        self.n_max = n_max

    def __call__(self, data):
        x, edge_index = data.x, data.edge_index
        features = torch.zeros(self.n_max, x.size(1))
        adj = torch.zeros(self.n_max, self.n_max)
        features[:x.size(0),:] = x
        adj[edge_index[0,:], edge_index[1,:]] = 1
        data.x = features
        data.adj = adj
        return data


# Argument parser
parser = argparse.ArgumentParser(description='pi_GNN')
parser.add_argument('--dataset', default='MUTAG', help='Dataset name')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', help='Initial learning rate')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='Batch size')
parser.add_argument('--epochs', type=int, default=300, metavar='N', help='Number of epochs to train')
parser.add_argument('--hidden-nodes', type=int, default=20, metavar='N', help='Number of latent nodes')
parser.add_argument('--hidden-dim', type=int, default=64, metavar='N', help='Size of hidden layer')
parser.add_argument('--use-dustbins', action='store_true', default=False, help='Whether to use dustbins')
args = parser.parse_args()

use_node_attr = False
if args.dataset == 'ENZYMES' or args.dataset == 'PROTEINS_full':
    use_node_attr = True

n_max = size_largest_graph[args.dataset]

dataset = TUDataset(root='./datasets/'+args.dataset, name=args.dataset, use_node_attr=use_node_attr, transform=T.Compose([Features(), MyTransform(n_max)]))

with open('data_splits/'+args.dataset+'_splits.json','rt') as f:
    for line in f:
        splits = json.loads(line)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def train(epoch, loader, optimizer):
    model.train()
    loss_all = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = F.nll_loss(model(data), data.y)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(loader.dataset)


def val(loader):
    model.eval()
    loss_all = 0

    for data in loader:
        data = data.to(device)
        loss_all += F.nll_loss(model(data), data.y, reduction='sum').item()
    return loss_all / len(loader.dataset)


def test(loader):
    model.eval()
    correct = 0

    for data in loader:
        data = data.to(device)
        pred = model(data).max(1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)

acc = []
for i in range(10):
    model = pi_GNN(dataset.num_features, args.hidden_dim, n_max, args.hidden_nodes, args.use_dustbins, args.dropout, dataset.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_index = splits[i]['model_selection'][0]['train']
    val_index = splits[i]['model_selection'][0]['validation']
    test_index = splits[i]['test']

    test_dataset = dataset[val_index]
    val_dataset = dataset[val_index]
    train_dataset = dataset[train_index]

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    print('---------------- Split {} ----------------'.format(i))

    best_val_loss, test_acc = 100, 0
    for epoch in range(1, args.epochs+1):
        train_loss = train(epoch, train_loader, optimizer)
        val_loss = val(val_loader)
        if best_val_loss >= val_loss:
            test_acc = test(test_loader)
            best_val_loss = val_loss
        if epoch % 20 == 0:
            print('Epoch: {:03d}, Train Loss: {:.7f}, '
                    'Val Loss: {:.7f}, Test Acc: {:.7f}'.format(
                    epoch, train_loss, val_loss, test_acc))
    acc.append(test_acc)
acc = torch.tensor(acc)
print('---------------- Final Result ----------------')
print('Mean: {:7f}, Std: {:7f}'.format(acc.mean(), acc.std()))
