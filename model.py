'''
The log_sinkhorn_iterations() and log_optimal_transport() functions are modified from https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/models/superglue.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn.parameter import Parameter

from torch_geometric.nn import GCNConv, GINConv

def log_sinkhorn_iterations(Z, log_mu, log_nu, iters):
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores, use_dustbins, alpha, iters):
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores)

    if use_dustbins:
        bins0 = alpha.expand(b, m, 1)
        bins1 = alpha.expand(b, 1, n)
        alpha = alpha.expand(b, 1, 1)
        couplings = torch.cat([torch.cat([scores, bins0], -1), torch.cat([bins1, alpha], -1)], 1)
    else:
        couplings = scores

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])

    if not use_dustbins:
        log_mu, log_nu = norm.expand(m), norm.expand(n)

    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z


class pi_GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_max, hidden_nodes, n_layers, use_dustbins, features, dropout, n_classes):
        super(pi_GNN, self).__init__()
        self.n_max = n_max
        self.n_layers = n_layers
        self.use_dustbins = use_dustbins
        self.hidden_nodes = hidden_nodes
        self.features = features

        if input_dim > 2:
            self.has_features = True
        else:
            self.has_features = False

        self.weight1 = Parameter(torch.FloatTensor(hidden_nodes, hidden_dim))
        self.alpha = Parameter(torch.tensor(.01))

        lst = list()
        if self.features == 'gcn':
            lst.append(GCNConv(input_dim, hidden_dim))
            for i in range(n_layers-1):
                lst.append(GCNConv(hidden_dim, hidden_dim))
            self.mp = nn.ModuleList(lst)
        elif self.features == 'gin':
            lst.append(GINConv(nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())))
            for i in range(n_layers-1):
                lst.append(GINConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())))
            self.mp = nn.ModuleList(lst)
        else:
            self.fc2 = nn.Linear(2, hidden_dim)
            if self.has_features:
                self.weight2 = Parameter(torch.FloatTensor(hidden_nodes, hidden_dim))
                self.fc3 = nn.Linear(input_dim-2, hidden_dim)

        self.ln1 = nn.LayerNorm(hidden_nodes**2)
        self.fc4 = nn.Linear((hidden_nodes**2), 256)

        if self.has_features:
            self.ln2 = nn.LayerNorm(hidden_nodes*(input_dim-2))
            self.fc5 = nn.Linear(hidden_nodes*(input_dim-2), 256)
            self.fc6 = nn.Linear(512, 64)
        else:
            self.fc6 = nn.Linear(256, 64)
        self.fc7 = nn.Linear(64, n_classes)

        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=2)

        self.bin_score = Parameter(torch.tensor(1.))
        self.init_weights()

    def init_weights(self):
        self.weight1.data.uniform_(-1, 1)
        if self.has_features and self.features == 'raw':
            self.weight2.data.uniform_(-1, 1)

    def forward(self, data):
        x, adj, edge_index = data.x, data.adj, data.edge_index
        #x = x.view(-1, self.n_max, x.size(1))
        adj = adj.view(-1, self.n_max, self.n_max)
        if self.features == 'raw':
            struc_feats = x[:,:2]
        else:
            struc_feats = x

        if self.features == 'gcn' or self.features == 'gin':
            for i in range(self.n_layers):
                struc_feats = self.relu(self.mp[i](struc_feats, edge_index))
                struc_feats = self.dropout(struc_feats)
        else:
            struc_feats = self.relu(self.fc2(struc_feats))

        struc_feats = struc_feats.view(-1, self.n_max, struc_feats.size(1))
        scores = self.relu(torch.einsum("abc,dc->adb", (struc_feats, self.weight1)))
        P_struc_feats = log_optimal_transport(scores, self.use_dustbins, self.bin_score, 100)
        P_struc_feats = torch.exp(P_struc_feats)

        if self.has_features and self.features == 'raw':
            feats = x[:,2:]
            feats = self.relu(self.fc3(feats))
            feats = feats.view(-1, self.n_max, feats.size(1))
            scores = self.relu(torch.einsum("abc,dc->adb", (feats, self.weight2)))
            P_feats = log_optimal_transport(scores, self.use_dustbins, self.bin_score, 100)
            P_feats = torch.exp(P_feats)
            
            alpha = self.sigmoid(self.alpha)
            P = alpha*P_struc_feats + (1-alpha)*P_feats
        else:
            P = P_struc_feats

        if self.use_dustbins:
            adj_aligned = torch.einsum("abc,acd->abd", (P[:,:-1,:-1], adj))
            adj_aligned = torch.einsum("abc,adc->abd", (adj_aligned, P[:,:-1,:-1]))

            if self.has_features:
                feats_aligned = torch.einsum("abc,acd->abd", (P[:,:-1,:-1], x[:,2:].view(-1, self.n_max, x[:,2:].size(1))))
        else:
            adj_aligned = torch.einsum("abc,acd->abd", (P, adj))
            adj_aligned = torch.einsum("abc,adc->abd", (adj_aligned, P))

            if self.has_features:
                feats_aligned = torch.einsum("abc,acd->abd", (P, x[:,2:].view(-1, self.n_max, x[:,2:].size(1))))

        adj_aligned = torch.reshape(adj_aligned, (adj_aligned.size(0), -1))
        adj_aligned = self.ln1(adj_aligned)
        out_adj = self.relu(self.fc4(adj_aligned))
        out_adj = self.dropout(out_adj)
    

        if self.has_features:
            feats_aligned = torch.reshape(feats_aligned, (feats_aligned.size(0), -1))
            feats_aligned = self.ln2(feats_aligned)
            out_feats = self.relu(self.fc5(feats_aligned))
            out_feats = self.dropout(out_feats)
            out = torch.cat([out_adj, out_feats], dim=1)
        else:
            out = out_adj
        
        out = self.relu(self.fc6(out))
        out = self.dropout(out)
        out = self.fc7(out)
        return F.log_softmax(out, dim=1)
