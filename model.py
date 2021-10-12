'''
The log_sinkhorn_iterations() and log_optimal_transport() functions are modified from https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/models/superglue.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn.parameter import Parameter

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
    def __init__(self, input_dim, hidden_dim, n_max, hidden_nodes, alpha, use_dustbins, dropout, n_classes):
        super(pi_GNN, self).__init__()
        self.n_max = n_max
        self.alpha = alpha
        self.use_dustbins = use_dustbins
        self.hidden_nodes = hidden_nodes
        self.has_feats = input_dim > 2
        self.weight1 = Parameter(torch.FloatTensor(hidden_nodes, hidden_dim))
        self.fc1 = nn.Linear(2, hidden_dim)
        if self.has_feats:
            self.weight2 = Parameter(torch.FloatTensor(hidden_nodes, hidden_dim))
            self.fc2 = nn.Linear(input_dim-2, hidden_dim)

        self.ln1 = nn.LayerNorm(hidden_nodes**2)
        self.fc3 = nn.Linear((hidden_nodes**2), 256)
        self.fc4 = nn.Linear(256, 128)

        self.ln2 = nn.LayerNorm(hidden_nodes*input_dim)
        self.fc5 = nn.Linear(hidden_nodes*input_dim, 256)
        self.fc6 = nn.Linear(256, 128)

        self.fc7 = nn.Linear(256, 64)
        self.fc8 = nn.Linear(64, n_classes)

        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=2)

        self.bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.init_weights()

    def init_weights(self):
        self.weight1.data.uniform_(-1, 1)
        if self.has_feats:
            self.weight2.data.uniform_(-1, 1)

    def forward(self, data):
        x, adj = data.x, data.adj
        x = x.view(-1, self.n_max, x.size(1))
        adj = adj.view(-1, self.n_max, self.n_max)

        if self.has_feats:
            struc_feats = x[:,:,-2:]
            feats = x[:,:,:-2]
        else:
            struc_feats = x
        
        struc_feats = self.relu(self.fc1(struc_feats))
        scores = self.relu(torch.einsum("abc,dc->adb", (struc_feats, self.weight1)))
        P_struc_feats = log_optimal_transport(scores, self.use_dustbins, self.bin_score, 100)
        P_struc_feats = torch.exp(P_struc_feats)

        if self.has_feats:
            feats = self.relu(self.fc2(feats))
            scores = self.relu(torch.einsum("abc,dc->adb", (feats, self.weight2)))
            P_feats = log_optimal_transport(scores, self.use_dustbins, self.bin_score, 100)
            P_feats = torch.exp(P_feats)
            
            P = self.alpha*P_struc_feats + (1-self.alpha)*P_feats
        else:
            P = P_struc_feats

        if self.use_dustbins:
            adj_aligned = torch.einsum("abc,acd->abd", (P[:,:-1,:-1], adj))
            adj_aligned = torch.einsum("abc,adc->abd", (adj_aligned, P[:,:-1,:-1]))

            feats_aligned = torch.einsum("abc,acd->abd", (P[:,:-1,:-1], x))
        else:
            adj_aligned = torch.einsum("abc,acd->abd", (P, adj))
            adj_aligned = torch.einsum("abc,adc->abd", (adj_aligned, P))

            feats_aligned = torch.einsum("abc,acd->abd", (P, x))

        adj_aligned = torch.reshape(adj_aligned, (adj_aligned.size(0), -1))
        feats_aligned = torch.reshape(feats_aligned, (feats_aligned.size(0), -1))

        adj_aligned = self.ln1(adj_aligned)
        out_adj = self.relu(self.fc3(adj_aligned))
        out_adj = self.dropout(out_adj)
        out_adj = self.relu(self.fc4(out_adj))

        feats_aligned = self.ln2(feats_aligned)
        out_feats = self.relu(self.fc5(feats_aligned))
        out_feats = self.dropout(out_feats)
        out_feats = self.relu(self.fc6(out_feats))

        out = torch.cat([out_adj, out_feats], dim=1)
        out = self.relu(self.fc7(out))
        out = self.dropout(out)
        out = self.fc8(out)
        return F.log_softmax(out, dim=1)
