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
    def __init__(self, input_dim, hidden_dim, n_max, hidden_nodes, use_dustbins, dropout, n_classes):
        super(pi_GNN, self).__init__()
        self.n_max = n_max
        self.use_dustbins = use_dustbins
        self.hidden_nodes = hidden_nodes
        self.weight = Parameter(torch.FloatTensor(hidden_nodes, hidden_dim))
        self.fc1 = nn.Linear(input_dim, hidden_dim)

        self.ln1 = nn.LayerNorm(hidden_nodes**2)
        self.fc2 = nn.Linear((hidden_nodes**2), 256)
        self.fc3 = nn.Linear(256, 128)

        self.ln2 = nn.LayerNorm(hidden_nodes*input_dim)
        self.fc4 = nn.Linear(hidden_nodes*input_dim, 256)
        self.fc5 = nn.Linear(256, 128)

        self.fc6 = nn.Linear(256, 64)
        self.fc7 = nn.Linear(64, n_classes)

        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=2)

        self.bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.init_weights()

    def init_weights(self):
        self.weight.data.uniform_(-1, 1)

    def forward(self, data):
        x, adj = data.x, data.adj
        features = x.view(-1, self.n_max, x.size(1))
        adj = data.adj
        adj = adj.view(-1, self.n_max, self.n_max)

        x = self.relu(self.fc1(features))
        scores = self.relu(torch.einsum("abc,dc->adb", (x, self.weight)))
        P = log_optimal_transport(scores, self.use_dustbins, self.bin_score, 100)
        P = torch.exp(P)

        if self.use_dustbins:
            adj_aligned = torch.einsum("abc,acd->abd", (P[:,:-1,:-1], adj))
            adj_aligned = torch.einsum("abc,adc->abd", (adj_aligned, P[:,:-1,:-1]))

            feats_aligned = torch.einsum("abc,acd->abd", (P[:,:-1,:-1], features))
        else:
            adj_aligned = torch.einsum("abc,acd->abd", (P, adj))
            adj_aligned = torch.einsum("abc,adc->abd", (adj_aligned, P))

            feats_aligned = torch.einsum("abc,acd->abd", (P, features))

        adj_aligned = torch.reshape(adj_aligned, (adj_aligned.size(0), -1))
        feats_aligned = torch.reshape(feats_aligned, (feats_aligned.size(0), -1))

        adj_aligned = self.ln1(adj_aligned)
        out_adj = self.relu(self.fc2(adj_aligned))
        out_adj = self.relu(self.fc3(out_adj))

        feats_aligned = self.ln2(feats_aligned)
        out_feats = self.relu(self.fc4(feats_aligned))
        out_feats = self.relu(self.fc5(out_feats))

        out = torch.cat([out_adj, out_feats], dim=1)
        out = self.relu(self.fc6(out))
        out = self.fc7(out)
        return F.log_softmax(out, dim=1)
