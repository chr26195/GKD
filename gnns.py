import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn import GCNConv, SGConv, GATConv, JumpingKnowledge, APPNP, MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import remove_self_loops, add_self_loops, degree
import scipy.sparse
import numpy as np
import math

class SpecialSpmmFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)



class GCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0, simple=False):
        super(GCNLayer, self).__init__()
        self.simple = simple
        if not simple:
            self.W = nn.Parameter(torch.zeros(in_channels, out_channels))
        self.dropout = dropout
        self.specialspmm = SpecialSpmm()
        self.reset_parameters()

    def reset_parameters(self):
        if not self.simple: nn.init.xavier_uniform_(self.W.data, gain=1.414)

    def forward(self, x, edge_index, ff = True):
        if not self.simple and ff: h = torch.matmul(x, self.W)
        else: h = x
        N = h.size(0)

        # weight_mat: hard and differentiable affinity matrix
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=N)
        src, dst = edge_index

        deg = degree(dst, num_nodes=N)
        deg_src = deg[src].pow_(-0.5)
        deg_src.masked_fill_(deg_src == float('inf'), 0)
        deg_dst = deg[dst].pow_(-0.5)
        deg_dst.masked_fill_(deg_dst == float('inf'), 0)
        edge_weight = deg_src * deg_dst

        h_prime = self.specialspmm(edge_index, edge_weight, torch.Size([N, N]), h)
        return h_prime




class GATLayer(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0, alpha=0.2):
        super(GATLayer, self).__init__()

        self.W = nn.Parameter(torch.zeros(in_channels, out_channels))
        self.a = nn.Parameter(torch.zeros(1, out_channels * 2))

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.specialspmm = SpecialSpmm()

        self.dropout = dropout
        self.out_channels = out_channels
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, x, edge_index, edge_weight=None, output_weight=False):
        h = torch.matmul(x, self.W)
        N = h.size(0)
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=N)
        src, dst = edge_index

        # edge_index, _ = remove_self_loops(edge_index)
        # edge_index, _ = add_self_loops(edge_index, num_nodes=N)

        if edge_weight is None:
            edge_h = torch.cat((h[src], h[dst]), dim=-1)  # [E, 2*D]

            edge_e = (edge_h * self.a).sum(dim=-1) # [E]
            edge_e = torch.exp(self.leakyrelu(edge_e))  # [E]
            edge_e = F.dropout(edge_e, p=self.dropout, training=self.training) # [E]
            # e = torch.sparse_coo_tensor(edge_index, edge_e, size=torch.Size([N, N]))
            e_expsum = self.specialspmm(edge_index, edge_e, torch.Size([N, N]), torch.ones(N, 1).to(x.device))
            assert not torch.isnan(e_expsum).any()

            # edge_e_ = F.dropout(edge_e, p=0.8, training=self.training)
            h_prime = self.specialspmm(edge_index, edge_e, torch.Size([N, N]), h)
            h_prime = torch.div(h_prime, e_expsum)  # [N, D] tensor
        else:
            h_prime = self.specialspmm(edge_index, edge_weight, torch.Size([N, N]), h)

        if output_weight:
            edge_expsum = e_expsum[dst].squeeze(1)
            return h_prime, torch.div(edge_e, edge_expsum)
        else:
            return h_prime
