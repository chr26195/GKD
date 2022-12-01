import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import num_nodes, to_dense_adj
from gnns import *
import numpy as np

class Kernel(nn.Module):
    def __init__(self, hidden_channels, out_channels, args, num_node):
        super(Kernel, self).__init__()
        self.args = args
        self.hidden_channels = hidden_channels
        self.num_node = num_node
        self.bns = nn.BatchNorm1d(hidden_channels, eps=1e-10, affine=False, track_running_stats=False)
        if self.args.dist_mode == 'pgkd':
            self.phi = nn.Parameter(torch.randn(hidden_channels, hidden_channels * self.args.s))
    
    def normalize(self, mat):
        norm = mat.diagonal().sqrt()
        norm_mat = torch.outer(norm, norm)
        mat = mat / norm_mat
        return mat

    def forward(self, x, y):
        if self.args.kernel == 'sigmoid':
            mat = nn.Tanh()(x @ y.transpose(-1, -2))
        elif self.args.kernel == 'gaussian':
            mat = torch.cdist(x, y, p=2) 
            mat = (-mat/self.args.t).exp()
        return self.normalize(mat) if self.args.ker_norm else mat

    def random(self, xt, xs): # n * d
        d = xt.shape[-1]
        W = torch.randn(self.args.m, d, d * self.args.s).to(xs.device)

        xt_ = torch.einsum('nd,adk->ank', xt, W).tanh() 
        xs_ = torch.einsum('nd,adk->ank', xs, W).tanh()
        te_mat = torch.einsum('aij,akj->aik', xt_, xt_) 
        st_mat = torch.einsum('aij,akj->aik', xs_, xs_) 

        e = torch.randn(self.args.m).to(xt_.device)
        te_mat = torch.einsum('a,aij->ij', e, te_mat)
        st_mat = torch.einsum('a,aij->ij', e, st_mat)
        return te_mat.detach(), st_mat

    def parametric(self, xt, xs, detach = False): 
        if detach:
            xt_ = (xt @ self.phi.detach().clone()).tanh() 
            xs_ = (xs @ self.phi.detach().clone()).tanh() 
        else:
            xt_ = (xt @ self.phi).tanh() 
            xs_ = (xs @ self.phi).tanh()
        te_mat = xt_ @ xt_.transpose(-1, -2)
        st_mat = xs_ @ xs_.transpose(-1, -2)
        return te_mat, st_mat

    def dist_loss(self, mt, ms, A = None):
        if A == None or self.args.delta == 1.0: 
            return nn.MSELoss(reduction='sum')(mt, ms)
        else: 
            return (nn.MSELoss(reduction='none')(mt, ms) * (A + (1-A) * self.args.delta)).sum()
    
    def rec_loss(self, x2, x1, m):
        return nn.MSELoss(reduction='sum')(self.bns(m @ x2.detach()), self.bns(x1.detach()))