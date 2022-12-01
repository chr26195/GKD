import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import num_nodes, to_dense_adj
from gnns import *
import numpy as np
from kernels import Kernel

class GeoDist(nn.Module):
    '''
    in_channels: number of features 
    hidden_channels: hidden size
    '''
    def __init__(self, in_channels, hidden_channels, out_channels, args, num_node, num_layers=2,
                 use_bn=False):
        super(GeoDist, self).__init__()
        self.args = args
        self.dropout = args.dropout
        self.num_node = num_node
        self.k = Kernel(hidden_channels, out_channels, args, num_node)

        self.teacher_gnn = nn.ModuleList([nn.Linear(in_channels, hidden_channels, bias=False)])
        self.student_gnn = nn.ModuleList([nn.Linear(in_channels, hidden_channels, bias=False)])
        self.bns = nn.BatchNorm1d(hidden_channels, eps=1e-10, affine=False, track_running_stats=False)
        self.bns2d = nn.BatchNorm2d(hidden_channels, affine=False, track_running_stats=False)

        if args.base_model == 'gcn':
            self.teacher_gnn.append(GCNLayer(hidden_channels, hidden_channels, dropout=self.dropout, simple=True))
            self.student_gnn.append(GCNLayer(hidden_channels, hidden_channels, dropout=self.dropout, simple=True))
        
            for _ in range(num_layers - 2):
                self.teacher_gnn.append(
                        GCNLayer(hidden_channels, hidden_channels))
                self.student_gnn.append(
                    GCNLayer(hidden_channels, hidden_channels))

            self.teacher_gnn.append(GCNLayer(hidden_channels, out_channels))
            self.student_gnn.append(GCNLayer(hidden_channels, out_channels))
            
        elif args.base_model == 'gat':
            self.teacher_gnn.append(GATLayer(hidden_channels, hidden_channels, dropout=self.dropout, simple=True))
            self.student_gnn.append(GATLayer(hidden_channels, hidden_channels, dropout=self.dropout, simple=True))
            
            for _ in range(num_layers - 2):
                self.teacher_gnn.append(
                        GATLayer(hidden_channels, hidden_channels, dropout=self.dropout))
                self.student_gnn.append(
                    GATLayer(hidden_channels, hidden_channels, dropout=self.dropout))
            
            self.teacher_gnn.append(GATLayer(hidden_channels, out_channels, dropout=self.dropout))
            self.student_gnn.append(GATLayer(hidden_channels, out_channels, dropout=self.dropout))
        
        self.activation = F.relu
        self.use_bn = use_bn

    def reset_parameters(self):
        for conv in self.teacher_gnn:
            conv.reset_parameters()
        for conv in self.student_gnn:
            conv.reset_parameters()

    def forward(self, data_full, data=None, mode='pretrain', dist_mode='label', t=1.0):
        if mode == 'pretrain':
            return self.forward_teacher(data_full)
        elif mode == 'train':  
            return self.forward_student(data_full, data, dist_mode, t)
        else:
            NotImplementedError
    
    def forward_teacher(self, data_full):
        x, edge_index = data_full.graph['node_feat'], data_full.graph['edge_index']
        x = self.teacher_gnn[0](x)
        for i in range(1, len(self.teacher_gnn) - 1):
            x = self.teacher_gnn[i](x, edge_index) # [n, h]
            if self.use_bn: x = self.bns(x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.teacher_gnn[-1](x, edge_index)
        return x
    
    def forward_student(self, data_full, data, dist_mode, t = 1):
        x_full, edge_index_full = data_full.graph['node_feat'], data_full.graph['edge_index']
        xt = xs = x_full # edge missing, node is seen for both teacher and student
        edge_index = data.graph['edge_index']
        
        if self.args.use_batch:
            idx = torch.randperm(len(data.share_node_idx))[:self.args.batch_size]
            share_node_idx = data.share_node_idx[idx]
        else: share_node_idx = data.share_node_idx
        train_idx = data.train_idx
        
        if dist_mode == 'no': 
            return self.no_kd(xs, xt, edge_index, edge_index_full, train_idx, share_node_idx)
        elif dist_mode == 'gkd':
            return self.gkd(xs, xt, edge_index, edge_index_full, train_idx, share_node_idx)
        elif dist_mode == 'pgkd':
            return self.pgkd(xs, xt, edge_index, edge_index_full, train_idx, share_node_idx)
        else:
            NotImplementedError
            
    def inference(self, data, mode='pretrain'):
        x, edge_index = data.graph['node_feat'], data.graph['edge_index']
        if mode == 'pretrain':
            x = self.teacher_gnn[0](x)
            for i in range(1, len(self.teacher_gnn) - 1):
                x = self.teacher_gnn[i](x, edge_index)
                if self.use_bn:
                    if 'share_node_idx' in data.__dict__.keys(): 
                        x[data.share_node_idx] = self.bns(x[data.share_node_idx])
                    else: x = self.bns(x)
                x = self.activation(x)
            x = self.teacher_gnn[-1](x, edge_index)
            return x
        elif mode == 'train':
            share_node_idx = data.share_node_idx
            x = self.student_gnn[0](x)
            for i in range(1, len(self.student_gnn) - 1):
                x = self.student_gnn[i](x, edge_index)
                if self.use_bn: x[share_node_idx] = self.bns(x[share_node_idx])
                x = self.activation(x)
            x = self.student_gnn[-1](x, edge_index)
            return x
        
    def no_kd(self, xs, xt, edge_index, edge_index_full, train_idx, share_node_idx):
        xs = self.student_gnn[0](xs)
        for i in range(1, len(self.student_gnn) - 1):
            xs = self.student_gnn[i](xs, edge_index)
            xs[share_node_idx] = self.bns(xs[share_node_idx])
            xs = self.activation(xs)
        y_logit_s = self.student_gnn[-1](xs, edge_index)
        return y_logit_s
    
    def gkd(self, xs, xt, edge_index, edge_index_full, train_idx, share_node_idx):
        if self.args.delta != 1:
            A = to_dense_adj(edge_index, max_num_nodes=self.num_node).squeeze().fill_diagonal_(1.)
            A = A[share_node_idx, :][:, share_node_idx]
        else: A = None
        
        loss_list = []
        xt = self.teacher_gnn[0](xt)
        xs = self.student_gnn[0](xs)
        
        for i in range(1, len(self.teacher_gnn) - 1):
            xt = self.teacher_gnn[i](xt, edge_index_full)
            xs = self.student_gnn[i](xs, edge_index)
            xt, xs = self.bns(xt), self.bns(xs)
            if self.args.kernel != 'random':
                mt = self.k(xt[share_node_idx], xt[share_node_idx]).detach()
                ms = self.k(xs[share_node_idx], xs[share_node_idx])
            else:
                mt, ms = self.k.random(xt[share_node_idx], xs[share_node_idx])
            loss_list.append(self.k.dist_loss(mt, ms, A))
            xt, xs = self.activation(xt), self.activation(xs)

        y_logit_t = self.teacher_gnn[-1](xt, edge_index_full)
        y_logit_s = self.student_gnn[-1](xs, edge_index)
        if self.args.include_last:
            if self.args.kernel != 'random':
                mt = self.k(y_logit_t[share_node_idx], y_logit_t[share_node_idx]).detach()
                ms = self.k(y_logit_s[share_node_idx], y_logit_s[share_node_idx])
            else:
                mt, ms = self.k.random(y_logit_t[share_node_idx], y_logit_s[share_node_idx])
            loss_list.append(self.k.dist_loss(mt, ms, A))
        gkd_dist_loss = sum(loss_list)/len(loss_list)
        if self.args.use_kd:
            dist_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(y_logit_s[train_idx]/self.args.tau, dim=1), F.softmax(y_logit_t[train_idx].detach()/self.args.tau, dim=1))
            return y_logit_s, gkd_dist_loss, dist_loss
        else:
            return y_logit_s, gkd_dist_loss

    def pgkd(self, xs, xt, edge_index, edge_index_full, train_idx, share_node_idx):
        if self.args.delta != 1:
            A = to_dense_adj(edge_index, max_num_nodes=self.num_node).squeeze().fill_diagonal_(1.)
            A = A[share_node_idx, :][:, share_node_idx]
        else: A = None
        
        xt = self.teacher_gnn[0](xt)
        xs = self.student_gnn[0](xs)    
        xt1, xs1 = xt[share_node_idx].clone(), xs[share_node_idx].clone()
            
        for i in range(1, len(self.teacher_gnn) - 1):
            xt = self.teacher_gnn[i](xt, edge_index_full)
            xs = self.student_gnn[i](xs, edge_index)
            xt, xs = self.bns(xt), self.bns(xs)
            xt, xs = self.activation(xt), self.activation(xs)

        y_logit_t = self.teacher_gnn[-1](xt, edge_index_full, ff=False)
        y_logit_s = self.student_gnn[-1](xs, edge_index, ff=False)
        
        y_logit_t0 = y_logit_t[share_node_idx]
        y_logit_s0 = y_logit_s[share_node_idx]
        
        mt, ms = self.k.parametric(y_logit_t0.detach(), y_logit_s0, detach=True)
        gkd_dist_loss = self.k.dist_loss(mt, ms, A)
        
        mt2, ms2 = self.k.parametric(y_logit_t0, y_logit_s0)
        rec_loss =  self.k.rec_loss(y_logit_t0.detach(), xt1.detach(), mt2) 
        rec_loss += self.k.rec_loss(y_logit_s0.detach(), xs1.detach(), ms2) 
        
        y_logit_t_, y_logit_s_ = torch.matmul(y_logit_t, self.teacher_gnn[-1].W), torch.matmul(y_logit_s, self.student_gnn[-1].W)
        if self.args.use_kd:
            dist_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(y_logit_s[train_idx]/self.args.tau, dim=1), F.softmax(y_logit_t[train_idx].detach()/self.args.tau, dim=1))
            return y_logit_s_, gkd_dist_loss, rec_loss, dist_loss
        else:
            return y_logit_s_, gkd_dist_loss, rec_loss