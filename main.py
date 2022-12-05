import argparse
import sys
import os, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torch.utils import data
from torch_geometric.utils import to_undirected, subgraph, add_remaining_self_loops, add_self_loops
from torch_scatter import scatter

from logger import Logger, SimpleLogger
from dataset import load_nc_dataset
from data_utils import normalize, gen_normalized_adjs, evaluate, eval_acc, eval_rocauc, eval_f1, to_sparse_tensor, \
    load_fixed_splits, remove_edges
from parse import parse_method, parser_add_main_args

import copy

torch.autograd.set_detect_anomaly(True)

# NOTE: data splits are consistent given fixed seed, see data_utils.rand_train_test_idx
def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

### Parse args ###
parser = argparse.ArgumentParser(description='General Training Pipeline')
parser_add_main_args(parser)
args = parser.parse_args()
print(args)

fix_seed(args.seed)

if args.cpu:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
print(device)


### Load and preprocess data ###
dataset = load_nc_dataset(args.dataset, args.sub_dataset, args.data_dir)

if len(dataset.label.shape) == 1:
    dataset.label = dataset.label.unsqueeze(1)

print(dataset.label.shape)
dataset.label = dataset.label.to(device)

# get the splits for all runs
if args.rand_split:
    split_idx_lst = [dataset.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop)
                     for _ in range(args.runs)]
elif args.dataset in ['ogbn-proteins', 'ogbn-arxiv', 'ogbn-products']:
    split_idx_lst = [dataset.load_fixed_splits()
                     for _ in range(args.runs)]
else:
    split_idx_lst = load_fixed_splits(dataset, name=args.dataset, protocol=args.protocol)

if args.dataset == 'ogbn-proteins':
    if args.method == 'mlp' or args.method == 'cs':
        dataset.graph['node_feat'] = scatter(dataset.graph['edge_feat'], dataset.graph['edge_index'][0],
                                             dim=0, dim_size=dataset.graph['num_nodes'], reduce='mean')
    else:
        dataset.graph['edge_index'] = to_sparse_tensor(dataset.graph['edge_index'],
                                                       dataset.graph['edge_feat'], dataset.graph['num_nodes'])
        dataset.graph['node_feat'] = dataset.graph['edge_index'].mean(dim=1)
        dataset.graph['edge_index'].set_value_(None)
    dataset.graph['edge_feat'] = None

n = dataset.graph['num_nodes']
# infer the number of classes for non one-hot and one-hot labels
c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
d = dataset.graph['node_feat'].shape[1]

# whether or not to symmetrize
if not args.directed and args.dataset != 'ogbn-proteins':
    dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])
edge_index_directed = dataset.graph['edge_index'][:, dataset.graph['edge_index'][1,:] >= dataset.graph['edge_index'][0,:] ]                          
edge_index_directed = edge_index_directed.to(device)


print(f"num nodes {n} | num classes {c} | num node feats {d}")

### Load method ###
model = parse_method(args, dataset, n, c, d, device)

# using rocauc as the eval function
if args.dataset in ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins'):
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.NLLLoss()
if args.metric == 'rocauc':
    eval_func = eval_rocauc
elif args.metric == 'f1':
    eval_func = eval_f1
else:
    eval_func = eval_acc

logger = Logger(args.runs, args)
model.train()
print('MODEL:', model)
dataset.graph['edge_index'], dataset.graph['node_feat'] = \
    dataset.graph['edge_index'].to(device), dataset.graph['node_feat'].to(device)

if args.dataset in ('yelp-chi', 'deezer-europe', 'fb100', 'twitch-e', 'ogbn-proteins'):
    if dataset.label.shape[1] == 1:
        dataset.label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)

dataset_mask = copy.deepcopy(dataset)

### Training loop ###
for run in range(args.runs):
    if args.dataset in ['cora', 'citeseer', 'pubmed'] and args.protocol == 'semi':
        split_idx = split_idx_lst[0]
    else:
        split_idx = split_idx_lst[run]
    train_idx = split_idx['train'].to(device)
    dataset.train_idx = train_idx

    # Processing for privileged information in each run
    if args.priv_type == 'edge':
        num = int(edge_index_directed.size(1) * (1 - args.priv_ratio)) # priv_ratio: information loss ratio
        idx = torch.randperm(edge_index_directed.size(1))[:num]
        edge_index_share = edge_index_directed[:, idx]
        try: dataset_mask.graph['edge_index'] = to_undirected(edge_index_share) 
        except: dataset_mask.graph['edge_index'] = edge_index_share
        dataset_mask.train_idx = train_idx
        dataset_mask.share_node_idx = torch.cat([train_idx, split_idx['valid'].to(device), split_idx['test'].to(device)], dim=-1)

    elif args.priv_type == 'node':
        train_num = train_idx.shape[0]
        num = int((1 - args.priv_ratio) * train_num) # removing certain ratio of train nodes on training node
        assert num < train_num 
        share_train_idx = train_idx[torch.randperm(train_num)[:num]]
        share_node_idx = torch.cat([share_train_idx, split_idx['valid'].to(device), split_idx['test'].to(device)], dim=-1)        
        dataset_mask.graph['edge_index'] = subgraph(share_node_idx, dataset.graph['edge_index'])[0]
        dataset_mask.train_idx = share_train_idx
        dataset_mask.share_node_idx = share_node_idx
    else:
        raise NotImplementedError
    
    model.reset_parameters()

    if args.mode == 'train': # loading teacher model
        model_dir = f'saved_models/{args.base_model}_{args.dataset}_{run}.pkl'
        if not os.path.exists(model_dir):
            raise FileNotFoundError
        else:
            model_dict = torch.load(model_dir)
            if not args.not_load_teacher:
                model.teacher_gnn.load_state_dict(model_dict)

    optimizer_te = torch.optim.Adam([{'params': model.teacher_gnn.parameters()}], lr=args.lr, weight_decay=args.weight_decay)
    optimizer_st = torch.optim.Adam([{'params': model.student_gnn.parameters()}], lr=args.lr, weight_decay=args.weight_decay)
    if args.dist_mode == 'pgkd': optimizer_k = torch.optim.Adam([{'params': model.k.parameters()}], lr=args.lr2, weight_decay=args.weight_decay)

    best_val = float('-inf')
    for epoch in range(args.epochs):
        model.train()
        train_start = time.time()
        if args.mode == 'pretrain':
            optimizer_te.zero_grad()
            out = model(dataset, mode='pretrain')
            if args.dataset in ('yelp-chi', 'deezer-europe', 'fb100', 'twitch-e', 'ogbn-proteins'): # binary classification
                loss = criterion(out[train_idx], dataset.label.squeeze(1)[train_idx].to(torch.float))
            else:
                out = F.log_softmax(out, dim=1)
                loss = criterion(out[train_idx], dataset.label.squeeze(1)[train_idx])
            loss.backward()
            optimizer_te.step()
            
        elif args.mode == 'train' and args.dist_mode != 'pgkd':
            optimizer_st.zero_grad()
            outputs = model(dataset, dataset_mask, mode='train', dist_mode=args.dist_mode, t=args.t)
            out = outputs[0] if type(outputs) == tuple else outputs

            if args.dataset in ('yelp-chi', 'deezer-europe', 'fb100', 'twitch-e', 'ogbn-proteins'):
                sup_loss = criterion(out[dataset_mask.train_idx], dataset_mask.label.squeeze(1)[dataset_mask.train_idx].to(torch.float))
            else:
                out = F.log_softmax(out, dim=1)
                sup_loss = criterion(out[dataset_mask.train_idx], dataset_mask.label.squeeze(1)[dataset_mask.train_idx])

            if args.dist_mode == 'no': loss = sup_loss
            elif args.dist_mode == 'gkd' and not args.use_kd:
                loss = (1 - args.alpha) * sup_loss + args.beta * outputs[1] * args.t * args.t
            elif args.dist_mode == 'gkd' and args.use_kd:
                loss = (1 - args.alpha) * sup_loss + args.alpha * outputs[1] + args.beta * outputs[2] * args.t * args.t
    
            loss.backward()
            optimizer_st.step()
            
        elif args.mode == 'train' and args.dist_mode == 'pgkd':
            outputs = model(dataset, dataset_mask, mode='train', dist_mode=args.dist_mode, t=args.t)
            out = outputs[0] if type(outputs) == tuple else outputs

            if args.dataset in ('yelp-chi', 'deezer-europe', 'fb100', 'twitch-e', 'ogbn-proteins'):
                sup_loss = criterion(out[dataset_mask.train_idx], dataset_mask.label.squeeze(1)[dataset_mask.train_idx].to(torch.float))
            else:
                out = F.log_softmax(out, dim=1)
                sup_loss = criterion(out[dataset_mask.train_idx], dataset_mask.label.squeeze(1)[dataset_mask.train_idx])

            if not args.use_kd:
                loss = (1 - args.alpha) * sup_loss + args.alpha * outputs[1]
            else:
                loss = (1 - args.alpha) * sup_loss + args.alpha * outputs[1] + args.beta * outputs[3] * args.t * args.t
                
            optimizer_k.zero_grad()            
            rec_loss = outputs[2]
            rec_loss.backward(retain_graph=True)
            optimizer_k.step()    
            
            optimizer_st.zero_grad()            
            loss.backward()
            optimizer_st.step()
            
        train_time = time.time() - train_start

        if args.mode == 'pretrain':
            if args.oracle:
                result = evaluate(model, dataset, split_idx, eval_func, criterion, args, test_dataset=dataset) 
            else:
                result = evaluate(model, dataset, split_idx, eval_func, criterion, args, test_dataset=dataset_mask) 
        elif args.mode == 'train':
            result = evaluate(model, dataset_mask, split_idx, eval_func, criterion, args)

        
        logger.add_result(run, result[:-1])
        if result[1] > best_val:
            best_val = result[1]
            if args.dataset != 'ogbn-proteins':
                best_out = F.softmax(result[-1], dim=1)
            else:
                best_out = result[-1]
            if args.mode == 'pretrain' and args.save_model:
                torch.save(model.teacher_gnn.state_dict(), f'saved_models/{args.base_model}_{args.dataset}_{run}.pkl')

        if epoch % args.display_step == 0:
            print(f'Epoch: {epoch:02d}, '
                  f'Loss: {loss:.4f}, '
                  f'Train: {100 * result[0]:.2f}%, '
                  f'Valid: {100 * result[1]:.2f}%, '
                  f'Test: {100 * result[2]:.2f}%')
            if args.print_prop:
                pred = out.argmax(dim=-1, keepdim=True)
                print("Predicted proportions:", pred.unique(return_counts=True)[1].float() / pred.shape[0])
    
    results = logger.print_statistics(run)

results = logger.print_statistics()

# ### Save results ###
filename = f'logs/{args.dataset}_{args.priv_type}.csv'
print(f"Saving results to {filename}")
with open(f"{filename}", 'a+') as write_obj:
    sub_dataset = f'{args.sub_dataset},' if args.sub_dataset else ''
    write_obj.write(f"data({args.dataset},{args.priv_type}{args.priv_ratio}), model({args.log_name},{args.base_model},{args.dist_mode}),\
        \t lr({args.lr}), wd({args.weight_decay}), alpha({args.alpha}), t({args.t}), dt({args.delta}) \t")
    write_obj.write("perf: {} $\pm$ {}\n".format(format(results.mean(), '.2f'), format(results.std(), '.2f')))