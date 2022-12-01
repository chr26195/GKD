from models import *
from data_utils import normalize


def parse_method(args, dataset, n, c, d, device):
    model = GeoDist(d, args.hidden_channels, c, args, n, num_layers=args.num_layers, use_bn=args.use_bn).to(device)
    return model


def parser_add_main_args(parser):
    # setup and protocol
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--sub_dataset', type=str, default='')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train_prop', type=float, default=.5,
                        help='training label proportion')
    parser.add_argument('--valid_prop', type=float, default=.25,
                        help='validation label proportion')

                        
    parser.add_argument('--protocol', type=str, default='semi',
                        help='protocol for cora datasets, semi or supervised')
    parser.add_argument('--rand_split', action='store_true', help='use random splits')
    parser.add_argument('--metric', type=str, default='acc', choices=['acc', 'rocauc', 'f1'],
                        help='evaluation metric')
    parser.add_argument('--runs', type=int, default=5, help='number of distinct runs')
    parser.add_argument('--epochs', type=int, default=100)

    parser.add_argument('--hidden_channels', type=int, default=32)

    parser.add_argument('--num_layers', type=int, default=3,
                        help='number of layers for deep methods')
    parser.add_argument('--gat_heads', type=int, default=8,
                        help='attention heads for gat')
    parser.add_argument('--out_heads', type=int, default=1,
                        help='out heads for gat')
    parser.add_argument('--hops', type=int, default=1,
                        help='power of adjacency matrix for certain methods')
    parser.add_argument('--lp_alpha', type=float, default=.1,
                        help='alpha for label prop')
    parser.add_argument('--gpr_alpha', type=float, default=.1,
                        help='alpha for gprgnn')
    parser.add_argument('--jk_type', type=str, default='max', choices=['max', 'lstm', 'cat'],
                        help='jumping knowledge type')
    parser.add_argument('--directed', action='store_true',
                        help='set to not symmetrize adjacency')
    parser.add_argument('--num_mlp_layers', type=int, default=1,
                        help='number of mlp layers in h2gcn')

    # display and utility
    parser.add_argument('--display_step', type=int,
                        default=5, help='how often to print')
    parser.add_argument('--cached', action='store_true',
                        help='set to use faster sgc')
    parser.add_argument('--print_prop', action='store_true',
                        help='print proportions of predicted class')
    
    parser.add_argument('--priv_type', type=str, choices=['edge', 'node'],
                        default='edge', help='type for privileged information')
    parser.add_argument('--priv_ratio', type=float,
                        default=0.5, help='ratio for privileged nodes/edges')
                        
    parser.add_argument('--save_model', action='store_true', help='save model')
    parser.add_argument('--save_name', type=str, default='gcn', help='saved model name')
    parser.add_argument('--log_name', type=str, default='none', help='log file appendix name')
    
    parser.add_argument('--base_model', type=str, default='gcn', choices=['gcn', 'gat'], help='which model')    
    parser.add_argument('--not_load_teacher', action='store_true', help='whether not load teacher model')
    
    parser.add_argument('--mode', type=str, choices=['pretrain', 'train'],
                        default='pretrain', help='mode for pretrain teacher or train student')
    parser.add_argument('--dist_mode', type=str,
                        default='label', help='mode for knowledge distillation')
    
    # training
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--use_bn', action='store_true', help='use batch norm')
    parser.add_argument('--use_batch', action='store_true')
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--oracle', action='store_true', help='whether using the complete graph for testing')
    parser.add_argument('--device', type=int, default=0, help='which gpu to use if any (default: 0)')

    
    # for distillation loss
    parser.add_argument('--alpha', type=float, default=0.5, help='distillation loss (GKD) weight')
    parser.add_argument('--delta', type=float, default=0.1, help='distillation loss hyperparameter') 

    parser.add_argument('--use_kd', action='store_true', help='whether use vanilla KD loss')
    parser.add_argument('--beta', type=float, default=0.0, help='weight for auxiliary KD loss')
    parser.add_argument('--tau', type=float, default=1.0, help='KD loss temperature') 

    parser.add_argument('--kernel', type=str, default='sigmoid', help='sigmoid/gaussian/random')
    parser.add_argument('--t', type=float, default=1.0, help='hyperparameter for Gauss-Weierstras kernel')
    parser.add_argument('--include_last', action='store_true', help='whether include last layer')
    parser.add_argument('--s', type=int, default=2, help='hyperparameter for random and parametric kernel')
    parser.add_argument('--m', type=int, default=1, help='hyperparameter for random kernel')
    parser.add_argument('--lr2', type=float, default=0.001)

    parser.add_argument('--sim', type=str, default='l2')
    parser.add_argument('--ker_norm', action='store_true')