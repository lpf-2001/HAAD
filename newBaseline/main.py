import torch
import numpy as np
import os 
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),"../"))
from DLWF_pytorch.train_utils import *
from utils.data import *
import argparse
import time

import os


def parse_arguments():
    def restricted_float(x):
        try:
            x = float(x)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{x} is not a valid float value.")

        if x < 0 or x > 1:
            raise argparse.ArgumentTypeError(f"{x} is not in the range [0, 1].")
        return x

    parser = argparse.ArgumentParser(description="GAPDiS parameters.")
    # training args
    parser.add_argument("--WF_model_name", type=str, default='awf', help="Default WF model [DF, AWF, VarCNN]")
    parser.add_argument("--dataset_name", type=str, default='sirinam95', help="Default dataset_name")
    parser.add_argument("--data_str", type=str, default='data', help="data key in dataset")
    parser.add_argument("--label_str", type=str, default='labels', help="label key in dataset")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for the dataset.")

    # perturbation generation args
    parser.add_argument("--max_len_perturbations", type=int, default=128, help="M, the allow max len of perturbations, range [1, 4999]")
    parser.add_argument("--max_iter", type=int, default=None, help="Maximum number of iterations. If None it be max_iter_multi * max_len_perturbations")
    parser.add_argument("--target_acc", type=restricted_float,
                        default=0, help="Target accuracy threshold, must be in the range [0, 1].")
    parser.add_argument("--topk_num", type=int, default=10, help="Number of top-k solutions to consider, range [1, 5000-M].")
    parser.add_argument("--m_max", type=int, default=8, help="Maximum number of dummy packets to insert in a delta_{i}.")
    parser.add_argument("--tabu_len_multi", type=int, default=5, help="Multiplier for tabu list length.")
    parser.add_argument("--sol_len_multi", type=int, default=4, help="Multiplier for solution length.")
    parser.add_argument("--cpm_len_den", type=int, default=2, help="Denominator for candidate pool length of critical position.")
    parser.add_argument("--init_rd_num", type=int, default=5, help="Number of initial random solutions.")
    parser.add_argument("--init_m_num", type=int, default=8, help="Number of initial dummy packets (for more solusion search filed).")
    parser.add_argument("--exch_len", type=int, default=16, help="Length of exchange operations for random replace.")
    parser.add_argument("--repl_rate", type=float, default=0.1, help="Replacement rate for solutions, [0, 1].")
    parser.add_argument("--muta_rate", type=float, default=0.2, help="Mutation rate for solutions, [0, 1].")
    parser.add_argument("--smp_cpm_rate", type=float, default=0.2, help="Sampling rate for candidate pool, [0, 1].")
    parser.add_argument("--toler", type=int, default=30, help="Tolerance for early stopping, [0, 100].")
    parser.add_argument("--max_iter_multi", type=int, default=8, help="Multiplier for maximum iterations, [0, 1].")
    parser.add_argument("--device", type=str, default='cuda', help="Device to use: 'cuda' or 'cpu'")

    args = parser.parse_args()
    return args


def load_dataset(dataset_path, data_str='data', label_str='labels'):
    train_dataset = np.load(dataset_path, allow_pickle=True)
    x = train_dataset[data_str]
    y = train_dataset[label_str]
    return x, y


def running_front(args):
    from FRONT import DirectionalFRONT
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # read in dataset
    if args.dataset_name == 'rimmer100':
        X_train, y_train, X_valid, y_valid, X_test, y_test = load_rimmer_dataset(
        input_size=200, num_classes=100, test_ratio=0.25, val_ratio=0.742)
    else:
        X_train, y_train, X_valid, y_valid, X_test, y_test = LoadDataNoDefCW(
        input_size=200, num_classes=95, test_ratio=0.25, val_ratio=0.1)
    
    # read in WF model
    config = ConfigObj('../DLWF_pytorch/My_tor.conf')
    learn_param = config[args.WF_model_name]
    wf_model = build_model_instance(args.WF_model_name, args.dataset_name, config=learn_param).to(device)
    wf_model.load_state_dict(torch.load(f'../utils/trained_model/{args.dataset_name}/{args.WF_model_name}.pkl'))
    start_time = time.time()
    # generate perturbations
    start_time = time.time()
    front = DirectionalFRONT(
        wf_model, 
        Nc=30, 
        Ns=1, 
        Wmin=0.1, 
        Wmax=0.2,
        device=device
    )  # Wmin=[0.01, 0.05 0.1 0.15 0.2], Wmax=[0.1 0.2 0.3 0.4]
    # AWF: Wmin=0.1 Wmax=0.1
    # DF: Wmin=0.1 Wmax=0.2
    

    print(f"Total running time: {time.time() - start_time:.2f} seconds")
    front.eval_performance(X_test.squeeze(), y_test.argmax(-1), batch_size=args.batch_size)
    return


def running_WalkieTalkie(args):
    from WalkieTalkie import WalkieTalkie
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # read in dataset
    config = ConfigObj('../DLWF_pytorch/My_tor.conf')
    learn_param = config[args.WF_model_name]
    if args.dataset_name == 'rimmer100':
        train_x, train_y, valid_x, valid_y, test_x, test_y = load_rimmer_dataset(
        input_size=200, num_classes=100, test_ratio=0.25, val_ratio=0.742)
    else:
        train_x, train_y, valid_x, valid_y, test_x, test_y = LoadDataNoDefCW(
        input_size=200, num_classes=95, test_ratio=0.25, val_ratio=0.1)

    # read in WF model
    wf_model = build_model_instance(args.WF_model_name, args.dataset_name, config=learn_param).to(device)
    wf_model.load_state_dict(torch.load(f'../utils/trained_model/{args.dataset_name}/{args.WF_model_name}.pkl'))
    start_time = time.time()
    wt = WalkieTalkie(wf_model, device=device)
    wt.eval_performance(test_x.squeeze(), test_y.argmax(-1), train_x.squeeze(), train_y.argmax(-1), batch_size=args.batch_size)  # train_x, train_y仅用于生成扰动
    print(f"Total running time: {time.time() - start_time:.2f} seconds")
    return


if __name__ == "__main__":
    import os
    set_seed(2025)
    print('Current path:', os.getcwd().replace('\\', '/'))
    args = parse_arguments()
    # running_gapdis_pytorch(args, perturbations=None)
    running_front(args)
    running_WalkieTalkie(args)