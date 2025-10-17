import numpy as np
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
from datetime import *
import sys
from configobj import ConfigObj
import torch.optim as optim
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),'../'))
sys.path.append(parent_dir)
from utils.data import *
from HAAD_ import *
from DLWF_pytorch.model import *
from torch.utils.data import DataLoader,Subset

from tqdm import tqdm
from perturbation import build_model_instance,eval_on_test,split_alpha_number
import os
import argparse
torch.cuda.empty_cache()
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    #调参
    val_ratio = 0.1
    test_ratio = 0.1
    batch_size = 128
    
    data_name, num_classes = split_alpha_number(dataset)
    v = torch.tensor( [71,5,71,6,13,5,42,5,24,5,3,6,30,6,6,4]).reshape(-1,2)
    
    if data_name == "sirinam":
        X_train, y_train, X_valid, y_valid, X_test, y_test = LoadDataNoDefCW(
            input_size=200, num_classes=num_classes, test_ratio=test_ratio, val_ratio=val_ratio)
    elif data_name == "rimmer":
        X_train, y_train, X_valid, y_valid, X_test, y_test = load_rimmer_dataset(
            input_size=200, num_classes=num_classes, test_ratio=test_ratio, val_ratio=val_ratio)
    V_model = build_model_instance(v_model, num_classes, config).to(device)
    V_model.load_state_dict(torch.load(parent_dir + f'/utils/trained_model/{dataset}/{v_model}.pkl'))
    V_model.eval()
    

    train_dataset = MyDataset(X_train,y_train)
    val_dataset = MyDataset(X_valid,y_valid)
    test_dataset = MyDataset(X_test,y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    evaluation_metrics = eval_on_test(V_model, test_loader, v, device, generate_adv_trace)
    acc_before, acc_after, total_cost = evaluation_metrics

   
    with open(f"{s_model}_{v_model}.txt",'a') as f:
        f.write("\n====================\n")
        f.write(f"Dataset:{dataset}\n")
        f.write(f"Verifier Model:{v_model}\n")
        f.write(f"Source Model:{s_model}\n")
        f.write(f"Overhead:{total_cost}\nAccuracy: {acc_before}======>{acc_after}\n")
        f.write("====================\n")
    f.close()
    print(f"done! see {s_model}_{v_model}.txt")

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Train and test a deep neural network (SDAE, CNN or LSTM)')

    parser.add_argument('--model', '-m', default='ensemble', type=str, choices=['cnn','lstm','sdae','varcnn','df','ensemble','awf'],help='choose model type cnn, lstm or sdae')

    parser.add_argument('--verifi_model', '-vm', default='awf', type=str, choices=['cnn','lstm','sdae','varcnn','df','ensemble','awf'],help='choose model type cnn, lstm or sdae')
    parser.add_argument('--dataset', '-d', default='sirinam95', type=str,
                       help='Dataset name')
    args = parser.parse_args()
    s_model = args.model
    v_model = args.verifi_model
    dataset = args.dataset
    config = ConfigObj(parent_dir+"/DLWF_pytorch/My_tor.conf")
    main()