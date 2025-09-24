import numpy as np
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
import json
import sys
from configobj import ConfigObj
import torch.optim as optim
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),'../'))
sys.path.append(parent_dir)
from utils.data import *
from Ant_algorithm import *
from DLWF_pytorch.model import *
from torch.utils.data import DataLoader,Subset
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from tqdm import tqdm
import os
import argparse
torch.cuda.empty_cache()
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
parser = argparse.ArgumentParser(description='Train and test a deep neural network (SDAE, CNN or LSTM)')

parser.add_argument('--s_model', '-sm', default='ensemble', type=str, choices=['cnn','lstm','sdae','varcnn','df','ensemble'],help='choose model type cnn, lstm or sdae')

parser.add_argument('--v_model', '-vm', default='df', type=str, choices=['cnn','lstm','sdae','varcnn','df','ensemble'],help='choose model type cnn, lstm or sdae')
parser.add_argument('--patch_start', '-ps', default=8, type=int)
parser.add_argument('--patch_end', '-pe', default=9, type=int)
args = parser.parse_args()

config = ConfigObj("/home/xuke/lpf/all/DLWF_pytorch/My_tor.conf")
criterion = nn.CrossEntropyLoss()
s_model = args.s_model
patch_start = args.patch_start
patch_end = args.patch_end
v_model = args.v_model
accuracy_list = []
f1_list = []
precision_list = []
overhead_list = []
for patch in range(patch_start,patch_end):
    numant = config["Ant"].as_int('numant')
    itermax = config["Ant"].as_int('itermax')
    batch_size = config["Ant"].as_int('batch_size')
    max_insert = config["Ant"].as_int('max_insert')
    val_ratio = config["Ant"].as_float('val_ratio')
    test_ratio = config["Ant"].as_float('test_ratio')

    print("=============Parameter Settings=============")
    print("numant:",numant)
    print("itermax:",itermax)
    print("batch_size:",batch_size)
    print("max_insert:",max_insert)

    # 加载数据集
    X_train, y_train, X_valid, y_valid, X_test, y_test=LoadDataNoDefCW(input_size=200,num_classes=95,test_ratio=0.1,val_ratio=0.1)
    print("val_ratio:",val_ratio,"test_ratio:",test_ratio)
    print("X_valid samples",X_valid.shape[0],"X_test samples",X_test.shape[0])
    print("Use X_train shape:",X_train.shape)
    train_dataset = MyDataset(X_train,y_train)
    val_dataset = MyDataset(X_valid,y_valid)
    test_dataset = MyDataset(X_test,y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # 加载模型
    model = torch.load('/home/xuke/lpf/all/DLWF_pytorch/trained_model/'+v_model+'.pkl')
    model.eval()
    # v = torch.tensor(np.load(f'{s_model}_perturbation_patch_{patch}.npy'))
    # v = torch.tensor(np.load(f'{s_model}_perturbation_patch_{patch}.npy'))
    # v =[[ 76,   1],
    #     [  0,   4],
    #     [ 51,   1],
    #     [ 93,   1],
    #     [ 57,   1],
    #     [ 42,   1],
    #     [ 36,   0],
    #     [ 39,   1],
    #     [141,   0]]
    count = 0
    recall = 0
    f1 = 0
    precision = 0
    accuracy = 0
    for batch_x_tensor, batch_y_tensor in tqdm(test_loader, desc='Testing'):
        batch_x_tensor = batch_x_tensor.float().to(device)
        batch_x_tensor = generate_adv_trace(v,batch_x_tensor)
        labels = batch_y_tensor.to(device).argmax(1).detach().cpu().numpy()
        outputs =  model(batch_x_tensor).argmax(1).detach().cpu().numpy()
        count = count+1
        # 1. 准确率 (Accuracy)
        accuracy += accuracy_score(labels, outputs)
        # 2. 召回率 (Recall)
        recall += recall_score(labels, outputs, average='weighted')
        # 3. 精确率 (Precision)
        precision += precision_score(labels, outputs, average='weighted')
        # 4. F1-score
        f1 += f1_score(labels, outputs, average='weighted')
    v = v.numpy()
    overhead = int(sum(v[:,1]))
    with open(f"{s_model}_{v_model}.txt",'a') as f:
        f.write(f"s_model:{s_model},v_model:{v_model},Patch:{patch},Overhead:{overhead},Accuracy: {accuracy/count:.3f},Recall: {recall/count:.3f},Precision: {precision/count:.3f},F1-score: {f1/count:.3f}\n")
    f.close()
    accuracy_list.append(round(accuracy/count,3))
    precision_list.append(round(precision/count,3))
    f1_list.append(round(f1/count,3))
    overhead_list.append(overhead)
with open(f"{s_model}_{v_model}.txt",'a') as f:
    f.write(f"Overhead:{overhead_list}\nAccuracy: {accuracy_list}\nPrecision: {precision_list}\nF1-score: {f1_list}\n")
f.close()