from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Subset
from configobj import ConfigObj
import sys
import torch.optim as optim
import os
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),'../'))
sys.path.append(parent_dir)
from utils.data import *
from torch.utils.data import DataLoader,Subset
from tqdm import tqdm
import os
import numpy as np

from DLWF_pytorch.train_utils import *

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import pdb
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
# from torchvision import datasets, transforms







class PacketWithSizeFunction(torch.autograd.Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, noise, inp, num):
        num = int(num)
        if num ==0:
            return inp
        
        tops = torch.argsort(noise,descending=False)
        
        perts = generate_perturbation(tops[:num])


    
        return perts

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        
        # print(grad_output.shape)
        return grad_output[:,:,0].sum(dim=0),grad_output[:,:,0].sum(dim=0), grad_output , None
    

class ADDNOISER(nn.Module):
    def __init__(self,inp,device):
        super(ADDNOISER, self).__init__()
        self.inp = inp
        self.z = torch.FloatTensor(size=(1,inp))
        self.z =self.z.to(device)
        self.nz = self.z.uniform_(-0,0.5)
        
        self.independent_where = nn.Sequential(
            nn.Linear(inp,500),
            nn.ReLU(),
            nn.Linear(500,inp)
        )
        self.independent_size = nn.Sequential(
            nn.Linear(inp,500),
            nn.ReLU(),
            nn.Linear(500,inp)
        )
    
    def weight_init(self):
        for m in self._modules:
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.01)
                m.bias.data.zero_()
    
    def forward(self,outsize=200):
        nz = self.nz
        ind_where = self.independent_where(nz)
        ind_size = self.independent_size(nz)
        
        if self.inp < outsize :
            z = torch.zeros_like(ind_where)
            ind_size = torch.cat([ind_size,z,z,z,z,z],dim=1)
            ind_where = torch.cat([ind_where,z,z,z,z,z],dim=1)
        ind_size = ind_size[:,:outsize]
        ind_where = ind_where[:,:outsize]    
        return ind_where.view(-1),ind_size.view(-1)
        
        
class discrim(nn.Module):
    def __init__(self,inp):
        super(discrim, self).__init__()
        self.inp = inp
        self.dependent = nn.Sequential(
                nn.Linear(inp,1000),
                nn.ReLU(),
                nn.Linear(1000,1000),
                nn.ReLU(),
                nn.Linear(1000,1)
            )
    
    
    def forward(self, inp):
        return  self.dependent(inp[:,:,0])

# perts records the index 扰动的下标
def generate_perturbation(change_points,size=200):

    index = [i for i in range(0,200)]

    for ind in range(size):
        if ind in change_points:
            if ind == 0:
                index.insert(0,index[0])
                index = index[:size]
            else:
                index.insert(ind,index[ind-1])
                index = index[:size]
        
    return torch.tensor(index)



decider= PacketWithSizeFunction.apply

def train_adv(size_model, optim_adv, model, data, label, num_to_add=0):
    model.train()
    model.zero_grad()  # 清除主模型的梯度

    # 对抗样本生成
    where, sizes = size_model()
    data_adv = decider(where, data, num_to_add)
    data_adv = data[:, data_adv, :]
    # 清除所有梯度（包括生成器和判别器）
    optim_adv.zero_grad()
    loss2 = 10/F.cross_entropy(model(data_adv), label)
    loss2.backward()  # 计算生成器梯度
    optim_adv.step()  # 更新生成器的参数
    # 返回主模型的交叉熵损失
    output = model(data_adv)
    return F.cross_entropy(output, label)




import argparse
parser = argparse.ArgumentParser(description='Your program description')
parser.add_argument('--s_model', '-m', default='ensemble', type=str,help='choose model type cnn, lstm or sdae')

parser.add_argument('--v_model', '-vm', default='df', type=str, choices=['lstm','awf','varcnn','df','ensemble'],help='choose model type cnn, lstm or sdae')
parser.add_argument('--dataset', '-d', default='rimmer100', type=str,help='choose dataset rimmer100 or rimmer50 or sirinam')
parser.add_argument('--packet_start', '-ps', default=30, type=int)
parser.add_argument('--packet_end', '-pe', default=31, type=int)
parser.add_argument('--step', '-s', default=2, type=int)
args = parser.parse_args()
set_seed(2025)  # ✅ 固定随机种子
packet_start = args.packet_start
packet_end = args.packet_end
s_model = args.s_model
v_model = args.v_model
dataset = args.dataset
step = args.step
config = ConfigObj('../DLWF_pytorch/My_tor.conf')
learn_param = config[s_model]

data_name, num_classes = split_alpha_number(dataset)

train_loader,val_loader,test_loader = load_datasets(data_name, num_classes, learn_param)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = build_model_instance(s_model,dataset,config).to(device)
model.load_state_dict(torch.load(f'../utils/trained_model/{dataset}/{s_model}.pkl'))

model2 = build_model_instance(v_model,dataset,config).to(device)
model2.load_state_dict(torch.load(f'../utils/trained_model/{dataset}/{v_model}.pkl'))

average = 0
accuracy_list = []
f1_list = []
precision_list = []
overhead_list = []
for overhead in range(packet_start,packet_end,step):
    sum_loss = 0
    accuracy = 0
    recall = 0
    precision = 0
    f1 = 0
    addnois = ADDNOISER(200, device).to(device)
    optim_nos = optim.Adam(addnois.parameters(),lr=0.001)

    # 设置随机种子以确保可重复性
    with tqdm(train_loader, unit="batch") as tepoch:
        count = 0 
        for batch_x_tensor, batch_y_tensor in tqdm(train_loader, desc='Training'):
            batch_x_tensor = batch_x_tensor.float().to(device)
            batch_y_tensor = batch_y_tensor.float().to(device)
            cur_label = model(batch_x_tensor)
            loss = train_adv(addnois,optim_nos,model,batch_x_tensor,batch_y_tensor,overhead)
            v_label = model(batch_x_tensor)
            sum_loss += loss.item()
            

    where,sizes = addnois()
    # 将 tensor 转换为 Python 列表（先转为 numpy 数组再转列表，确保数据类型正确）
    where_list = where.cpu().detach().numpy().tolist()


    # 存储到文件（可根据需求选择 txt、json 等格式，这里以 txt 为例）
    with open(f'{dataset}_blanket.txt', 'w') as f:
        f.write(f"overhead:{overhead}, where_list:{str(where_list)}\n")
   
    print(f"{dataset} perturbation 已转换为列表并存储到 {dataset}_blanket.txt")

    real_sum_accuracy = 0
    with tqdm(test_loader, unit="batch") as tepoch:
        count = 0 
        for batch_x_tensor, batch_y_tensor in tqdm(test_loader, desc='Training'):
            
            batch_x_tensor = batch_x_tensor.float().to(device)
            labels = batch_y_tensor.float().to(device).argmax(1).detach().cpu().numpy()

            data_adv = decider(where,batch_x_tensor,overhead)
            data_adv = batch_x_tensor[:,data_adv,:]
            outputs = model2(data_adv).argmax(1).detach().cpu().numpy()
            # print(labels)
            # 1. 准确率 (Accuracy)
            accuracy += accuracy_score(labels, outputs)
            # 2. 召回率 (Recall)
            recall += recall_score(labels, outputs, average='weighted')
            # 3. 精确率 (Precision)
            precision += precision_score(labels, outputs, average='weighted')
            # 4. F1-score
            f1 += f1_score(labels, outputs, average='weighted')
            count = count+1
    
    accuracy_list.append(round(accuracy/count,3))
    precision_list.append(round(precision/count,3))
    f1_list.append(round(f1/count,3))
    overhead_list.append(overhead)
    with open(f"{s_model}_{v_model}.txt",'a') as f:
        f.write(f"s_model:{s_model},v_model:{v_model},Overhead:{overhead},Accuracy: {accuracy/count:.3f},Recall: {recall/count:.3f},Precision: {precision/count:.3f},F1-score: {f1/count:.3f}\n")
    f.close()


with open(f"{s_model}_{v_model}.txt",'a') as f:
    f.write(f"Overhead:{overhead_list}\nAccuracy: {accuracy_list}\nPrecision: {precision_list}\nF1-score: {f1_list}\n")
f.close()