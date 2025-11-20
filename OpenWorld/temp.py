import numpy as np
import torch 
import matplotlib.pyplot as plt
import os
import sys
import tqdm
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve, auc,precision_recall_curve
from configobj import ConfigObj
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

project_root = os.getcwd()
print(project_root)
os.chdir(project_root)
sys.path.append(project_root)

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 禁用 GPU


from OpenWorld.DataLoder_forOW import load_datasets
from OpenWorld.CF_Model import Load_Classfy_model,get_confidence
from Defence_method.DFD.DFD_def import DFD_def
from Defence_method.MiniPatch.MiniPatch import MiniPatch_pertubation
from Defence_method.HAAD.HAAD import HAAD_pertubation
from Defence_method.BLANKET.Blanket_def import *
from utils.data import *


def Front_def(model,test_x,device):
    print("Front EXECUTING...")
    from Defence_method.FRONT.front_def import running_front
    front_data=running_front(model,test_x,device)
    print("Front_data:",front_data.shape)
    return front_data
def Walkie_Talkie_def(model,test_x,test_y,train_x,train_y,device):
    print("Walkie_Talkie EXECUTING...")
    from Defence_method.WalkieTakie.WT_def import running_WalkieTalkie
    walkie_talkie_data=running_WalkieTalkie(model,test_x,test_y,train_x,train_y,device)
    print("Walkie_Talkie_data:",walkie_talkie_data.shape)
    return walkie_talkie_data
if __name__ == "__main__":
    # 加载默认配置项
    model_name = 'awf'  # 数据集可选： sirinam95 rimmer100
    config = ConfigObj("/home/xuke/lpf/HAAD/DLWF_pytorch/My_tor.conf")
    learn_param = config[model_name]
    batch_size=128
    substitute_model_name='ensemble'
    # 加载原始数据集
    data_name='rimmer100'  # 数据集可选： sirinam95 rimmer100
    model_name = 'awf'  # 固定使用awf作为攻击模型
    if data_name=='sirinam95':
        num_classes=95    # 类别数量，对应  95 100
    elif data_name=='rimmer100':
        num_classes=100  # 类别数量，对应  95 100 
        
    
    X_train,y_train,X_test,y_test = load_datasets(data_name, num_classes, learn_param)
    
    # 加载识别模型
    model=Load_Classfy_model(model_name, data_name)
    
    # 得到防御数据集

    ori_labels=y_test
    y_test=[1 if i<num_classes else 0 for i in y_test]

    front_data=Front_def(model,X_test[:200],device)
    walkie_talkie_data=Walkie_Talkie_def(model,X_test[:200],ori_labels[:200],X_train,y_train,device)