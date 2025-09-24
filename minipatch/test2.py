import pandas as pd
import os 
import sys
parent_dir = os.path.abspath(os.path.join('/home/xuke/lpf/all/minipatch/','../'))
sys.path.append(parent_dir)
from utils.data import *
from perturb_utils import perturb_trace
from Ant.Ant_algorithm import *
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader
import torch
torch.cuda.empty_cache()



from tqdm import tqdm
import argparse
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# X_train, y_train, X_valid, y_valid, X_test, y_test=LoadDataNoDefCW(input_size=200,num_classes=95,test_ratio=0.1,val_ratio=0.1)
X_train, y_train, X_valid, y_valid, X_test, y_test=load_rimmer_dataset(input_size=200,test_ratio=0.1,val_ratio=0.1)
test_dataset = MyDataset(X_test,y_test)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)
count = 0
recall = 0
f1 = 0
precision = 0
accuracy = 0
ad_recall = 0
ad_f1 = 0
ad_precision = 0
ad_accuracy = 0

parser = argparse.ArgumentParser(description='Train and test a deep neural network (SDAE, CNN or LSTM)')

parser.add_argument('--s_model', '-sm', default='ensemble', type=str, choices=['cnn','lstm','sdae','varcnn','df','ensemble'],help='choose model type cnn, lstm or sdae')
parser.add_argument('--patch_start', '-ps', default=8, type=int)
parser.add_argument('--patch_end', '-pe', default=9, type=int)
parser.add_argument('--v_model', '-vm', default='cnn', type=str, choices=['cnn','lstm','sdae','varcnn','df','ensemble'],help='choose model type cnn, lstm or sdae')
args = parser.parse_args()
s_model = args.s_model
v_model = args.v_model
patch_start = args.patch_start
patch_end = args.patch_end

model = torch.load('/home/xuke/lpf/all/DLWF_pytorch/trained_model/'+v_model+'.pkl')



# 加载数据集
accuracy_list = []
f1_list = []
precision_list = []
overhead_list = []
with torch.no_grad():
    for patch in range(patch_start,patch_end):
   
        for inputs, labels in tqdm(test_loader):
            inputs, labels = inputs.float(), labels.float()
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).argmax(1).detach().cpu().numpy()       
            labels = labels.argmax(1).detach().cpu().numpy()
            count = count+1
            # 1. 准确率 (Accuracy)
            accuracy += accuracy_score(labels, outputs)
            
            # 2. 召回率 (Recall)
            recall += recall_score(labels, outputs, average='weighted')
            
            # 3. 精确率 (Precision)
            precision += precision_score(labels, outputs, average='weighted')

            # 4. F1-score
            f1 += f1_score(labels, outputs, average='weighted')
            
            
            ad  = torch.tensor(perturb_trace(inputs.cpu().numpy(), [17,6,34,5,28,6,97,6,2,-6,65,6])).float().to(device)
            y_ad = model(ad).argmax(1).detach().cpu().numpy()  
            # 1. 准确率 (Accuracy)
            ad_accuracy += accuracy_score(labels, y_ad)
            
            # 2. 召回率 (Recall)
            ad_recall += recall_score(labels, y_ad, average='weighted')
            
            # 3. 精确率 (Precision)
            ad_precision += precision_score(labels, y_ad, average='weighted')

            # 4. F1-score
            ad_f1 += f1_score(labels, y_ad, average='weighted')
        print(accuracy/count)
        overhead = sum([abs([17,6,34,5,28,6,97,6,2,-6,65,6][i]) for i in range(1,len([17,6,34,5,28,6,97,6,2,-6,65,6]),2)])
        with open(f"{s_model}_{v_model}.txt",'a') as f:
            f.write(f"s_model:{s_model},v_model:{v_model},Patch:{patch},Overhead:{overhead},Accuracy: {ad_accuracy/count:.3f},Recall: {ad_recall/count:.3f},Precision: {ad_precision/count:.3f},F1-score: {ad_f1/count:.3f}\n")
            f.close()
        accuracy_list.append(round(ad_accuracy/count,3))
        precision_list.append(round(ad_precision/count,3))
        f1_list.append(round(ad_f1/count,3))
        overhead_list.append(overhead)

with open(f"{s_model}_{v_model}.txt",'a') as f:
    f.write(f"Overhead:{overhead_list}\nAccuracy: {accuracy_list}\nPrecision: {precision_list}\nF1-score: {f1_list}\n")
f.close()

