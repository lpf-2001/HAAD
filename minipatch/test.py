import pandas as pd
import os 
import sys
work_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(work_dir, '..')
print(parent_dir)
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
parser.add_argument('--v_model', '-vm', default='df', type=str, choices=['cnn','lstm','sdae','varcnn','df','ensemble'],help='choose model type cnn, lstm or sdae')
parser.add_argument('--dataset', '-d', default='Sirinam', type=str,help='choose dataset Rimmer100 or Sirinam')
args = parser.parse_args()
s_model = args.s_model
v_model = args.v_model
patch_start = args.patch_start
patch_end = args.patch_end

model = torch.load(work_dir+'/../DLWF_pytorch/trained_model/rimmer100/'+v_model+'.pkl')

dataset = args.dataset.lower()
print("Dataset:",dataset)
if dataset == 'rimmer100':
    X_train, y_train, X_valid, y_valid, X_test, y_test=load_rimmer_dataset(input_size=200,num_classes=100,test_ratio=0.1,val_ratio=0.1)
elif dataset == 'sirinam':
    X_train, y_train, X_valid, y_valid, X_test, y_test=LoadDataNoDefCW(input_size=200,num_classes=95,test_ratio=0.1,val_ratio=0.1,dataset='sirinam95')

test_dataset = MyDataset(X_test,y_test)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)

perturbation = [12,6,3,6,47,6,33,5,97,5,81,6,36,6,19,6]
perturbation = np.array(perturbation).flatten()
# 加载数据集

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
            
            
            
            ad  = torch.tensor(perturb_trace(inputs.cpu().numpy(), np.array(perturbation))).float().to(device)
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
        a = perturbation
        overhead = sum([abs(a[i]) for i in range(1,len(a),2)])
        with open(f"{s_model}_{v_model}.txt",'a') as f:
            f.write(f"s_model:{s_model},v_model:{v_model},Patch:{patch},Overhead:{overhead},Accuracy: {ad_accuracy/count:.3f},Recall: {ad_recall/count:.3f},Precision: {ad_precision/count:.3f},F1-score: {ad_f1/count:.3f}\n")
            f.close()



