from tqdm import tqdm
import numpy as np
from decimal import Decimal, ROUND_HALF_UP
import logging  # 导入 logging 模块
import sys
import torch.optim as optim
import os
import torch
import pdb
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),'../'))
sys.path.append(parent_dir)
from utils.data import *

is_log=True
def loginfo(message):
    logging.info(message)

# 自定义四舍五入
def round_up(n, digits=0):
    return Decimal(str(n)).quantize(Decimal('1e-{0}'.format(digits)), rounding=ROUND_HALF_UP)
def DFDdown(original_sequence, disturbance_rate):
    burst_len=[]
    disturbed_sequence=[]
    burst_count=0
    i=0
    inject_sum=0
    for packet in original_sequence:
        disturbed_sequence.append(packet)
        if packet==-1:
            burst_count+=1
            if burst_count==2 and i>0:
                inject_num=int(round_up(burst_len[i-1]*disturbance_rate))
                inject_sum+=inject_num
                disturbance_injection = [-1] * inject_num  # 插入的数据包，由于是下行流量，所以插入 '-1'
                disturbed_sequence.extend(disturbance_injection) 
        else:
            if burst_count>0:
                burst_len.append(burst_count)
                burst_count=0
                i=i+1
        
    return disturbed_sequence,inject_sum  
def DFDup(original_sequence, disturbance_rate):
    burst_len=[]
    disturbed_sequence=[]
    burst_count=0
    i=0
    inject_sum=0
    for packet in original_sequence:
        disturbed_sequence.append(packet)
        if packet==1:
            burst_count+=1
            if burst_count==2 and i>0:
                inject_num=int(round_up(burst_len[i-1]*disturbance_rate))
                inject_sum+=inject_num
                disturbance_injection = [1] * inject_num  # 插入的数据包，假设插入的包的内容是 '1'，可以根据需要调整
                disturbed_sequence.extend(disturbance_injection) 
        
        else:
            if burst_count>0:
                burst_len.append(burst_count)
                burst_count=0
                i=i+1
        
    return disturbed_sequence,inject_sum

NB_CLASSES = 95
X_train, y_train, X_open, y_open, X_test, y_test  = LoadDataNoDefCW(input_size=200,num_classes=NB_CLASSES,test_ratio=0.8,val_ratio=0.1)
model = torch.load('/home/xuke/lpf/HAAD/utils/trained_model/sirinam95/awf.pkl')
print("X_test shape:",X_test.shape)
disturbance_rate = 0.5 # 扰动率 150%

loginfo("Inject only up flow,use disturbance_rate=150%! ")
inject_sum=0
disturbed_X=[]


    

for sequence in tqdm(X_test[:,:,0]):
    disturbed_sequence,injectsum_this= DFDup(sequence, disturbance_rate)     
    inject_sum+=injectsum_this
    disturbed_X.append(disturbed_sequence[:200])  

disturbed_X = np.array(disturbed_X)
print("disturbed_X.shape:",disturbed_X.shape)
loginfo(f"disturbed_X.shape: {disturbed_X.shape}")

disturbed_X= disturbed_X[:,:,np.newaxis]
print("disturbed_X.shape:",disturbed_X.shape)
print("average inject:",inject_sum/(NB_CLASSES*100))
loginfo(f"disturbed_X.shape: {disturbed_X.shape}")
loginfo(f"average inject: {inject_sum/(NB_CLASSES*100)}")
    
# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sum_batch_accuracy = 0
real_sum_accuracy = 0
count = 0
cur_result = []
adv_result = []
real_result = []


test_dataset = MyDataset(X_test,y_test)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
test_adv_dataset = MyDataset(disturbed_X,y_test)
test_adv_loader = DataLoader(test_adv_dataset, batch_size=256, shuffle=False)

for batch_x, batch_y in test_loader:
    batch_x_tensor = batch_x.float().to(device)
    batch_y_tensor = batch_y.float().to(device)
    cur_label = model(batch_x_tensor)
    cur_result.append(cur_label.argmax(-1).cpu().numpy())
    real_result.append(batch_y_tensor.argmax(-1).cpu().numpy())
    
for batch_x, batch_y in test_adv_loader:
    batch_x_tensor = batch_x.float().to(device)
    batch_y_tensor = batch_y.float().to(device)
    adv_label = model(batch_x_tensor)
    adv_result.append(adv_label.argmax(-1).cpu().numpy())

cur_result = np.concatenate(cur_result).flatten()
adv_result = np.concatenate(adv_result).flatten()
real_result = np.concatenate(real_result).flatten()
recall_real = recall_score(cur_result,real_result,average='weighted')
recall = recall_score(cur_result,adv_result,average='weighted')
precision = precision_score(cur_result,adv_result,average='weighted')
f1 = f1_score(cur_result,adv_result,average='weighted')
    
with open ('test.txt','a') as f:
    f.write("recall: "+str(recall)+"\n")
    f.write("precision: "+str(precision)+"\n")
    f.write("f1: "+str(f1)+"\n")
print("real_recall:",recall_real)

print("recall:",recall)
print("precision:",precision)
print("f1:",f1)
# # 测试数据
# original_sequence = [1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1,0,0,0,0]
# # original_sequence = [1, 1, 1, 1]
# # original_sequence = [-1, -1, -1, -1]
# print("扰动前的序列：", original_sequence)


# disturbance_rate = 0.8  # 扰动率 50%

# disturbed_sequence = DFD(original_sequence, disturbance_rate)   
# print("扰动后的序列：", disturbed_sequence)
            
