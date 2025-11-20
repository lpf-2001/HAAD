from tqdm import tqdm
import numpy as np
from decimal import Decimal, ROUND_HALF_UP
import logging  # 导入 logging 模块
import sys
import torch.optim as optim
import os
import torch
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),'../'))
sys.path.append(parent_dir)
from utils.data import load_rimmer_dataset

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

NB_CLASSES = 100
X_train, y_train, X_open, y_open, X_test, y_test  = load_rimmer_dataset(input_size=200,test_ratio=0.8,val_ratio=0.1)
model = torch.load('/home/xuke/lpf/all_close/DLWF_pytorch/trained_model/df.pkl')

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

disturbed_X= torch.tensor(disturbed_X[:,:,np.newaxis])
X_test = torch.tensor(X_test)
y_test = torch.tensor(y_test)
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
for i in tqdm(range(0,X_test.shape[0])):
    batch_x_tensor = X_test[i:i+1].float().to(device)
    batch_y_tensor = y_test[i:i+1].float().to(device)
    pertubed_x_tensor = disturbed_X[i:i+1].float().to(device)
    # print(batch_x_tensor==pertubed_x_tensor)
    
    cur_label = model(batch_x_tensor)
    v_label = model(pertubed_x_tensor)
    # print(cur_label.shape)
    # print(v_label.shape)
    # print(batch_y_tensor.shape)
    cur_result.append(cur_label.argmax(-1).cpu().numpy())
    adv_result.append(v_label.argmax(-1).cpu().numpy())
    sum_batch_accuracy = ((cur_label.argmax(-1)==v_label.argmax(-1))and(cur_label.argmax(-1)==batch_y_tensor.argmax(-1))).item() + sum_batch_accuracy
    real_sum_accuracy = (cur_label.argmax(-1)==batch_y_tensor.argmax(-1)).item() + real_sum_accuracy
    count = count+1
    
with open ('test.txt','a') as f:
    f.write("average inject: "+str(inject_sum/(NB_CLASSES*100))+'\n')
    f.write("accuracy change: "+str(real_sum_accuracy/count)+"->"+str(sum_batch_accuracy/real_sum_accuracy)+'\n')
print("accuracy change:",real_sum_accuracy/count,"->",sum_batch_accuracy/real_sum_accuracy)
cur_result = np.concatenate(cur_result).flatten()
adv_result = np.concatenate(adv_result).flatten()
recall = recall_score(cur_result,adv_result,average='weighted')
precision = precision_score(cur_result,adv_result,average='weighted')
f1 = f1_score(cur_result,adv_result,average='weighted')
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
            
