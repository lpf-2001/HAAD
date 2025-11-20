from tqdm import tqdm
import numpy as np
from decimal import Decimal, ROUND_HALF_UP
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
import logging  # 导入 logging 模块
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),'../'))
sys.path.append(parent_dir)
project_root=os.getcwd()
print(project_root)
os.chdir(project_root)
sys.path.append(project_root)
# import numpy as np
from utils.data import *


# DFD——both  

# 在上下行两个方向同时插包
# 自定义四舍五入
def round_up(n, digits=0):
    return Decimal(str(n)).quantize(Decimal('1e-{0}'.format(digits)), rounding=ROUND_HALF_UP)

def DFDall(original_sequence, up_disturbance_rate,down_disturbance_rate):
    burst_len=[]
    current_packet=original_sequence[0]
    current_count=0
    disturbed_sequence=[]

    i=0
    inject_sum=0
    for packet in original_sequence:
        i=i+1
        disturbed_sequence.append(packet)
        if packet==0:
            break
        
        else:
            if packet==current_packet:
                current_count=current_count+1
                if(current_count==2 and len(burst_len)>1):
                    if(current_packet==1):
                        inject_num=int(round_up(burst_len[len(burst_len)-2]*up_disturbance_rate))
                    elif(current_packet==-1):
                        inject_num=int(round_up(burst_len[len(burst_len)-2]*down_disturbance_rate))
                    inject_sum+=inject_num
                    disturbance_injection = [current_packet] * inject_num  # 插入的数据包，假设插入的包的内容是 '1'，可以根据需要调整
                    disturbed_sequence.extend(disturbance_injection)
            else:
                
                burst_len.append(current_count)
                current_packet=packet
                current_count=1
    return disturbed_sequence,inject_sum
Dataname=["Rimmer"]
Classfy_modelname=["DF","AWF","VarCNN"]




up_disturbance_rate = 1.5  # 扰动率 100%
down_disturbance_rate = 0.25  # 扰动率 100%

for data in Dataname:
    if data=="AWF":
        NB_CLASSES=103
    elif data=="DF":
        NB_CLASSES=95
    elif data=="Sirinam":
        NB_CLASSES=95
    elif data=="Rimmer":
        NB_CLASSES=100

    inject_sum=0
    disturbed_X=[]
    X_train, y_train, X_open, y_open, x_test, y_test  = load_rimmer_dataset(input_size=200,test_ratio=0.01,val_ratio=0.1)
    for sequence in tqdm(x_test):
        disturbed_sequence,injectsum_this= DFDall(sequence, up_disturbance_rate,down_disturbance_rate)  
        inject_sum+=injectsum_this
        if len(disturbed_sequence) < 200:
            disturbed_sequence.extend([0] * (200 - len(disturbed_sequence)))
        disturbed_X.append(disturbed_sequence[:200])  

    disturbed_X = np.array(disturbed_X,dtype=np.float64)

test_dataset = MyDataset(x_test,y_test)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

print("disturbed_X.shape:",disturbed_X.shape)
test_adv_dataset = MyDataset(disturbed_X,y_test)
test_adv_loader = DataLoader(test_adv_dataset, batch_size=256, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('/home/xuke/lpf/all_close/DLWF_pytorch/trained_model/df.pkl')
cur_result = []
real_result = []
adv_result = []

for batch_x, batch_y in test_loader:
    batch_x_tensor = batch_x.float().to(device)
    batch_y_tensor = batch_y.float().to(device)
    cur_label = model(batch_x_tensor)
    cur_result.append(cur_label.argmax(-1).cpu().numpy())
    real_result.append(batch_y_tensor.argmax(-1).cpu().numpy())
    
for batch_x, batch_y in test_adv_loader:
    batch_x_tensor = batch_x.unsqueeze(-1).float().to(device)
    # print("batch_x.shape:",batch_x.shape)
    batch_x_tensor = batch_x_tensor.float().to(device)
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
print("average inject:",inject_sum/len(test_loader.dataset))
print("recall:",recall)
print("precision:",precision)
print("f1:",f1)
    
    # for model in Classfy_modelname:
    #     classfy = ModelWrapper.ModelWrapper(model,data,is_argmax=True)
    #     pre_label=classfy(disturbed_X)
    #     ACC,F1,TPR,FPR,overall_ACC=metrics.get_metrics(y_test,pre_label)
    #     print(f"DFD attack {model} in {data} dataset:")
    #     print('acc: ',np.mean(ACC))
    #     print('F1:',np.mean(F1))
    #     print('TPR:',np.mean(TPR))
    #     print('FPR:',np.mean(FPR))
    #     print('overall_ACC:',overall_ACC)
    #     loginfo(f"DFD attack {model} in {data} dataset:")
    #     loginfo(f'acc: {np.mean(ACC)}')
    #     loginfo(f'F1: {np.mean(F1)}')
    #     loginfo(f'TPR: {np.mean(TPR)}')
    #     loginfo(f'FPR: {np.mean(FPR)}')
    #     loginfo(f'overall_ACC: {overall_ACC}')

