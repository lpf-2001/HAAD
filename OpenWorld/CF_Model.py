import os
import sys
import torch
import argparse
from DataLoder_forOW import *
from tool_utils import  *

from tqdm import tqdm
from configobj import ConfigObj
# 自定义模块路径

from model import *

# ==========================================
# 全局设备与基本设置
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import time
import os
import torch

config = ConfigObj("/home/xuke/lpf/HAAD/DLWF_pytorch/My_tor.conf")
# print(config)  # 检测是否成功加载配置文件
def get_confidence(model, X_test,class_num,batch_size, device='cuda'):
    """
    分析模型对给定测试集的预测置信度
    """
    model.eval()
    model.to(device)
    # === 性能指标统计 ===
    all_preds = np.empty((0,class_num))
    data_num=len(X_test)
    for i in tqdm(range(0, data_num, batch_size)):
            left= i
            right= min((i+batch_size), data_num)
            batch_data=X_test[left:right]
            batch_tensor = torch.from_numpy(batch_data).float().to(device)
            logit=model(batch_tensor)
            del batch_tensor
            probs=torch.softmax(logit,dim=1)
            preds=probs.detach().cpu().numpy()
            all_preds=np.concatenate((all_preds,preds),axis=0)
    return np.max(all_preds,axis=1)

def Load_Classfy_model(model_name, data_name):
    """
    根据模型名称与数据集名称构建模型实例
    """
    model = build_model_instance(model_name, data_name, config).to(device)
    model_path=f"/home/xuke/lpf/HAAD/utils/trained_model/{data_name}/{model_name}.pkl"
    model.load_state_dict(torch.load(model_path))
    return model




# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    project_path=os.getcwd()
    sys.path.append(project_path)
    from utils.data import *
     # 加载模型与数据集
    data_name='sirinam95'  # 数据集可选： sirinam95 rimmer100
    model_name = 'awf'  # 固定使用awf作为攻击模型
    if data_name=='sirinam95':
        num_classes=95    # 类别数量，对应  95 100
    elif data_name=='rimmer100':
        num_classes=100  # 类别数量，对应  95 100 
    model=Load_Classfy_model(model_name, data_name)
    
    learn_param = config[model_name]
    
    # 加载识别模型
   
    test_loader = load_datasets(data_name, num_classes, learn_param)
    get_confidence(model, test_loader, num_classes, device='cuda')
    print('get confidence done.')
    