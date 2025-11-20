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
def No_def(X_test):
    return X_test

def DFD(X_test ,updisturbance_rate,downdisturbance_rate):
    
    print("DFD EXECUTING...")
    #dfd 应该是需要 （batchsize,200）
    x_test_for_dfd=np.squeeze(X_test, axis=-1)
    DFD_data=DFD_def(x_test_for_dfd ,updisturbance_rate,downdisturbance_rate)
    DFD_data = np.array(DFD_data)
    DFD_data=DFD_data[:,:,np.newaxis]
    print("DFD_data:",DFD_data.shape)
    return  np.array(DFD_data)
def Blanket_def(X_test,data_name,model_name,device):
    print("BLANKET EXECUTING...")
    Blanket_data=Blanket_pertubation(X_test,data_name,model_name,device)
    print("Blanket_data:",Blanket_data.shape)
    return  Blanket_data

def MiniPatch_def(X_test,data_name):
    print("MiniPatch EXECUTING...")
    MiniPatch_data=MiniPatch_pertubation(data_name,"DF",X_test)
    print("Minipatch_data:",MiniPatch_data.shape)
    return MiniPatch_data


def HAAD_def(X_test,modelname="DF",dataname="DF"):
    # the shape of old_trace is [batch_size,200,1]
    print("GAPDis EXECUTING...")
    GAPDis_data=HAAD_pertubation(modelname,dataname,X_test)
    print("GAPDis_data:",GAPDis_data.shape)
    return GAPDis_data

def Front_def(model,test_x,device):
    print("Front EXECUTING...")
    from Defence_method.FRONT.front_def import running_front
    Front_data=running_front(model,test_x,device)
    print("Front_data:",Front_data.shape)
    return Front_data
def Walkie_Talkie_def(model,test_x,test_y,train_x,train_y,device):
    print("Walkie_Talkie EXECUTING...")
    from Defence_method.WalkieTakie.WT_def import running_WalkieTalkie
    walkie_talkie_data=running_WalkieTalkie(model,test_x,test_y,train_x,train_y,device)
    print("Walkie_Talkie_data:",walkie_talkie_data.shape)
    return walkie_talkie_data

def Draw_PT():
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import precision_recall_curve

    plt.figure(figsize=(8, 4.5))  # 画布尺寸

    # 绘制随机参考线（正例比例）
    pos_ratio = np.mean(y_test)
    plt.axhline(y=pos_ratio, color='k', linestyle='--', label=f'Random (POS-RATIO={pos_ratio:.2f})')

    # 各模型的预测结果
    models = {
        'Origin': predict_labels_Nodef,
        'Walkie-Talkie': predict_labels_Walkie_Talkie,
        'DFD': predict_labels_DFD,
        'BLANKET': predict_labels_blanket,
        'MiniPatch': predict_labels_MiniPatch,
        'Front': predict_labels_Front,
        'HAAD(df)': predict_labels_HAAD_DF,          # ✅ 新增 HAAD(df)
        'HAAD(ensemble)': predict_labels_HAAD_Ensemble,  # ✅ 对齐 ROC 版本
    }

    # 定义调色板
    color_map = plt.cm.get_cmap('tab10')

    # 绘制每个模型的 P-T 曲线
    for idx, (name, preds) in enumerate(models.items()):
        precision, _, thresholds = precision_recall_curve(y_test, preds)
        plt.plot(thresholds, precision[:-1], 
                 label=name, linewidth=3, color=color_map(idx))

    # 设置图形属性
    plt.grid()
    plt.xlabel('Classification Threshold', fontsize=16)
    plt.ylabel('Precision', fontsize=16)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title(f'WF-model: {model_name}    Dataset: {data_name}', fontsize=16)
    plt.legend(loc='lower right', fontsize=9,ncol=2)
    plt.subplots_adjust(left=0.2, bottom=0.15)

    # 保存图形
    plt.savefig(f'/home/xuke/lpf/HAAD/OpenWorld/PT_Save/{model_name}_in_{data_name}_PT.pdf',
                dpi=800, bbox_inches='tight', pad_inches=0)
    plt.close()

    print(f"Success saved P-T comparison plot for {data_name} dataset.")

def Drew_ROC():
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc

    plt.figure(figsize=(8, 4.5))
    plt.plot([0, 1], [0, 1], 'k--')

    # 定义模型名与对应的预测结果
    models = {
        'Origin': predict_labels_Nodef,
        'Walkie-Talkie': predict_labels_Walkie_Talkie,
        'DFD': predict_labels_DFD,
        'BLANKET': predict_labels_blanket,
        'MiniPatch': predict_labels_MiniPatch,
        'Front': predict_labels_Front,
        'HAAD(df)': predict_labels_HAAD_DF,
        'HAAD(ensemble)': predict_labels_HAAD_Ensemble,
    }

    # 定义调色板（tab10 有 10 种标准配色）
    color_map = plt.cm.get_cmap('tab10')

    # 遍历模型绘图
    for idx, (name, preds) in enumerate(models.items()):
        fpr, tpr, _ = roc_curve(y_test, preds)
        aucval = auc(fpr, tpr)
        plt.plot(
            fpr, tpr,
            label=f'{name} (AUC={round(aucval, 4)})',
            linewidth=3,
            color=color_map(idx)
        )

    # 设置图形属性
    plt.grid()
    plt.xlabel('FPR', fontsize=16)
    plt.ylabel('TPR', fontsize=16)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title(f'WF-model: {model_name}    Dataset: {data_name}', fontsize=16)
    plt.legend(loc='lower right', fontsize=9,ncol=2)
    plt.subplots_adjust(left=0.2, bottom=0.15)

    # 保存图形
    plt.savefig(f'/home/xuke/lpf/HAAD/OpenWorld/ROC_Save/{model_name}_in_{data_name}_ROC.pdf',
                dpi=800, bbox_inches='tight', pad_inches=0)
    plt.close()

    print(f"Success saved ROC comparison plot for {data_name} dataset.")

def Draw_PR():
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import precision_recall_curve, auc

    # 创建画布
    plt.figure(figsize=(8, 4.5))

    # 绘制随机猜测参考线
    pos_ratio = np.mean(y_test)
    plt.axhline(y=pos_ratio, color='k', linestyle='--',
                label=f'Random (POS-RATIO={pos_ratio:.2f})')

    # 模型预测结果
    models = {
        'Origin': predict_labels_Nodef,
        'Walkie-Talkie': predict_labels_Walkie_Talkie,
        'DFD': predict_labels_DFD,
        'BLANKET': predict_labels_blanket,
        'MiniPatch': predict_labels_MiniPatch,
        'Front': predict_labels_Front,
        'HAAD(df)': predict_labels_HAAD_DF,           # ✅ 新增
        'HAAD(ensemble)': predict_labels_HAAD_Ensemble,  # ✅ 新增
    }

    # 调色板
    color_map = plt.cm.get_cmap('tab10')

    # 绘制每个模型的 PR 曲线
    for idx, (name, preds) in enumerate(models.items()):
        precision, recall, _ = precision_recall_curve(y_test, preds)
        pr_auc = auc(recall, precision)
        plt.plot(
            recall,
            precision,
            label=f'{name} (PR-AUC={pr_auc:.4f})',
            linewidth=3,
            color=color_map(idx)
        )

    # 图形属性
    plt.grid()
    plt.xlabel('Recall', fontsize=16)
    plt.ylabel('Precision', fontsize=16)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title(f'WF-model: {model_name}    Dataset: {data_name}', fontsize=16)
    plt.legend(loc='lower right', fontsize=9,ncol=2)
    plt.subplots_adjust(left=0.2, bottom=0.15)

    # 保存图像
    plt.savefig(
        f'/home/xuke/lpf/HAAD/OpenWorld/PR_Save/{model_name}_in_{data_name}_PR.pdf',
        dpi=800, bbox_inches='tight', pad_inches=0
    )
    plt.close()

    print(f"Success saved PR comparison plot for {data_name} dataset.")


#加载数据集
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


    # 加载扰动流量
    dfd_data=DFD(X_test,1.5,0.25)

    HAAD_data_DF=HAAD_def(X_test,'df',data_name)
    HAAD_data_Ensemble=HAAD_def(X_test,'ensemble',data_name)
    blanket_data=Blanket_def(X_test,data_name,model_name,device)
    MiniPatch_data=MiniPatch_def(X_test,data_name)
    Front_data=Front_def(model,X_test,device)
    Walkie_Talkie_data=Walkie_Talkie_def(model,X_test,ori_labels,X_train,y_train,device)
    # Walkie_Talkie_data=dfd_data
    
    print("==>Model predicting...")
    predict_labels_Nodef=get_confidence(model,X_test,num_classes,batch_size)
    predict_labels_DFD=get_confidence(model,dfd_data,num_classes,batch_size)
    predict_labels_blanket=get_confidence(model,blanket_data,num_classes,batch_size)
    predict_labels_MiniPatch=get_confidence(model,MiniPatch_data,num_classes,batch_size)
    predict_labels_Front=get_confidence(model,Front_data,num_classes,batch_size)
    predict_labels_Walkie_Talkie =get_confidence(model,Walkie_Talkie_data,num_classes,batch_size)
    predict_labels_HAAD_DF=get_confidence(model,HAAD_data_DF,num_classes,batch_size)
    predict_labels_HAAD_Ensemble=get_confidence(model,HAAD_data_Ensemble,num_classes,batch_size)
    Draw_PR()
    Draw_PT()
    Drew_ROC()
    # print('label.shape:',y_test.shape)


    

#数据统一shape batch_size,200,1

