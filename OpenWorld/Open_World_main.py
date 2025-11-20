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
    from Defence_method.Front.Front_def import running_Front
    Front_data=running_Front(model,test_x,device)
    print("Front_data:",Front_data.shape)
    return Front_data
def Walkie_Talkie_def(model,test_x,test_y,train_x,train_y,device):
    print("Walkie_Talkie EXECUTING...")
    from Defence_method.WalkieTakie.WT_def import running_WalkieTalkie
    walkie_talkie_data=running_WalkieTalkie(model,test_x,test_y,train_x,train_y,device)
    print("Walkie_Talkie_data:",walkie_talkie_data.shape)
    return walkie_talkie_data

def Draw_PT():
    # 创建一个图形，准备绘制所有模型的 P-T 曲线
    plt.figure(figsize=(8, 4.5))  #画布尺寸

    # 绘制正例占比参考线(表示模型完全随机猜测时，其精确率等于正例在总样本中的占比)
    pos_ratio = np.mean(y_test)  # 正例在总样本中的占比
    plt.axhline(y=pos_ratio, color='k', linestyle='--', label=f'Random (POS-RATIO={pos_ratio:.2f})')

    # 第一个模型的 P-T 曲线//无防御
    precision1, _, thresholds1 = precision_recall_curve(y_test, predict_labels_Nodef)
    # 注意：precision_recall_curve返回的precision比thresholds多一个元素，需截断最后一个值以匹配
    plt.plot(thresholds1, precision1[:-1], 
            label='Origin', linewidth=3, color='tab:blue')

    # 第二个模型的 P-T 曲线//DFD
    precision2, _, thresholds2 = precision_recall_curve(y_test, predict_labels_DFD)
    plt.plot(thresholds2, precision2[:-1], 
            label='DFD', linewidth=3, color='tab:pink')

    # 第三个模型的 P-T 曲线//blanket
    precision3, _, thresholds3 = precision_recall_curve(y_test, predict_labels_blanket)
    plt.plot(thresholds3, precision3[:-1], 
            label='BLANKET', linewidth=3, color='tab:brown')

    # 第四个模型的 P-T 曲线//MiniPatch
    precision4, _, thresholds4 = precision_recall_curve(y_test, predict_labels_MiniPatch)
    plt.plot(thresholds4, precision4[:-1], 
            label='MiniPatch', linewidth=3, color='tab:purple')
    
    # 第五个模型的 P-T 曲线//Front
    precision5, _, thresholds5 = precision_recall_curve(y_test, predict_labels_Front)
    plt.plot(thresholds5, precision5[:-1], 
            label='Front', linewidth=3, color='tab:olive')
    # 第六个模型的 P-T 曲线//Walkie Talkie
    precision5, _, thresholds5 = precision_recall_curve(y_test, predict_labels_Walkie_Talkie)
    plt.plot(thresholds5, precision5[:-1], 
            label='Walkie-Talkie', linewidth=3, color='tab:cyan')
    
    # 第七个模型的 P-T 曲线//HAAD
    precision5, _, thresholds5 = precision_recall_curve(y_test, predict_labels_HAAD)
    plt.plot(thresholds5, precision5[:-1], 
            label='HAAD', linewidth=3, color='tab:orange')

    # 设置图形的一些属性
    plt.grid()

    plt.xlabel('Classification Threshold',fontsize=16)
    plt.ylabel('Precision',fontsize=16)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    # plt.title(f'WF-model: {Classfymodel}    Dataset: {DataSet}',fontsize=16)
    plt.title(f'WF-model: {model_name}    Dataset: {data_name}',fontsize=16)
    plt.legend(loc='lower right',fontsize=9)
    plt.subplots_adjust(left=0.2, bottom=0.15)  # Adjust left and bottom margins
    # 保存图形
    plt.savefig(f'/home/xuke/lpf/HAAD/OpenWorld/PT_Save/{model_name}_in_{data_name}_PT.pdf', dpi=800, bbox_inches='tight', pad_inches=0)
    plt.close() 

    # 输出成功信息
    print(f"Success saved P-T comparison plot for {data_name} dataset.")

def Drew_ROC():
     # 创建一个图形，准备绘制所有模型的 ROC 曲线
    plt.figure(figsize=(8, 4.5))  #画布尺寸

    # 绘制对角参考线
    plt.plot([0, 1], [0, 1], 'k--')

    # 第一个模型的 ROC 曲线//无防御
    fpr1, tpr1, _ = roc_curve(y_test, predict_labels_Nodef)
    aucval1 = auc(fpr1, tpr1)
    plt.plot(fpr1, tpr1, label=f'Origin (AUC={round(aucval1, 4)})', linewidth=3,color='tab:blue')

    # # 第二个模型的 ROC 曲线//Dfd
    fpr2, tpr2, _ = roc_curve(y_test, predict_labels_DFD)
    aucval2 = auc(fpr2, tpr2)
    plt.plot(fpr2, tpr2, label=f'DFD (AUC={round(aucval2, 4)})', linewidth=3,color='tab:pink')

    # #第三个方法的ROC曲线 //blanket
    fpr3,tpr3,_=roc_curve(y_test,predict_labels_blanket)
    aucval3= auc(fpr3,tpr3)
    plt.plot(fpr3,tpr3,label=f'BLANKET (AUC={round(aucval3, 4)})', linewidth=3,color='tab:brown')

    #第四个方法的ROC曲线 //MiniPatch

    fpr4,tpr4,_=roc_curve(y_test,predict_labels_MiniPatch)
    aucval4= auc(fpr4,tpr4)
    plt.plot(fpr4,tpr4,label=f'MiniPatch (AUC={round(aucval4, 4)})', linewidth=3,color='tab:purple')

    
    # 第五个模型的 ROC 曲线//Front
    fpr5,tpr5,_=roc_curve(y_test,predict_labels_Front)
    aucval5= auc(fpr5,tpr5)
    plt.plot(fpr5,tpr5,label=f'Front (AUC={round(aucval5, 5)})', linewidth=3,color='tab:olive')

    # 第六个模型的 ROC 曲线//Walkie Talkie
    fpr6,tpr6,_=roc_curve(y_test,predict_labels_Walkie_Talkie)
    aucval6= auc(fpr6,tpr6)
    plt.plot(fpr6,tpr6,label=f'Walkie-Talkie (AUC={round(aucval6, 6)})', linewidth=3,color='tab:cyan')

    #第七个方法的ROC曲线 //HAAD(df)
    fpr7,tpr7,_=roc_curve(y_test,predict_labels_HAAD_DF)
    aucval7= auc(fpr7,tpr7)
    plt.plot(fpr7,tpr7,label=f'HAAD (AUC={round(aucval7, 7)})', linewidth=3,color='tab:orange')

    #第八个方法的ROC曲线 //HAAD(ensemble)
    fpr8,tpr8,_=roc_curve(y_test,predict_labels_HAAD_Ensemble)
    aucval8= auc(fpr8,tpr8)
    plt.plot(fpr8,tpr8,label=f'HAAD(ensemble) (AUC={round(aucval8, 7)})', linewidth=3,color='tab:orange')


    # 设置图形的一些属性
    plt.grid()

    plt.xlabel('FPR',fontsize=16)
    plt.ylabel('TPR',fontsize=16)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    # plt.title(f'WF-model: {Classfymodel}    Dataset: {DataSet}',fontsize=16)
    plt.title(f'WF-model: {model_name}    Dataset: {data_name}',fontsize=16)
    plt.legend(loc='lower right',fontsize=11)
    plt.subplots_adjust(left=0.2, bottom=0.15)  # Adjust left and bottom margins
    # 保存图形
    plt.savefig(f'/home/xuke/lpf/HAAD/OpenWorld/ROC_Save/{model_name}_in_{data_name}_ROC.pdf', dpi=800, bbox_inches='tight', pad_inches=0)
    plt.close() 

    # 输出成功信息
    print(f"Success saved ROC comparison plot for {data_name} dataset.")

def Draw_PR():

    # 创建一个图形，准备绘制所有模型的 PR 曲线
    plt.figure(figsize=(8, 4.5))  # 画布尺寸

    # 计算正例占比（作为随机猜测的基准线）
    pos_ratio = np.mean(y_test)  # y_test 为真实标签（0/1），正例占比 = 正例数 / 总样本数
    plt.axhline(y=pos_ratio, color='k', linestyle='--', label=f'Random (POS-RATIO={round(pos_ratio, 4)})')
            # pos-ratio 表示正样本在整个数据集中的占比，作为随机猜测的精确率基准线。一个有用的模型，其 PR 曲线必须整体位于 pos_ratio 基准线的上方
    # 第一个模型的 PR 曲线//无防御
    precision1, recall1, _ = precision_recall_curve(y_test, predict_labels_Nodef)
    pr_auc1 = auc(recall1, precision1)  # PR曲线的AUC（横轴为recall，纵轴为precision）
    plt.plot(recall1, precision1, label=f'Origin (PR-AUC={round(pr_auc1, 4)})', linewidth=3, color='tab:blue')

    # 第二个模型的 PR 曲线//DFD
    precision2, recall2, _ = precision_recall_curve(y_test, predict_labels_DFD)
    pr_auc2 = auc(recall2, precision2)
    plt.plot(recall2, precision2, label=f'DFD (PR-AUC={round(pr_auc2, 4)})', linewidth=3, color='tab:pink')

    # 第三个模型的 PR 曲线//blanket
    precision3, recall3, _ = precision_recall_curve(y_test, predict_labels_blanket)
    pr_auc3 = auc(recall3, precision3)
    plt.plot(recall3, precision3, label=f'BLANKET (PR-AUC={round(pr_auc3, 4)})', linewidth=3, color='tab:brown')

    # 第四个模型的 PR 曲线//MiniPatch
    precision4, recall4, _ = precision_recall_curve(y_test, predict_labels_MiniPatch)
    pr_auc4 = auc(recall4, precision4)
    plt.plot(recall4, precision4, label=f'MiniPatch (PR-AUC={round(pr_auc4, 4)})', linewidth=3, color='tab:purple')

    # 第五个模型的 PR 曲线//Front
    precision5, recall5, _ = precision_recall_curve(y_test, predict_labels_Front)
    pr_auc5 = auc(recall5, precision5)
    plt.plot(recall5, precision5, label=f'Front (PR-AUCAUC={round(pr_auc5, 5)})', linewidth=3, color='tab:olive')

    # 第六个模型的 PR 曲线//Walkie-Talkie
    precision6, recall6, _ = precision_recall_curve(y_test, predict_labels_Walkie_Talkie)
    pr_auc6 = auc(recall6, precision6)
    plt.plot(recall6, precision6, label=f'Walkie-Talkie (PR-AUC={round(pr_auc6, 6)})', linewidth=3, color='tab:cyan')
    
    # 第七个模型的 PR 曲线//HAAD
    precision7, recall7, _ = precision_recall_curve(y_test, predict_labels_HAAD)
    pr_auc7 = auc(recall7, precision7)
    plt.plot(recall7, precision7, label=f'HAAD (PR-AUC={round(pr_auc7, 7)})', linewidth=3, color='tab:orange')

    # 设置图形的一些属性
    plt.grid()

    plt.xlabel('Recall', fontsize=16)  # 横轴为召回率
    plt.ylabel('Precision', fontsize=16)  # 纵轴为精确率
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlim(0, 1)  # 召回率范围 [0,1]
    plt.ylim(0, 1)  # 精确率范围 [0,1]
    plt.title(f'WF-model: {model_name}    Dataset: {data_name}', fontsize=16)
    plt.legend(loc='lower right', fontsize=11)
    plt.subplots_adjust(left=0.2, bottom=0.15)  # 调整边距
    # 保存图形
    plt.savefig(f'/home/xuke/lpf/HAAD/OpenWorld/PR_Save/{model_name}_in_{data_name}_PR.pdf', dpi=800, bbox_inches='tight', pad_inches=0)
    plt.close()

    # 输出成功信息
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

    HAAD_data=HAAD_def(X_test,substitute_model_name,data_name)
    blanket_data=Blanket_def(X_test,data_name,model_name,device)
    MiniPatch_data=MiniPatch_def(X_test,data_name)
    Front_data=Front_def(model,X_test,device)
    Walkie_Talkie_data=Walkie_Talkie_def(model,X_test,ori_labels,X_train,y_train,device)
    
    print("==>Model predicting...")
    predict_labels_Nodef=get_confidence(model,X_test,num_classes,batch_size)
    predict_labels_DFD=get_confidence(model,dfd_data,num_classes,batch_size)
    predict_labels_blanket=get_confidence(model,blanket_data,num_classes,batch_size)
    predict_labels_MiniPatch=get_confidence(model,MiniPatch_data,num_classes,batch_size)
    predict_labels_Front=get_confidence(model,Front_data,num_classes,batch_size)
    predict_labels_Walkie_Talkie =get_confidence(model,Walkie_Talkie_data,num_classes,batch_size)
    predict_labels_HAAD=get_confidence(model,HAAD_data,num_classes,batch_size)

    Draw_PR()
    Draw_PT()
    Drew_ROC()
    # print('label.shape:',y_test.shape)


    

#数据统一shape batch_size,200,1

