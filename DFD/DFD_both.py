#!/usr/bin/env python3
# coding: utf-8

from tqdm import tqdm
import numpy as np
from decimal import Decimal, ROUND_HALF_UP
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import os
import sys
from configobj import ConfigObj
import torch
from torch.utils.data import TensorDataset, DataLoader

# 将项目根加入路径（按你原来写法）
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(parent_dir)
project_root = os.getcwd()
os.chdir(project_root)
sys.path.append(project_root)

# 你的项目内工具
from utils.data import LoadDataNoDefCW, load_rimmer_dataset  # 确保这些函数/类存在
from DLWF_pytorch.train_utils import build_model_instance,split_alpha_number,set_seed  # 假设这个函数存在并可用

set_seed(2025)
# ==========================
# 设置
# ==========================
seq_len = 200
dataset = "sirinam95"
model_name = "awf"
up_disturbance_rate = 1.5     #base setting 1.5
down_disturbance_rate = 0.25  #base setting 0.25
batch_size = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataname, num_classes = split_alpha_number(dataset)

# 评估函数（自动适配输入维度）
def evaluate(loader):
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            # 适配输入维度：模型期望 (batch, seq_len, 1)
            if batch_x.dim() == 2:
                batch_x = batch_x.unsqueeze(-1)
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x)
            preds = logits.argmax(dim=-1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(batch_y.cpu().numpy())
    if len(all_preds) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    return np.concatenate(all_preds, axis=0), np.concatenate(all_labels, axis=0)

# 自定义四舍五入
def round_up(n, digits=0):
    return Decimal(str(n)).quantize(Decimal('1e-{0}'.format(digits)), rounding=ROUND_HALF_UP)

# DFD 插包攻击函数（保持你原逻辑）
def DFDall(original_sequence, up_disturbance_rate, down_disturbance_rate):
    burst_len = []
    # 防止输入为空或第一个元素不是有效包，做个保护
    if len(original_sequence) == 0:
        return [], 0
    current_packet = original_sequence[0]
    current_count = 0
    disturbed_sequence = []
    inject_sum = 0

    for packet in original_sequence:
        disturbed_sequence.append(packet)
        if packet == 0:
            break
        else:
            if packet == current_packet:
                current_count += 1
                if current_count == 2 and len(burst_len) > 1:
                    if current_packet == 1:
                        inject_num = int(round_up(burst_len[-2] * up_disturbance_rate))
                    elif current_packet == -1:
                        inject_num = int(round_up(burst_len[-2] * down_disturbance_rate))
                    else:
                        inject_num = 0
                    if inject_num > 0:
                        inject_sum += inject_num
                        disturbance_injection = [current_packet] * inject_num
                        disturbed_sequence.extend(disturbance_injection)
            else:
                burst_len.append(current_count)
                current_packet = packet
                current_count = 1
    disturbed_sequence = [float(x) for x in disturbed_sequence]
    return disturbed_sequence, inject_sum

# ==========================
# 主流程
# ==========================


if dataname == "sirinam":
    # 加载数据（LoadDataNoDefCW 返回 X_train, y_train, X_open, y_open, x_test, y_test）
    X_train, y_train, X_open, y_open, x_test, y_test = LoadDataNoDefCW(input_size=seq_len, num_classes=num_classes, test_ratio=0.25, val_ratio=0.25)
elif dataname == "rimmer":
    X_train, y_train, X_open, y_open, x_test, y_test = load_rimmer_dataset(input_size=seq_len, num_classes=num_classes, test_ratio=0.25, val_ratio=0.25)

    print(f"x_test.shape: {np.shape(x_test)}")  # 方便调试

inject_sum = 0
disturbed_X = []

for i, sequence in enumerate(x_test):
    disturbed_sequence, injectsum_this = DFDall(sequence, up_disturbance_rate, down_disturbance_rate)
    inject_sum += injectsum_this

    # 截断或补齐
    if len(disturbed_sequence) < seq_len:
        disturbed_sequence.extend([0] * (seq_len - len(disturbed_sequence)))
    elif len(disturbed_sequence) > seq_len:
        disturbed_sequence = disturbed_sequence[:seq_len]
    disturbed_X.append(disturbed_sequence)
        
    
# 现在可以安全转换为 numpy 数组
disturbed_X = np.array(disturbed_X, dtype=np.float32)  # shape (N, seq_len)

# 确保 x_test 是 numpy 并具有正确维度 (N, seq_len) 或 (N, seq_len, 1)
x_test = np.array(x_test, dtype=np.float32)
if x_test.ndim == 3 and x_test.shape[2] == 1:
    # (N, seq_len, 1) - 保持不变
    pass
elif x_test.ndim == 2:
    # (N, seq_len) -> 添加 channel 维
    x_test = x_test[:, :seq_len]  # 保证长度
    x_test = x_test.reshape((x_test.shape[0], seq_len, 1))
else:
    # 其他情况尽量裁剪或扩维
    x_test = x_test.reshape((x_test.shape[0], seq_len, -1))
    if x_test.shape[2] != 1:
        # 如果最后一维不是1，取第一通道或者 squeeze
        x_test = x_test[:, :, 0:1]

# disturbed_X 需要变成 (N, seq_len, 1) 才能送入模型（若模型期待 channel）
disturbed_X_tensor = torch.tensor(disturbed_X, dtype=torch.float32).unsqueeze(-1)  # (N, seq_len, 1)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)  # 应该是 (N, seq_len, 1) 也可接受 (N, seq_len)

# 标签
# 如果 y_test 是 one-hot (N, C) -> 转为 (N,)
y_test = np.array(y_test)
if y_test.ndim == 2:
    # assume one-hot
    y_test = y_test.argmax(axis=1)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# DataLoaders
test_loader = DataLoader(TensorDataset(x_test_tensor, y_test_tensor), batch_size=batch_size, shuffle=False)
adv_loader = DataLoader(TensorDataset(disturbed_X_tensor, y_test_tensor), batch_size=batch_size, shuffle=False)

model_path = f'/home/xuke/lpf/HAAD/utils/trained_model/{dataset}/{model_name}.pkl'  # 你原来路径样式，请根据实际路径修改
config = ConfigObj('../DLWF_pytorch/My_tor.conf')  # 假设你有 Config 类
learn_param = config[model_name]
model = build_model_instance(model_name, dataset, config=learn_param).to(device)
state = torch.load(model_path, map_location=device)
model.load_state_dict(state)
model.eval()



# 运行评估
cur_result, real_result = evaluate(test_loader)
adv_result, _ = evaluate(adv_loader)

# 计算指标（处理一下 division by zero 的情况）
if len(real_result) == 0:
    raise RuntimeError("Empty real_result: check your test_loader and y_test.")

accuracy_before = accuracy_score(real_result, cur_result)
accuracy_after = accuracy_score(real_result, adv_result)
attack_success_rate = 1.0 - (accuracy_after / accuracy_before) if accuracy_before > 0 else float('nan')

recall = recall_score(real_result, adv_result, average='weighted', zero_division=0)
precision = precision_score(real_result, adv_result, average='weighted', zero_division=0)
f1 = f1_score(real_result, adv_result, average='weighted', zero_division=0)

# 打印并写入文件
print(f"\n=== {dataset} dataset DFD Attack Results ===")
print("Accuracy before attack:", accuracy_before)
print("Accuracy after attack: ", accuracy_after)
print("DSR  ", attack_success_rate)
print("Recall:  ", recall)
print("Precision:", precision)
print("F1-score:", f1)
print("Average injected packets per trace:", inject_sum / float(x_test_tensor.shape[0]))

os.makedirs('results', exist_ok=True)
with open(os.path.join('results', 'dfd_attack_results.txt'), 'a') as f:
    f.write(f"{dataset} dataset DFD attack results:\n")
    f.write(f"up_disturbance_rate: {up_disturbance_rate}, down_disturbance_rate: {down_disturbance_rate}\n")
    f.write(f"Accuracy before attack: {accuracy_before}\n")
    f.write(f"Accuracy after attack: {accuracy_after}\n")
    f.write(f"DSR: {attack_success_rate}\n")
    f.write(f"Recall: {recall}\n")
    f.write(f"Precision: {precision}\n")
    f.write(f"F1-score: {f1}\n")
    f.write(f"Average injected packets per trace: {inject_sum / float(x_test_tensor.shape[0])}\n\n")
