import torch
import torch.nn as nn
import numpy as np
import argparse
import sys
from configobj import ConfigObj
import torch.optim as optim
import os
import re
from tqdm import tqdm
from datetime import *
from torch.utils.data import DataLoader, Subset
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(parent_dir)
from utils.data import *
from HAAD_ import *
from DLWF_pytorch.model import *


def split_alpha_number(s):
    m = re.match(r'^([^\d]+)(\d+)$', s)
    if m:
        return m.group(1), int(m.group(2))
    return s, None


dataset  = "sirinam95"
s_model = None
v_model = None



def main():

    dirname = os.path.dirname(os.path.abspath(__file__))
    print(dirname)

    torch.cuda.empty_cache()
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = ConfigObj(dirname + "/../DLWF_pytorch/My_tor.conf")

    k = config["Ant"].as_int('patch_nums')
    numant = config["Ant"].as_int('numant')
    itermax = config["Ant"].as_int('itermax')
    batch_size = config["Ant"].as_int('batch_size')
    max_insert = config["Ant"].as_int('max_insert')
    val_ratio = config["Ant"].as_float('val_ratio')
    test_ratio = config["Ant"].as_float('test_ratio')

    print("=============Parameter Settings=============")
    print("numant:", numant)
    print("itermax:", itermax)
    print("batch_size:", batch_size)
    print("max_insert:", max_insert)

    # ------------------- 数据加载 -------------------
    data_name, num_classes = split_alpha_number(dataset)
    if data_name == "sirinam":
        X_train, y_train, X_valid, y_valid, X_test, y_test = LoadDataNoDefCW(input_size=200, num_classes=num_classes, test_ratio=test_ratio, val_ratio=val_ratio)
    elif data_name == "rimmer":
        X_train, y_train, X_valid, y_valid, X_test, y_test = load_rimmer_dataset(input_size=200, num_classes=num_classes, test_ratio=test_ratio, val_ratio=val_ratio)
    print("dataset",dataset,"val_ratio:", val_ratio, "test_ratio:", test_ratio)
    print("X_valid samples", X_valid.shape[0], "X_test samples", X_test.shape[0])
    print("Use X_train shape:", X_train.shape)

    train_dataset = MyDataset(X_train, y_train)
    val_dataset = MyDataset(X_valid, y_valid)
    test_dataset = MyDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # ------------------- 模型加载 -------------------
    
    S_model = torch.load(dirname + f'/../DLWF_pytorch/trained_model/{dataset}/{s_model}.pkl')
    S_model.eval()
    V_model = torch.load(dirname + f'/../DLWF_pytorch/trained_model/{dataset}/{v_model}.pkl')
    V_model.eval()

    # ------------------- 主循环 -------------------


    from U_perturbation import PerturbationEvaluator
    # 初始化评估器

    num_patchs = 10  # 设定patch数
    for patch in range(5,num_patchs):
        evaluator = PerturbationEvaluator(S_model, device)
        for x, y in train_loader:
            # 1. 用 HAAD 搜索候选扰动
            haad = HAAD(model=S_model, max_iters=itermax, patches=k, num_ants=numant, max_insert=max_insert)
            patches = haad.run(x,y)

            # 2. 评估这些扰动
            evaluator.evaluate_batch(x, y, patches, generate_adv_trace)

        # 每个 epoch 结束后，取前 k 个扰动
        top_patches = evaluator.top_k(k=patch)
        print(f"Epoch {patch}: Top patches = {top_patches}")



        # ------------------- 测试集评估 -------------------
        origin_sum, sum_, sample_sum = 0, 0, 0
        for batch_x_tensor, batch_y_tensor in tqdm(test_loader, desc='Testing'):
            batch_x_tensor = batch_x_tensor.float().to(device)
            origin_sum += (batch_y_tensor.to(device).argmax(1) == V_model(batch_x_tensor).argmax(1)).sum().item()

            v_tensor = torch.tensor(top_patches, dtype=torch.int64, device=device)  # [k,2]
            x_adv = generate_adv_trace(v_tensor, batch_x_tensor)

            sum_ += (batch_y_tensor.to(device).argmax(1) == V_model(x_adv).argmax(1)).sum().item()
            sample_sum += batch_x_tensor.shape[0]
        
        cost = v_tensor[:,1].sum()
        v = v_tensor.cpu().numpy().flatten()
        with open("HAAD.txt", 'a') as f:
            now = datetime.now(timedelta(hours=8))
            f.write(
                now.strftime("%Y-%m-%d %H:%M:%S")+'\n'+
                f"dataset:{dataset},s_model:{s_model},v_model:{v_model}\n"
                f"result:{str(v)}\n"
                f"iter:{itermax},antnum:{numant},sample_sum:{X_test.shape[0]},patch:{patch}\n"
                f"overhead:{cost},accuracy:{origin_sum/sample_sum}->{sum_/sample_sum}\n"
            )
        f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HAAD defense')
    parser.add_argument('--model', '-m', default='ensemble', type=str, choices=['cnn','lstm','sdae','ensemble','df','varcnn'],help='choose substitute model: cnn, lstm or sdae')
    parser.add_argument('--verifi_model', '-vm', default='ensemble',type=str,choices=['cnn','lstm','sdae','ensemble','df'],help='choose verified model: cnn, lstm or sdae')
    parser.add_argument('--dataset', '-d', default='sirinam95', type=str, choices=['sirinam95','rimmer100','rimmer200','rimmer500','rimmer800'],help='choose dataset')
    s_model = parser.parse_args().model
    v_model = parser.parse_args().verifi_model
    dataset = parser.parse_args().dataset
    main()