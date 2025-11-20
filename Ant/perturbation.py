import torch
import random, numpy as np, os
import torch.nn as nn
import argparse
import sys
from configobj import ConfigObj
import torch.optim as optim
import os
from tqdm import tqdm
from datetime import datetime, timezone, timedelta
from torch.utils.data import DataLoader
from HAAD_utils import *

# -------------------- 路径设置 --------------------
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(parent_dir)
from utils.data import *
from HAAD_ import *
from DLWF_pytorch.train_utils import *
from DLWF_pytorch.model import *
from U_perturbation import run_universal_selector  


# ============================================================
#   主函数入口
# ============================================================
def main(base_patch_nums = 8,alpha=0.9):
    tz = timezone(timedelta(hours=8))
    now = datetime.now(tz)
    print(now.strftime("%Y-%m-%d %H:%M:%S"))

    dirname = os.path.dirname(os.path.abspath(__file__))
    print(dirname)
    
    torch.cuda.empty_cache()
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = ConfigObj(dirname + "/../DLWF_pytorch/My_tor.conf")

    # 读取配置
    
    numant = config["Ant"].as_int('numant')
    itermax = config["Ant"].as_int('itermax')
    batch_size = config["Ant"].as_int('batch_size')
    max_insert = config["Ant"].as_int('max_insert')
    val_ratio = config["Ant"].as_float('val_ratio')
    test_ratio = config["Ant"].as_float('test_ratio')

    print("=============Parameter Settings=============")
    print("patch_nums:", base_patch_nums)
    print("numant:", numant)
    print("itermax:", itermax)
    print("batch_size:", batch_size)
    print("max_insert:", max_insert)

    # ------------------- 数据加载 -------------------
    data_name, num_classes = split_alpha_number(dataset)
    if data_name == "sirinam":
        X_train, y_train, X_valid, y_valid, X_test, y_test = LoadDataNoDefCW(
            input_size=200, num_classes=num_classes, test_ratio=test_ratio, val_ratio=val_ratio)
    elif data_name == "rimmer":
        X_train, y_train, X_valid, y_valid, X_test, y_test = load_rimmer_dataset(
            input_size=200, num_classes=num_classes, test_ratio=test_ratio, val_ratio=val_ratio)

    print("dataset", dataset, "val_ratio:", val_ratio, "test_ratio:", test_ratio)
    print("X_train samples", X_train.shape[0])
    print("X_valid samples", X_valid.shape[0])
    print("X_test samples", X_test.shape[0])

    train_dataset = MyDataset(X_train, y_train)
    test_dataset = MyDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # ------------------- 模型加载 -------------------
    
    

    S_model = build_model_instance(s_model, dataset, config).to(device)
    S_model.load_state_dict(torch.load(dirname + f'/../utils/trained_model/{dataset}/{s_model}.pkl'))
    
    S_model.eval()
    
    V_model = build_model_instance(v_model, dataset, config).to(device)
    V_model.load_state_dict(torch.load(dirname + f'/../utils/trained_model/{dataset}/{v_model}.pkl'))
    V_model.eval()

    # ------------------- 收集HAAD结果 -------------------
    print("\n" + "="*80)
    print("Step 1: 收集各batch的HAAD最优组合")
    print("="*80)
    
    haad = HAAD(
        model=S_model,          # 模型
        num_ants=15,          # 蚂蚁数量（影响搜索广度）
        patches=base_patch_nums,  # 基础patch数量
        max_iters=50,       # 最大迭代次数
        diffusion_radius=3,
        diffusion_radius2=1

    )
    
  
    
    haad_results = []  # 存储每个batch的最优组合
    eval_batches = []  # 存储用于评估的batches
    
    # 从训练集收集候选组合
    max_train_batches = 20  # 限制训练batch数量，避免过长
    for batch_idx, (x_batch, y_batch) in enumerate(tqdm(train_loader, desc="收集HAAD组合")):
        patches, _ = haad.run(x_batch, y_batch)
        haad_results.append(patches)
        
        if batch_idx >= max_train_batches - 1:
            break
    
    print(f"收集到 {len(haad_results)} 个候选组合")
    
    # 从测试集收集评估batches
    max_eval_batches = 20  # 评估batch数量
    for batch_idx, (x_batch, y_batch) in enumerate(tqdm(test_loader, desc="收集评估batches")):
        eval_batches.append((x_batch, y_batch))
        
        if batch_idx >= max_eval_batches - 1:
            break
    
    print(f"收集到 {len(eval_batches)} 个评估batches")

    # ------------------- 调用改进的通用扰动算法 -------------------
    print("\n" + "="*80)
    print("Step 2: 运行改进的Universal Patch Selector")
    print("="*80)
    
    top_patches = run_universal_selector(
        S_model=S_model,
        device=device,
        generate_adv_trace_fn=generate_adv_trace,
        haad_results=haad_results,
        eval_batches=eval_batches,
        k=base_patch_nums,                      # 选择num个patches
        global_weight=alpha,         # 全局贡献权重
        local_weight=1-alpha,          # 局部贡献权重 0.9
        overlap_threshold=0.5,     # 位置重叠阈值
        strategy='greedy_diverse'  # 选择策略
    )
    
    print(f"\n最终选中的patches ({len(top_patches)}):")
    for i, p in enumerate(top_patches, 1):
        print(f"  {i}. {p}")

    # ------------------- 在测试集上验证 -------------------
    print("\n" + "="*80)
    print("Step 3: 在完整测试集上验证效果")
    print("="*80)
    
    acc_before, acc_after, cost = eval_on_test(
        V_model, 
        test_loader, 
        top_patches, 
        device,
        generate_adv_trace
    )
    
    
    success_rate = 1 - acc_after / acc_before if acc_before > 0 else 0
    
    print(f"\n最终结果:")
    print(f"  原始准确率: {acc_before:.4f}")
    print(f"  扰动后准确率: {acc_after:.4f}")
    print(f"  攻击成功率: {success_rate:.4f}")
    print(f"  总开销 (插入tokens): {cost}")

    # ------------------- 保存结果 -------------------
    result_file = f"{dataset}_universal_comparison.txt"
    with open(result_file, "a") as f:
        tz = timezone(timedelta(hours=8))
        now = datetime.now(tz)
        f.write(f"{'='*80}\n")
        f.write(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}]\n")
        f.write(f"Dataset: {dataset}\n")
        f.write(f"alpha:{alpha}\n")
        f.write(f"S_model: {s_model}\n")
        f.write(f"V_model: {v_model}\n")
        f.write(f"Params: patch_nums={base_patch_nums}, numant={numant}, "
                f"itermax={itermax}, max_insert={max_insert}\n")
        f.write(f"\nTop patches:\n")
        f.write(f"{top_patches}\n")
        f.write(f"\nResults:\n")
        f.write(f"  Accuracy: {acc_before:.4f} -> {acc_after:.4f}\n")
        f.write(f"  Success rate: {success_rate:.4f}\n")
        f.write(f"  Cost: {cost}\n")
        f.write(f"{'='*80}\n\n")
    
    print(f"\n结果已保存到: {result_file}")

    torch.cuda.empty_cache()



# ============================================================
# from pyinstrument import Profiler






if __name__ == "__main__":
    parser = get_args()
    args = parser.parse_args()

    s_model = args.model
    v_model = args.verifi_model
    dataset = args.dataset
    config = ConfigObj("../DLWF_pytorch/My_tor.conf")
    set_seed(args.seed)  # ✅ 固定随机种子

    for i in range(args.start_patch,args.end_patch):
        main(i)

