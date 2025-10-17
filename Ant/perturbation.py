import torch
import torch.nn as nn
import numpy as np
import argparse
import sys
from configobj import ConfigObj
import torch.optim as optim
import os
from tqdm import tqdm
from datetime import *
from torch.utils.data import DataLoader

# -------------------- 路径设置 --------------------
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(parent_dir)
from utils.data import *
from HAAD_ import *
from DLWF_pytorch.model import *
from U_perturbation import run_improved_universal_selector  # ✅ 使用改进版


def split_alpha_number(s):
    import re
    m = re.match(r'^([^\d]+)(\d+)$', s)
    if m:
        return m.group(1), int(m.group(2))
    return s, None


# ============================================================
#   全局评估函数
# ============================================================
def eval_on_test(V_model, test_loader, top_patches, device, generate_adv_trace_fn):
    """
    评估选中的patches在测试集上的效果
    
    参数:
        V_model: 验证模型
        test_loader: 测试数据集
        top_patches: 选中的patches列表 [[pos, num], ...]
        device: 设备
        generate_adv_trace_fn: 生成对抗样本的函数
    """
    origin_sum, adv_sum, sample_sum = 0, 0, 0
    
    # 转换patches为tensor
    v_tensor = torch.tensor(top_patches, dtype=torch.int64, device=device)
    
    V_model.eval()
    with torch.no_grad():
        for batch_x_tensor, batch_y_tensor in tqdm(test_loader, desc='[Global Eval]'):
            batch_x_tensor = batch_x_tensor.float().to(device)
            batch_y_tensor = batch_y_tensor.to(device)
            
            # 原始准确率
            origin_pred = V_model(batch_x_tensor)
            origin_sum += (batch_y_tensor.argmax(1) == origin_pred.argmax(1)).sum().item()
            
            # 扰动后准确率
            x_adv = generate_adv_trace_fn(v_tensor, batch_x_tensor)
            adv_pred = V_model(x_adv)
            adv_sum += (batch_y_tensor.argmax(1) == adv_pred.argmax(1)).sum().item()
            
            sample_sum += batch_x_tensor.shape[0]
    
    acc_before = origin_sum / sample_sum
    acc_after = adv_sum / sample_sum
    total_cost = int(v_tensor[:, 1].sum().item())
    
    return acc_before, acc_after, total_cost

# ==========================================
# 模型构建
# ==========================================
def build_model_instance(model_type, num_classes, config):
    if model_type == "cnn":
        return Tor_cnn(200, num_classes)
    elif model_type == "df":
        return DFNet(num_classes)
    elif model_type == "varcnn":
        return VarCNN(200, num_classes)
    elif model_type == "lstm":
        mp = config['lstm']['model_param']
        return Tor_lstm(
            input_size=mp.as_int('input_size'),
            hidden_size=mp.as_int('hidden_size'),
            num_layers=mp.as_int('num_layers'),
            num_classes=num_classes
        )
    elif model_type == "sdae":
        layers = [config[str(i)] for i in range(1, config.as_int('nb_layers') + 1)]
        config['layers'] = layers
        return build_model(
            learn_params=config, train_gen=None, test_gen=None,
            steps=config.as_int('batch_size'), nb_classes=num_classes
        )
    elif model_type == "ensemble":
        model1 = VarCNN(200, num_classes)
        mp = config["lstm"]['model_param']
        model2 = Tor_lstm(
            input_size=mp.as_int('input_size'),
            hidden_size=mp.as_int('hidden_size'),
            num_layers=mp.as_int('num_layers'),
            num_classes=num_classes
        )
        learn_params = config["sdae"]
        layers = [learn_params[str(x)] for x in range(1, learn_params.as_int('nb_layers') + 1)]
        learn_params['layers'] = layers
        #不采用SDAE
        # model3 = build_model(
        #     learn_params=learn_params, train_gen=train_loader, test_gen=None,
        #     steps=learn_params.as_int('batch_size'), nb_classes=num_classes
        # )
        model3 = DFNet(num_classes=num_classes)
        return Tor_ensemble_model(model1, model2, model3, num_classes=num_classes)
    elif model_type == "awf":
        return AWFNet(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

# ============================================================
#   主函数入口
# ============================================================
def main(base_patch_nums = 8):
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
    S_model = build_model_instance(s_model, num_classes, config).to(device)
    S_model.load_state_dict(torch.load(dirname + f'/../utils/trained_model/{dataset}/{s_model}.pkl'))
    
    S_model.eval()
    
    V_model = build_model_instance(v_model, num_classes, config).to(device)
    V_model.load_state_dict(torch.load(dirname + f'/../utils/trained_model/{dataset}/{v_model}.pkl'))
    V_model.eval()

    # ------------------- 收集HAAD结果 -------------------
    print("\n" + "="*80)
    print("Step 1: 收集各batch的HAAD最优组合")
    print("="*80)
    
    haad = HAAD(
        model=S_model,          # 模型
        num_ants=30,          # 蚂蚁数量（影响搜索广度）
        patches=base_patch_nums,  # 基础patch数量
        max_iters=100       # 最大迭代次数

    )
    
  
    
    haad_results = []  # 存储每个batch的最优组合
    eval_batches = []  # 存储用于评估的batches
    
    # 从训练集收集候选组合
    max_train_batches = 300  # 限制训练batch数量，避免过长
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
    
    top_patches = run_improved_universal_selector(
        S_model=S_model,
        device=device,
        generate_adv_trace_fn=generate_adv_trace,
        haad_results=haad_results,
        eval_batches=eval_batches,
        k=base_patch_nums,                      # 选择10个patches
        global_weight=0.9,         # 全局贡献权重
        local_weight=0.1,          # 局部贡献权重
        individual_weight=0,     # 独立效果权重
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
    result_file = "universal_summary_improved.txt"
    with open(result_file, "a") as f:
        tz = timezone(timedelta(hours=8))
        now = datetime.now(tz)
        f.write(f"{'='*80}\n")
        f.write(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}]\n")
        f.write(f"Dataset: {dataset}\n")
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
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Improved Universal Perturbation with Conflict Detection'
    )
    parser.add_argument('--model', '-m', default='ensemble', type=str, 
                       help='Surrogate model name')
    parser.add_argument('--verifi_model', '-vm', default='ensemble', type=str,
                       help='Victim model name')
    parser.add_argument('--dataset', '-d', default='sirinam95', type=str,
                       help='Dataset name')
    args = parser.parse_args()

    s_model = args.model
    v_model = args.verifi_model
    dataset = args.dataset
    config = ConfigObj("../DLWF_pytorch/My_tor.conf")
    

    
    for i in range(4,10):
        main(i)