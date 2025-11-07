import os
import re
import sys
import torch
import datetime
import time
from configobj import ConfigObj
import pytz
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from model import *

# 自定义模块路径
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils'))
sys.path.append(parent_dir)

from data import *

# ==========================================
# 辅助函数
# ==========================================
def split_alpha_number(s):
    m = re.match(r'^([^\d]+)(\d+)$', s)
    return (m.group(1), int(m.group(2))) if m else (s, None)



def curtime():
    tz = pytz.timezone('Asia/Chongqing')
    return datetime.datetime.now(tz).strftime('%H:%M:%S')

def log_info(dataset, num_classes, model_name, msg, id=None):
    print(f"> {msg}")
    log_path = f'../utils/trained_model/{dataset}{num_classes}/{dataset}_{model_name}.out'
    with open(log_path, "a") as f:
        prefix = f"ID {id} {curtime()}>\t" if id else ""
        f.write(f"{prefix}{msg}\n")
        
        
        
# ==========================================
# 数据加载
# ==========================================
def load_datasets(data_name, num_classes, learn_param):
    val_ratio = learn_param.as_float('val_ratio')
    test_ratio = learn_param.as_float('test_ratio')

    if data_name == "sirinam":
        loader = LoadDataNoDefCW
    elif data_name == "rimmer":
        loader = load_rimmer_dataset
    else:
        raise ValueError(f"Unknown dataset name: {data_name}")

    X_train, y_train, X_valid, y_valid, X_test, y_test = loader(
        input_size=200, num_classes=num_classes, val_ratio=val_ratio, test_ratio=test_ratio
    )

    print(f"Train: {X_train.shape}, Valid: {X_valid.shape}, Test: {X_test.shape}")

    batch_size = learn_param.as_int('batch_size')
    return (
        DataLoader(MyDataset(X_train, y_train), batch_size=batch_size, shuffle=True),
        DataLoader(MyDataset(X_valid, y_valid), batch_size=batch_size, shuffle=False),
        DataLoader(MyDataset(X_test, y_test), batch_size=batch_size, shuffle=False),
    )



# ==========================================
# 模型构建
# ==========================================
def build_model_instance(model_type, dataset, config):
    data_name, num_classes = split_alpha_number(dataset)
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
    elif model_type == "awf":
        return AWFNet(num_classes=num_classes)
    elif model_type == "ensemble":
        m1 = "varcnn"
        m2 = "lstm"
        m3 = "df"
        Model1 = VarCNN(200, num_classes).to(device)
        Model2= Tor_lstm(
            input_size=config['lstm']['model_param'].as_int('input_size'),
            hidden_size=config['lstm']['model_param'].as_int('hidden_size'),
            num_layers=config['lstm']['model_param'].as_int('num_layers'),
            num_classes=num_classes
        ).to(device)
        Model3 = DFNet(num_classes=num_classes).to(device)
        
        base_dir = f"../utils/trained_model/{dataset}"
        for model_obj, fname in zip([Model1, Model2, Model3],
                                    [f"{m1}.pkl", f"{m2}.pkl", f"{m3}.pkl"]):
            path = os.path.join(base_dir, fname)
            if os.path.exists(path):
                model_obj.load_state_dict(torch.load(path, map_location=device))
                print(f"✅ Loaded {fname} weights from {path}")
            else:
                print(f"⚠ {fname} not found — using random init")
            
        
        
        return Tor_ensemble_model(Model1, Model2, Model3).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    
    
# ==========================================
# 训练与验证
# ==========================================
def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss, total_acc = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.float().to(device), y.float().to(device)
            outputs = model(x)
            loss = criterion(outputs, y.argmax(1))
            total_loss += loss.item()
            total_acc += (outputs.argmax(1) == y.argmax(1)).sum().item()
    acc = total_acc / len(dataloader.dataset)
    return total_loss / len(dataloader), acc

def test_metrics(model, dataloader):
    model.eval()
    acc, recall, precision, f1 = 0, 0, 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.float().to(device), y.float().to(device)
            preds = model(x).argmax(1).cpu().numpy()
            labels = y.argmax(1).cpu().numpy()
            acc += accuracy_score(labels, preds)
            recall += recall_score(labels, preds, average='weighted')
            precision += precision_score(labels, preds, average='weighted')
            f1 += f1_score(labels, preds, average='weighted')
    n = len(dataloader)
    return acc/n, recall/n, precision/n, f1/n


def analyze_model_performance(model, dataloader, dataset_name, num_classes, model_name, 
                              train_time=None, device='cuda', log_path=None, 
                              warmup=10, repeat=100):
    """
    分析模型在测试集上的性能与部署开销
    """
    model.eval()
    model.to(device)

    # === 1️⃣ 模型参数统计 ===
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    torch.save(model.state_dict(), "temp_model.pth")
    model_size_mb = os.path.getsize("temp_model.pth") / (1024 * 1024)
    os.remove("temp_model.pth")

    # === 2️⃣ 推理延迟统计 ===
    dummy_input = next(iter(dataloader))[0].float().to(device)[:1]  # 取一个样本
    with torch.no_grad():
        for _ in range(warmup):  # 预热
            _ = model(dummy_input)

        torch.cuda.synchronize()
        start = time.time()
        for _ in range(repeat):
            _ = model(dummy_input)
        torch.cuda.synchronize()
        end = time.time()
    
    avg_latency = (end - start) / repeat
    throughput = 1.0 / avg_latency

    # === 3️⃣ 性能指标统计 ===
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.float().to(device), y.float().to(device)
            preds = model(x).argmax(1).cpu().numpy()
            labels = y.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)

    acc = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    # === 4️⃣ 输出报告 ===
    report = (
    f"\n📊 ====== Model Performance Summary ======\n"
    f"📁 Dataset: {dataset_name}{num_classes}\n"
    f"🧠 Model: {model_name}\n"
    f"-------------------------------------------\n"
    f"🔢 Trainable Params: {num_params:,}\n"
    f"💾 Model Size: {model_size_mb:.2f} MB\n"
    f"⚙️  Avg Inference Latency: {avg_latency * 1000:.3f} ms\n"
    f"🚀 Throughput: {throughput:.2f} samples/s\n"
    )

    if train_time:
        report += f"🕒 Training Time: {train_time:.2f}s\n"

    report += (
        f"-------------------------------------------\n"
        f"✅ Accuracy:  {acc:.4f}\n"
        f"📈 Recall:    {recall:.4f}\n"
        f"🎯 Precision: {precision:.4f}\n"
        f"🏆 F1-score:  {f1:.4f}\n"
        f"===========================================\n"
    )


    print(report)

    # === 5️⃣ 写入日志文件 ===
    if log_path is None:
        log_path = f"../utils/trained_model/{dataset_name}{num_classes}/{dataset_name}_{model_name}_summary.out"
    with open(log_path, "a") as f:
        f.write(report + "\n")

    return {
        "params": num_params,
        "size_MB": model_size_mb,
        "latency_ms": avg_latency * 1000,
        "throughput": throughput,
        "acc": acc,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "train_time": train_time,
    }


def evaluate_ensemble_vs_single(m1, m2, m3, dataname, num_classes, dataloader, device='cuda'):
    
    
    
    
    config = ConfigObj("My_tor.conf")
    learn_param = config['ensemble']

    Model1 = VarCNN(200, num_classes).to(device)
    Model2= Tor_lstm(
        input_size=config['lstm']['model_param'].as_int('input_size'),
        hidden_size=config['lstm']['model_param'].as_int('hidden_size'),
        num_layers=config['lstm']['model_param'].as_int('num_layers'),
        num_classes=num_classes
    ).to(device)
    Model3 = DFNet(num_classes=num_classes).to(device)
    
    base_dir = f"../utils/trained_model/{dataname}{num_classes}"
    for model_obj, fname in zip([Model1, Model2, Model3],
                                [f"{m1}.pkl", f"{m2}.pkl", f"{m3}.pkl"]):
        path = os.path.join(base_dir, fname)
        if os.path.exists(path):
            model_obj.load_state_dict(torch.load(path, map_location=device))
            print(f"✅ Loaded {fname} weights from {path}")
        else:
            print(f"⚠ {fname} not found — using random init")


    

    # 5️⃣ 构建加权投票 ensemble
    ensemble = build_model_instance('ensemble',dataname+str(num_classes),config).to(device)


    ensemble.load_state_dict(torch.load(f"../utils/trained_model/{dataname}{num_classes}/ensemble.pkl", map_location=device))
    
    
    ensemble.eval()
    Model1.eval()
    Model2.eval()
    Model3.eval()

    total = 0
    correct_ens = 0
    correct_m1 = 0
    correct_m2 = 0
    correct_m3 = 0
    fix_count = 0   # 单模型错，集成改正
    degrade_count = 0 # 单模型对，集成反而错

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.argmax(-1).to(device)

            out_ens = ensemble(x)                   # 集成输出
            out1 = Model1(x)
            out2 = Model2(x)
            out3 = Model3(x)

            pred_ens = out_ens.argmax(dim=-1)
            p1 = out1.argmax(dim=-1)
            p2 = out2.argmax(dim=-1)
            p3 = out3.argmax(dim=-1)

            # 最强单模型（从三个中选一个你觉得最强的，也可逐样本选最大置信度）
            # 这里简单用 VarCNN 作为 baseline
            pred_best = p3    

            # 统计正确数
            correct_ens += (pred_ens == y).sum().item()
            correct_m1 += (p1 == y).sum().item()
            correct_m2 += (p2 == y).sum().item()
            correct_m3 += (p3 == y).sum().item()

            # 样本级比较
            for i in range(len(y)):
                if pred_best[i] != y[i] and pred_ens[i] == y[i]:
                    fix_count += 1
                if pred_best[i] == y[i] and pred_ens[i] != y[i]:
                    degrade_count += 1

            total += y.size(0)
    log_info(dataname, num_classes, "ensemble_vs_single", "===== ✅ 集成 vs 单模型 对比结果 =====")
    log_info(dataname, num_classes, "ensemble_vs_single", f"✔ Dataset: {dataname}{num_classes}")
    log_info(dataname, num_classes, "ensemble_vs_single", f"✔ Ensemble  Acc: {correct_ens/total:.4f}")
    log_info(dataname, num_classes, "ensemble_vs_single", f"✔ {m1}    Acc: {correct_m1/total:.4f}")
    log_info(dataname, num_classes, "ensemble_vs_single", f"✔ {m2}    Acc: {correct_m2/total:.4f}")
    log_info(dataname, num_classes, "ensemble_vs_single", f"✔ {m3}      Acc: {correct_m3/total:.4f}")
    log_info(dataname, num_classes, "ensemble_vs_single", "--------------------------------------")
    log_info(dataname, num_classes, "ensemble_vs_single", f"🔹 修正错误样本数 (单模型错，但集成对) : {fix_count}")
    log_info(dataname, num_classes, "ensemble_vs_single", f"🔸 退化样本数   (单模型对，但集成错) : {degrade_count}")
    log_info(dataname, num_classes, "ensemble_vs_single", f"🔹 提升比例: {fix_count / total:.4f}")
    log_info(dataname, num_classes, "ensemble_vs_single", f"🔸 退化比例: {degrade_count / total:.4f}")


    print("===== ✅ 集成 vs 单模型 对比结果 =====")
    print(f"✔ Ensemble  Acc: {correct_ens/total:.4f}")
    print(f"✔ {m1}    Acc: {correct_m1/total:.4f}")
    print(f"✔ {m2}    Acc: {correct_m2/total:.4f}")
    print(f"✔ {m3}      Acc: {correct_m3/total:.4f}")
    print("--------------------------------------")
    print(f"🔹 修正错误样本数 (单模型错，但集成对) : {fix_count}")
    print(f"🔸 退化样本数   (单模型对，但集成错) : {degrade_count}")
    print(f"🔹 提升比例: {fix_count / total:.4f}")
    print(f"🔸 退化比例: {degrade_count / total:.4f}")
