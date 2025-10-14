import os
import re
import sys
import torch
import datetime
import pytz
import argparse
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from configobj import ConfigObj

# 自定义模块路径
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils'))
sys.path.append(parent_dir)
from model import *
from data import *

# ==========================================
# 全局设备与基本设置
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 辅助函数
# ==========================================
def split_alpha_number(s):
    m = re.match(r'^([^\d]+)(\d+)$', s)
    return (m.group(1), int(m.group(2))) if m else (s, None)

def curtime():
    tz = pytz.timezone('Asia/Chongqing')
    return datetime.datetime.now(tz).strftime('%H:%M:%S')

def log_info(dataset, model_name, msg, id=None):
    print(f"> {msg}")
    log_path = f'../utils/trained_model/{data_name}{num_classes}/{dataset}_{model_name}.out'
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
        model3 = AWFNet(num_classes=num_classes)
        return Tor_ensemble_model(model1, model2, model3, num_classes=num_classes)
    elif model_type == "awf":
        return AWFNet(num_classes=num_classes)
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

# ==========================================
# 主训练函数
# ==========================================
import time
from termcolor import colored  # ✅ pip install termcolor

def train(model, learn_param, data_name, model_name, num_classes):
    train_loader, val_loader, test_loader = load_datasets(data_name, num_classes, learn_param)
    criterion = nn.CrossEntropyLoss()

    optimizer_name = learn_param['optimizer']
    lr = learn_param[optimizer_name].as_float('learning_rate')

    optimizer_dict = {
        "rmsprop": lambda: optim.RMSprop(model.parameters(), lr=lr),
        "adamax": lambda: optim.Adamax(model.parameters(), lr=lr),
        "sgd": lambda: optim.SGD(model.parameters(), lr=lr,
                                 momentum=learn_param[optimizer_name].as_float('momentum'),
                                 weight_decay=learn_param[optimizer_name].as_float('decay')),
        "adamw": lambda: optim.AdamW(model.parameters(), lr=lr)
    }
    if optimizer_name not in optimizer_dict:
        raise ValueError(f"❌ Unsupported optimizer: {optimizer_name}")
    optimizer = optimizer_dict[optimizer_name]()

    # === 记录训练参数 ===
    log_info(data_name, model_name, f"🚀 Training started | Optimizer={optimizer_name}, lr={lr}")
    log_info(data_name, model_name, f"Model Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    best_f1 = 0
    epochs = learn_param.as_int('nb_epochs')
    s_time = time.time()
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        model.train()
        total_loss, total_correct = 0.0, 0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            x, y = x.float().to(device), y.float().to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y.argmax(1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_correct += (outputs.argmax(1) == y.argmax(1)).sum().item()

        # === 每个 epoch 训练结果 ===
        train_loss = total_loss / len(train_loader)
        train_acc = total_correct / len(train_loader.dataset)
        epoch_time = time.time() - start_time

        if epoch % 5 == 0: 
            # === 验证 ===
            val_loss, val_acc = evaluate(model, val_loader, criterion)
            acc, recall, precision, f1 = test_metrics(model, test_loader)

            # === 控制台输出（彩色） ===
            print(colored(f"\n[Epoch {epoch}/{epochs}] "
                        f"TrainLoss={train_loss:.4f}, TrainAcc={train_acc:.3f}, "
                        f"ValLoss={val_loss:.4f}, ValAcc={val_acc:.3f}, "
                        f"TestF1={f1:.3f} | Time={epoch_time:.1f}s", "cyan"))

            # === 文件日志 ===
            log_info(data_name, model_name,
                    f"[Epoch {epoch}] "
                    f"TrainLoss={train_loss:.4f}, TrainAcc={train_acc:.3f}, "
                    f"ValLoss={val_loss:.4f}, ValAcc={val_acc:.3f}, "
                    f"TestAcc={acc:.3f}, Recall={recall:.3f}, Precision={precision:.3f}, F1={f1:.3f}, "
                    f"Time={epoch_time:.1f}s")

            # === 模型保存 ===
            if f1 > best_f1:
                best_f1 = f1
                save_path = f"../utils/trained_model/{data_name}{num_classes}/{data_name}/{model_name}.pkl"
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(model.state_dict(), save_path)
                print(colored(f"💾 New best model saved! F1={f1:.3f}", "green"))
                log_info(data_name, model_name, f"✅ Saved model (F1={f1:.3f}) to {save_path}")

    train_time = time.time()-s_time  # 你可以在 train() 中用 time.time() 计算整个训练时间
    metrics = analyze_model_performance(
        model=model, 
        dataloader=test_loader, 
        dataset_name=data_name, 
        model_name=model_name, 
        train_time=train_time, 
        device=device
    )
    print(colored(f"\n🎯 Training completed! Best F1={best_f1:.3f}", "yellow"))
    log_info(data_name, model_name, f"🎯 Training completed! Best F1={best_f1:.3f}")

import time
import os
import torch
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

def analyze_model_performance(model, dataloader, dataset_name, model_name, 
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
        f"🕒 Training Time: {train_time:.2f}s\n" if train_time else "" +
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
        log_path = f"../utils/trained_model/{data_name}{num_classes}/{dataset_name}_{model_name}_summary.out"
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





# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='cnn', choices=['cnn', 'lstm', 'sdae', 'ensemble', 'df', 'varcnn','awf'])
    parser.add_argument('--train_model', '-t', action='store_true')
    parser.add_argument('--dataset', '-d', default='sirinam95', choices=['sirinam95', 'rimmer100', 'rimmer200', 'rimmer500', 'rimmer900'])
    args = parser.parse_args()

    data_name, num_classes = split_alpha_number(args.dataset)
    model_name = args.model

    print(f"Dataset: {data_name} | Classes: {num_classes} | Model: {model_name}")

    config = ConfigObj("My_tor.conf")
    learn_param = config[model_name]

    model = build_model_instance(model_name, num_classes, config).to(device)
    if args.train_model:
        train(model, learn_param, data_name, model_name, num_classes)
