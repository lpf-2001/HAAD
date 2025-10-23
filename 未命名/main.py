import os
import sys
import torch
import argparse
import torch.nn as nn
from train_ensemble import train_ensemble
from train_utils import *
import torch.optim as optim
from tqdm import tqdm
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
        "adamw": lambda: optim.AdamW(model.parameters(), lr=lr),
        "adam": lambda: optim.Adam(model.parameters(), lr=lr),
    }
    if optimizer_name not in optimizer_dict:
        raise ValueError(f"❌ Unsupported optimizer: {optimizer_name}")
    optimizer = optimizer_dict[optimizer_name]()

    # === 记录训练参数 ===
    log_info(data_name, num_classes ,model_name, f"🚀 Training started | Optimizer={optimizer_name}, lr={lr}")
    log_info(data_name, num_classes, model_name, f"Model Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

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
            log_info(data_name, num_classes, model_name, 
                    f"[Epoch {epoch}] "
                    f"TrainLoss={train_loss:.4f}, TrainAcc={train_acc:.3f}, "
                    f"ValLoss={val_loss:.4f}, ValAcc={val_acc:.3f}, "
                    f"TestAcc={acc:.3f}, Recall={recall:.3f}, Precision={precision:.3f}, F1={f1:.3f}, "
                    f"Time={epoch_time:.1f}s")

            # === 模型保存 ===
            if f1 > best_f1:
                best_f1 = f1
                save_path = f"../utils/trained_model/{data_name}{num_classes}/{model_name}.pkl"
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(model.state_dict(), save_path)
                print(colored(f"💾 New best model saved! F1={f1:.3f}", "green"))
                log_info(data_name, num_classes, model_name, f"✅ Saved model (F1={f1:.3f}) to {save_path}")

    train_time = time.time()-s_time  # 你可以在 train() 中用 time.time() 计算整个训练时间
    metrics = analyze_model_performance(
        model=model, 
        dataloader=test_loader, 
        dataset_name=data_name, 
        num_classes=num_classes,
        model_name=model_name, 
        train_time=train_time, 
        device=device
    )
    print(colored(f"\n🎯 Training completed! Best F1={best_f1:.3f}", "yellow"))
    log_info(data_name, num_classes, model_name, f"🎯 Training completed! Best F1={best_f1:.3f}")

import time
import os
import torch



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

    
    if args.train_model:
        if model_name == 'ensemble':
            # 调用专用 ensemble 训练函数（它会内部构建并加载 VarCNN/DFNet/Tor_lstm）
            train_ensemble(learn_param, data_name, num_classes)
        else:
            model = build_model_instance(model_name, num_classes, config).to(device)
            train(model, learn_param, data_name, model_name, num_classes)
