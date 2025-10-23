import os
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from termcolor import colored
from configobj import ConfigObj
from model import *
import sys

# 引入 utils
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils'))
sys.path.append(parent_dir)
from train_utils import load_datasets, test_metrics, log_info, analyze_model_performance
from model import VarCNN, DFNet, Tor_lstm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def train_ensemble(learn_param, data_name, num_classes, config_path="My_tor.conf"):
    """训练加权投票 ensemble"""
    # 1️⃣ 加载数据
    train_loader, val_loader, test_loader = load_datasets(data_name, num_classes, learn_param)

    # 2️⃣ 配置
    config = ConfigObj(config_path)
    mp = config['lstm']['model_param']

    # 3️⃣ 加载子模型
    model_varcnn = VarCNN(200, num_classes).to(device)
    model_lstm = Tor_lstm(
        input_size=mp.as_int('input_size'),
        hidden_size=mp.as_int('hidden_size'),
        num_layers=mp.as_int('num_layers'),
        num_classes=num_classes
    ).to(device)
    model_df = DFNet(num_classes=num_classes).to(device)

    # 4️⃣ 尝试加载已有权重
    base_dir = f"../utils/trained_model/{data_name}{num_classes}"
    for model_obj, fname in zip([model_varcnn, model_lstm, model_df],
                                ["varcnn.pkl", "lstm.pkl", "df.pkl"]):
        path = os.path.join(base_dir, fname)
        if os.path.exists(path):
            model_obj.load_state_dict(torch.load(path, map_location=device))
            print(f"✅ Loaded {fname} weights from {path}")
        else:
            print(f"⚠ {fname} not found — using random init")

    # 5️⃣ 构建加权投票 ensemble
    ensemble = Tor_ensemble_model(model_varcnn, model_lstm, model_df).to(device)

    # 6️⃣ 冻结子模型
    for p in ensemble.model1.parameters():
        p.requires_grad = False
    for p in ensemble.model2.parameters():
        p.requires_grad = False
    for p in ensemble.model3.parameters():
        p.requires_grad = False

    # 7️⃣ 构建 optimizer，仅训练权重
    optimizer_name = learn_param['optimizer']
    lr = learn_param[optimizer_name].as_float('learning_rate')
    optimizer = {
        'adamax': lambda: optim.Adamax([ensemble.weights], lr=lr),
        'adam': lambda: optim.Adam([ensemble.weights], lr=lr),
        'sgd': lambda: optim.SGD([ensemble.weights], lr=lr,
                                 momentum=learn_param[optimizer_name].as_float('momentum'),
                                 weight_decay=learn_param[optimizer_name].as_float('decay')),
        'rmsprop': lambda: optim.RMSprop([ensemble.weights], lr=lr)
    }.get(optimizer_name, lambda: optim.Adam([ensemble.weights], lr=lr))()

    criterion = nn.CrossEntropyLoss()
    epochs = learn_param.as_int('nb_epochs')

    log_info(data_name, num_classes, "ensemble",
             f"🚀 Ensemble weighted training started | Optimizer={optimizer_name}, lr={lr}")
    total_trainable = sum(p.numel() for p in ensemble.parameters() if p.requires_grad)
    print(f"🔧 Trainable parameters (weights only): {total_trainable}")

    # 8️⃣ 训练循环
    best_f1 = 0.0
    s_time = time.time()
    for epoch in range(1, epochs + 1):
        ensemble.train()
        total_loss = 0.0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            x, y = x.float().to(device), y.float().to(device)
            optimizer.zero_grad()
            outputs = ensemble(x)
            loss = criterion(outputs, y.argmax(1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 验证集评估
        ensemble.eval()
        with torch.no_grad():
            acc, recall, precision, f1 = test_metrics(ensemble, val_loader)

        print(colored(
            f"[Epoch {epoch}/{epochs}] Loss={total_loss/len(train_loader):.4f} "
            f"ValAcc={acc:.3f} F1={f1:.3f} Weights={ensemble.weights.data.cpu().numpy()}",
            "cyan"))

        # 保存最佳模型
        if f1 > best_f1:
            best_f1 = f1
            save_path = os.path.join(base_dir, "ensemble.pkl")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(ensemble.state_dict(), save_path)
            print(colored(f"💾 New best ensemble saved! F1={best_f1:.3f}", "green"))
            log_info(data_name, num_classes, "ensemble",
                     f"✅ Saved model (F1={best_f1:.3f}) to {save_path}")

    # 9️⃣ 测试集评估
    ensemble.eval()
    acc, recall, precision, f1 = test_metrics(ensemble, test_loader)
    print(colored(
        f"🎯 Final Test | Acc={acc:.3f}, Recall={recall:.3f}, Precision={precision:.3f}, F1={f1:.3f}",
        "yellow"))
    print(f"🎯 Final Weights: {ensemble.weights.data.cpu().numpy()}")

    train_time = time.time() - s_time
    analyze_model_performance(
        model=ensemble,
        dataloader=test_loader,
        dataset_name=data_name,
        num_classes=num_classes,
        model_name="ensemble",
        train_time=train_time,
        device=device
    )

    log_info(data_name, num_classes, "ensemble", f"🎯 Training completed! Best F1={best_f1:.3f}")
