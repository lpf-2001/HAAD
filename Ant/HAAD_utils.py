import re
from model import *
import os



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



def split_alpha_number(s):
    m = re.match(r'^([^\d]+)(\d+)$', s)
    if m:
        return m.group(1), int(m.group(2))
    return s, None


def build_model_instance(model_type, dataset, config):   #模型类型“ensemble”，数据集名称“rimemr100”，配置文件
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
    