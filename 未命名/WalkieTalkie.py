import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import Dataset, DataLoader

def get_metrics(y_true, y_pred):
   

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    return accuracy, precision, recall, f1

def direction_to_burst(direction_seq, max_burst_len=4000):
    """
    Convert direction sequence to burst sequence
    
    Args:
        direction_seq: shape [B, L] or [B, L, 1], direction sequence
        max_burst_len: maximum burst sequence length
    
    Returns:
        burst_seq: shape [B, M], burst sequence
    """
    if isinstance(direction_seq, np.ndarray):
        direction_seq = torch.from_numpy(direction_seq)
    
    # Handle shape [B, L, 1] -> [B, L]
    if len(direction_seq.shape) == 3:
        direction_seq = direction_seq.squeeze(-1)
    
    batch_size = direction_seq.shape[0]
    seq_len = direction_seq.shape[1]

    burst_seqs = []

    for i in range(batch_size):
        seq = direction_seq[i]
        bursts = []
        last_dir = 0
        count = 0

        for j in range(seq_len):
            val = seq[j].item()
            if val == 0:
                break  # Padding reached
            if val == last_dir:
                count += 1
            else:
                if last_dir != 0:
                    bursts.append(last_dir * count)
                last_dir = val
                count = 1
        # 写入最后一个burst
        if count > 0 and last_dir != 0:
            bursts.append(last_dir * count)

        # 填充到最大长度
        pad_len = max_burst_len - len(bursts)
        bursts_padded = bursts + [0] * pad_len
        burst_seqs.append(bursts_padded)

    return torch.tensor(burst_seqs, dtype=torch.int32)


def burst_to_direction(burst_seq, max_dir_len=5000):
    """
    Convert burst sequence back to direction sequence
    
    Args:
        burst_seq: shape [B, M], burst sequence
        max_dir_len: maximum direction sequence length
    
    Returns:
        direction_seq: shape [B, L], direction sequence
    """
    if isinstance(burst_seq, np.ndarray):
        burst_seq = torch.from_numpy(burst_seq)
    
    batch_size = burst_seq.shape[0]
    burst_len = burst_seq.shape[1]

    direction_seqs = []

    for i in range(batch_size):
        bursts = burst_seq[i]
        directions = []

        for j in range(burst_len):
            val = bursts[j].item()
            if val == 0:
                break
            repeat_count = abs(val)
            direction = 1 if val > 0 else -1
            for k in range(repeat_count):
                if len(directions) >= max_dir_len:  # 截断逻辑：防止超过最大长度
                    break
                directions.append(direction)

        # 填充到最大长度
        current_len = len(directions)
        pad_len = max(0, max_dir_len - current_len)
        dir_padded = directions + [0] * pad_len
        direction_seqs.append(dir_padded)

    return torch.tensor(direction_seqs, dtype=torch.int32)


class DynamicDataset(Dataset):
    def __init__(self, x, y, return_idx=True):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).float()
        
        self.return_idx = return_idx
        self.setXY(x, y)
    
    def setXY(self, x, y):
        # 支持 [B, F] 和 [B, F, 1] 两种输入格式
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)  # [B, F] -> [B, F, 1]
        self.x = x
        self.y = y
    
    def setX(self, x):
        # 支持 [B, F] 和 [B, F, 1] 两种输入格式
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)  # [B, F] -> [B, F, 1]
        self.x = x
    
    def getX(self):
        return self.x
    
    def getY(self):
        return self.y
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        if self.return_idx:
            return self.x[idx], self.y[idx], idx
        return self.x[idx], self.y[idx]


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def label_accuracy(pred, target):
    """Computes the accuracy of model predictions matching the target labels"""
    batch_size = target.shape[0]
    correct = np.sum(pred == target)
    accuracy = correct / batch_size * 100.0
    return accuracy


class WalkieTalkie:

    def __init__(self, model, device='cuda'):
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.device = device

    def generate_walkie_talkie_samples(self, test_x, test_y, train_x, train_y, max_burst_len=None, max_dir_len=None):
        """
        根据Walkie-Talkie思路生成扰动样本

        参数:
        test_x: numpy array or Tensor, shape=[batch_size, seq_len] or [batch_size, seq_len, 1], 测试集方向序列
        test_y: numpy array or Tensor, shape=[batch_size], 测试集标签
        train_x: numpy array or Tensor, shape=[train_size, seq_len] or [train_size, seq_len, 1], 训练集方向序列
        train_y: numpy array or Tensor, shape=[train_size], 训练集标签
        max_burst_len: int, 最大burst向量长度，默认为测试集转换后的最大burst长度
        max_dir_len: int, 最大方向序列长度，默认为测试集的序列长度

        返回:
        perturbed_x: numpy array, shape=[batch_size, seq_len, 1], 扰动后的方向序列
        """
        # 确保输入是tensor并处理形状
        if isinstance(test_x, np.ndarray):
            test_x = torch.from_numpy(test_x)
        if isinstance(test_y, np.ndarray):
            test_y = torch.from_numpy(test_y)
        if isinstance(train_x, np.ndarray):
            train_x = torch.from_numpy(train_x)
        if isinstance(train_y, np.ndarray):
            train_y = torch.from_numpy(train_y)
        
        # Handle shape [B, L, 1] -> [B, L]
        original_shape_3d = False
        if len(test_x.shape) == 3:
            original_shape_3d = True
            test_x = test_x.squeeze(-1)
        if len(train_x.shape) == 3:
            train_x = train_x.squeeze(-1)
        
        # 确定最大burst长度和方向序列长度
        if max_dir_len is None:
            max_dir_len = test_x.shape[1]

        # 转换为burst表示
        if max_burst_len is None:
            # 计算测试集的最大burst长度
            test_bursts_temp = direction_to_burst(test_x)
            max_burst_len = (test_bursts_temp != 0).sum(dim=1).max().item()
        
        print("Begin test DS --> BS")
        test_bursts = direction_to_burst(test_x, max_burst_len)

        # 为每个测试样本随机选择一个不同类别的训练样本
        batch_size = test_x.shape[0]
        train_size = train_x.shape[0]

        # 转换训练集为burst表示
        print("Begin train DS --> BS")
        train_bursts = direction_to_burst(train_x, max_burst_len)

        # 为每个测试样本选择诱饵样本
        perturbed_bursts = []
        virtual_packet_counts = []

        print("Begin generate perturbed x")
        for i in range(batch_size):
            # 当前测试样本的类别
            current_label = test_y[i]

            # 筛选出不同类别的训练样本索引
            different_class_mask = train_y != current_label
            different_class_indices = torch.where(different_class_mask)[0]

            # 随机选择一个不同类别的训练样本
            random_idx = torch.randint(0, len(different_class_indices), (1,)).item()
            decoy_idx = different_class_indices[random_idx]

            # 获取测试样本和诱饵样本的burst表示
            test_burst = test_bursts[i]
            decoy_burst = train_bursts[decoy_idx]

            # 生成超序列（Supersequence）
            # 对于每个位置，取绝对值的最大值，并保留测试样本的符号
            test_abs = torch.abs(test_burst)
            decoy_abs = torch.abs(decoy_burst)
            max_abs = torch.maximum(test_abs, decoy_abs)
            test_sign = torch.sign(test_burst)

            # 超序列 = 符号 * 最大绝对值
            supersequence = test_sign * max_abs
            perturbed_bursts.append(supersequence)

            # --- 新增逻辑：计算方向序列前max_dir_len中的新增虚拟包数 ---
            burst_lens = max_abs
            burst_diff = (max_abs - test_abs).long()
            # 计算方向序列索引（每个burst结束后索引）
            cumulative_burst = torch.cumsum(burst_lens, dim=0)
            mask = cumulative_burst <= max_dir_len
            valid_len = mask.sum().item()
            # 只取前 valid_len 个 burst 的新增包
            added_virtual = burst_diff[:valid_len].sum().item()
            virtual_packet_counts.append(added_virtual)

        # 最后求平均虚拟包数量
        avg_virtual_packets = np.mean(virtual_packet_counts)
        print(f"Average number of DummyCell inserted in first {max_dir_len} positions: {avg_virtual_packets}")

        # 转换回方向序列
        perturbed_bursts = torch.stack(perturbed_bursts)
        print("Begin BS --> DS")
        perturbed_x = burst_to_direction(perturbed_bursts, max_dir_len)
        
        # Convert to numpy and restore original shape if needed
        perturbed_x = perturbed_x.numpy()
        if original_shape_3d:
            perturbed_x = np.expand_dims(perturbed_x, axis=-1)  # [B, L] -> [B, L, 1]

        return perturbed_x

    def validation_novel(self, dataloader):
        self.model.eval()
        losses = AverageMeter('Loss', ':.4e')
        pre = []
        target = []
        
        with torch.no_grad():
            for data, label, indices in dataloader:
                data = data.to(self.device)
                label = label.to(self.device).long()

                output = self.model(data)
                loss = self.criterion(output, label)

                losses.update(loss.item(), data.shape[0])
                pre.extend(output.argmax(dim=1).cpu().numpy())
                target.extend(label.cpu().numpy())
        
        pre = np.array(pre, dtype=np.float32)
        target = np.array(target, dtype=np.float32)
        accuracy, precision, recall, f1 = get_metrics(target, pre)
        return losses.avg, accuracy, precision, recall, f1

    def eval_performance(self, eval_x, eval_y, train_x, train_y, batch_size=128):
        """
        评估性能
        
        Args:
            eval_x: shape [num, 200, 1] or [num, 200]
            eval_y: shape [num]
            train_x: shape [num_train, 200, 1] or [num_train, 200]
            train_y: shape [num_train]
        """
        ori_dataset = DynamicDataset(eval_x, eval_y)
        ori_dataloader = DataLoader(ori_dataset, batch_size=batch_size, shuffle=False)
        
        loo, accuracy1, precision, recall, f1 = self.validation_novel(ori_dataloader)
        print('Performance before attack:')
        print(f'Overall_acc: {accuracy1}, loss: {loo}, TPR: {precision}, FPR: {recall}, F1: {f1}, ACC: {accuracy1}')

        print('eval_x.shape:', eval_x.shape)
        perturbed_x = self.generate_walkie_talkie_samples(
            eval_x, eval_y, train_x, train_y, max_burst_len=4000, max_dir_len=200)
        
        print('perturbed_x.shape:', perturbed_x.shape)

        pert_dataset = DynamicDataset(perturbed_x, eval_y)
        pert_dataloader = DataLoader(pert_dataset, batch_size=batch_size, shuffle=False)
        
        loo, accuracy2, precision, recall, f1 = self.validation_novel(pert_dataloader)
        print('Performance after attack:')
        print(f'Overall_acc: {accuracy2}, loss: {loo}, TPR: {precision}, FPR: {recall}, F1: {f1}, ACC: {accuracy2}')
        print("DSR:",1-accuracy2/accuracy1)

    def eval_performance_for_batch_baseline(self, eval_x, eval_y, train_x, train_y, batch_size=128):
        ori_dataset = DynamicDataset(eval_x, eval_y)
        ori_dataloader = DataLoader(ori_dataset, batch_size=batch_size, shuffle=False)
        
        loo, tpr, fpr, f1, acc, overall_acc = self.validation_novel(ori_dataloader)
        print('Performance before attack:')
        print(f'Overall_acc: {overall_acc}, loss: {loo}, TPR: {tpr}, FPR: {fpr}, F1: {f1}, ACC: {acc}')

        print('eval_x.shape:', eval_x.shape)
        if hasattr(self, 'perturbed_x') and self.perturbed_x.shape == eval_x.shape:
            print('Reuse previous perturbed x')  # reuse perturbed_x
        else:
            self.perturbed_x = self.generate_walkie_talkie_samples(
                eval_x, eval_y, train_x, train_y, max_burst_len=4000, max_dir_len=200)
        
        print('perturbed_x.shape:', self.perturbed_x.shape)

        pert_dataset = DynamicDataset(self.perturbed_x, eval_y)
        pert_dataloader = DataLoader(pert_dataset, batch_size=batch_size, shuffle=False)
        
        loo, tpr, fpr, f1, acc, overall_acc = self.validation_novel(pert_dataloader)
        print('Performance after attack:')
        print(f'Overall_acc: {overall_acc}, loss: {loo}, TPR: {tpr}, FPR: {fpr}, F1: {f1}, ACC: {acc}')


# 