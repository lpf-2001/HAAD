import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def get_metrics(y_true, y_pred):
   

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    return accuracy, precision, recall, f1

class DynamicDataset(Dataset):
    def __init__(self, x, y, return_idx=True):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).float()
        
        self.return_idx = return_idx
        self.setXY(x, y)
    
    def setXY(self, x, y):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # [B,F]-->[B,1,F]
        self.x = x
        self.y = y
    
    def setX(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # [B,F]-->[B,1,F]
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


class DirectionalFRONT:
    """
    面向方向序列的FRONT算法变体实现
    
    参数:
        Nc: 客户端最大伪包预算
        Ns: 代理端最大伪包预算，set to 0 for fair comp.
        Wmin: 最小填充窗口大小(序列长度比例)  理论上应该测试0.01, 0.05 0.1 0.15 0.2
        Wmax: 最大填充窗口大小(序列长度比例)  理论上应该测试0.1 0.2 0.3 0.4
            # AWF: Wmin=0.1 Wmax=0.1
            # DF: Wmin=0.1 Wmax=0.2
        client_direction: 客户端伪包方向(1或-1)
        server_direction: 服务器伪包方向(1或-1)
    """
    def __init__(self, model, Nc=128, Ns=1, Wmin=0.1, Wmax=0.1,
                 client_direction=1, server_direction=-1, device='cuda'):
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.device = device
        self.Nc = Nc
        self.Ns = Ns
        self.Wmin = Wmin
        self.Wmax = Wmax
        self.client_direction = client_direction
        self.server_direction = server_direction
    
    def _sample_padding_parameters(self, batch_size):
        """采样每批次的填充参数"""
        # 采样伪包数量
        nc = torch.randint(
            self.Nc, 
            self.Nc + 1, 
            (batch_size,),
            device=self.device
        )
        ns = torch.randint(
            1, 
            self.Ns + 1, 
            (batch_size,),
            device=self.device
        )
        
        # 采样填充窗口大小(序列长度比例)
        wc = torch.rand(batch_size, device=self.device) * (self.Wmax - self.Wmin) + self.Wmin
        ws = torch.rand(batch_size, device=self.device) * (self.Wmax - self.Wmin) + self.Wmin
        
        return nc, ns, wc, ws
    
    def _rayleigh_sample(self, shape, w):
        """从瑞利分布中采样"""
        # 生成标准均匀分布随机数
        u = torch.rand(*shape, device=self.device).clamp(1e-6, 1.0 - 1e-6)
        # 通过逆变换采样瑞利分布
        return w * torch.sqrt(-2.0 * torch.log(1.0 - u))

    def _generate_insertion_indices(self, batch_size, seq_lengths, nc, ns, wc, ws):
        """生成插入索引"""
        
        # 计算前端窗口大小(序列长度的比例)
        client_front_size = (seq_lengths.float() * wc).long()
        server_front_size = (seq_lengths.float() * ws).long()

        # 对每个样本生成插入索引
        client_indices_list = []
        server_indices_list = []

        for i in range(batch_size):
            # 客户端伪包插入时间(从瑞利分布采样)
            t_client = self._rayleigh_sample((nc[i].item(),), wc[i])
            # 归一化到[0,1]范围
            max_t_client = t_client.max()
            if max_t_client > 0:
                relative_pos_client = t_client / max_t_client
            else:
                relative_pos_client = torch.zeros_like(t_client)
            # 转换为实际索引(前端窗口内)
            client_indices = (relative_pos_client * client_front_size[i].float()).long()
            client_indices_list.append(client_indices.cpu().numpy().tolist())

            # 服务器伪包插入时间(从瑞利分布采样)
            t_server = self._rayleigh_sample((ns[i].item(),), ws[i])
            # 归一化到[0,1]范围
            max_t_server = t_server.max()
            if max_t_server > 0:
                relative_pos_server = t_server / max_t_server
            else:
                relative_pos_server = torch.zeros_like(t_server)
            # 转换为实际索引(前端窗口内)
            server_indices = (relative_pos_server * server_front_size[i].float()).long()
            server_indices_list.append(server_indices.cpu().numpy().tolist())

        # 转换为张量并填充到最大长度
        max_client_indices = nc.max().item()
        max_server_indices = ns.max().item()

        # 填充到相同长度
        client_indices_padded = []
        for indices in client_indices_list:
            padded = indices + [-1] * (max_client_indices - len(indices))
            client_indices_padded.append(padded)
        
        server_indices_padded = []
        for indices in server_indices_list:
            padded = indices + [-1] * (max_server_indices - len(indices))
            server_indices_padded.append(padded)

        client_indices_tensor = torch.tensor(client_indices_padded, dtype=torch.long, device=self.device)
        server_indices_tensor = torch.tensor(server_indices_padded, dtype=torch.long, device=self.device)

        return client_indices_tensor, server_indices_tensor
    
    def __call__(self, direction_sequences):
        """
        处理方向序列批次，生成插入索引
        
        参数:
            direction_sequences: 方向序列批次 [batch_size, sequence_length]
            
        返回:
            client_insert_indices: 客户端伪包插入索引 [batch_size, max_nc]
            server_insert_indices: 服务器伪包插入索引 [batch_size, max_ns]
            client_packet_directions: 客户端伪包方向 [batch_size, max_nc]
            server_packet_directions: 服务器伪包方向 [batch_size, max_ns]
        """
        if isinstance(direction_sequences, np.ndarray):
            direction_sequences = torch.from_numpy(direction_sequences).to(self.device)
        
        batch_size, sequence_length = direction_sequences.shape
        
        # 采样填充参数
        nc, ns, wc, ws = self._sample_padding_parameters(batch_size)
        
        # 序列长度(不包含填充)
        seq_lengths = torch.ones(batch_size, dtype=torch.long, device=self.device) * sequence_length
        
        # 生成插入索引
        client_indices, server_indices = self._generate_insertion_indices(
            batch_size, seq_lengths, nc, ns, wc, ws
        )
        
        # 生成对应的方向
        max_nc = client_indices.shape[1]
        max_ns = server_indices.shape[1]

        client_directions = torch.full(
            (batch_size, max_nc), 
            self.client_direction, 
            dtype=torch.long,
            device=self.device
        )

        server_directions = torch.full(
            (batch_size, max_ns), 
            self.server_direction, 
            dtype=torch.long,
            device=self.device
        )
        
        return client_indices, server_indices, client_directions, server_directions

   

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

    def get_perturbed_data(self, data, perturbations):
        """
        data: [1, Feat] 单条序列
        perturbations: 插入索引列表
        返回扰动后的序列，不改变长度
        """
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        if len(data.shape) == 3:
            data = np.squeeze(data, axis=1)  # [1, 1, Feat] -> [1, Feat]
        
        batch_size, n = data.shape
        result = np.ones((batch_size, n), dtype=np.float32)

        original_index = 0
        insert_count = 0
        for pos in perturbations:
            if pos + insert_count >= n:
                break
            num_elements_to_copy = max(0, pos - original_index)
            result[:, original_index + insert_count:original_index + insert_count + num_elements_to_copy] = \
                data[:, original_index:original_index + num_elements_to_copy]
            insert_count += 1
            original_index = pos
        if original_index < n:
            result[:, original_index + insert_count:] = data[:, original_index: n - insert_count]

        result = np.expand_dims(result, axis=2)  # [1, Feat] -> [1,Feat,1]
        return result, insert_count  # 返回实际插入包数量

    def eval_performance(self, eval_x, eval_y, batch_size=128):
        ori_dataset = DynamicDataset(eval_x, eval_y)
        ori_dataloader = DataLoader(ori_dataset, batch_size=batch_size, shuffle=False)
        
        losses, accuracy1, precision, recall, f1 = self.validation_novel(ori_dataloader)
        print('Performance before attack:')
        print(f'Overall_acc: {accuracy1}, loss: {losses}, TPR: {precision}, FPR: {recall}, F1: {f1}, ACC: {accuracy1}')

        x = eval_x
        assert len(x.shape) == 2, f'x.shape={x.shape}, not [batch, Feat].'
        print('x.shape:', x.shape)

        # 获取客户端和服务器的插入索引
        client_indices, server_indices, _, _ = self.__call__(x)

        perturbed_x_list = []
        inserted_counts = []
        print('Begin generate perturbed x')

        for i in range(len(x)):
            single_x = np.expand_dims(x[i], axis=0)
            # 合并客户端和服务器索引
            valid_indices = np.concatenate([
                client_indices[i][client_indices[i] != -1].cpu().numpy(),
                server_indices[i][server_indices[i] != -1].cpu().numpy()
            ])
            perturbed_single_x, count = self.get_perturbed_data(single_x, valid_indices)
            perturbed_x_list.append(perturbed_single_x)
            inserted_counts.append(count)

        perturbed_x = np.concatenate(perturbed_x_list, axis=0)

        # 平均每条序列插入包数量
        avg_inserted = np.mean(inserted_counts)
        print(f'Average inserted packets per sequence (client + server): {avg_inserted:.2f}')

        pert_dataset = DynamicDataset(perturbed_x, eval_y)
        pert_dataloader = DataLoader(pert_dataset, batch_size=batch_size, shuffle=False)
        
        losses, accuracy2, precision, recall, f1 = self.validation_novel(pert_dataloader)
        print('Performance after attack:')
        print(f'Overall_acc: {accuracy2}, loss: {losses}, TPR: {precision}, FPR: {recall}, F1: {f1}, ACC: {accuracy2}')
        print("DSR:",1-accuracy2/accuracy1)
        return avg_inserted
    
    def perturb_batch_torch(self, x_test, return_metadata=False):
        """
        PyTorch版本的批量扰动函数，保持在GPU上
        
        参数:
            batch_x: torch.Tensor [batch_size, seq_len] 或 [batch_size, 1, seq_len]
            return_metadata: 是否返回元数据
        
        返回:
            perturbed_x: torch.Tensor [batch_size, 1, seq_len]
        """
        # 统一为 [batch_size, seq_len]
        if len(x_test.shape) == 3:
            x_test = x_test.squeeze(-1)
        batch_x=torch.from_numpy(x_test).float().to(self.device)
        batch_size, seq_len = batch_x.shape
        device = batch_x.device
        
        # 生成插入索引
        client_indices, server_indices, _, _ = self.__call__(batch_x)
        
        # 批量生成扰动样本
        perturbed_list = []
        inserted_counts = []
        
        for i in range(batch_size):
            single_x = batch_x[i:i+1]  # [1, seq_len]
            
            # 合并有效索引
            valid_client = client_indices[i][client_indices[i] != -1]
            valid_server = server_indices[i][server_indices[i] != -1]
            valid_indices = torch.cat([valid_client, valid_server])
            valid_indices = torch.sort(valid_indices)[0]
            
            # 转numpy处理（因为get_perturbed_data是numpy实现）
            single_x_np = single_x.cpu().numpy()
            valid_indices_np = valid_indices.cpu().numpy()
            
            perturbed_np, count = self.get_perturbed_data(single_x_np, valid_indices_np)
            perturbed_tensor = torch.from_numpy(perturbed_np).float().to(device)
            
            perturbed_list.append(perturbed_tensor)
            inserted_counts.append(count)
        
        perturbed_x = torch.cat(perturbed_list, dim=0)
        
        if return_metadata:
            return perturbed_x, inserted_counts, client_indices, server_indices
        else:
            return perturbed_x