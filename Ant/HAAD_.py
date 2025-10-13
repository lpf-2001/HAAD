import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 扰动插入函数
# ============================================================
def generate_adv_trace(perturbations: torch.Tensor, traces: torch.Tensor) -> torch.Tensor:
    """在 traces 中插入对抗扰动"""
    if perturbations.numel() == 0:
        return traces

    perturbations = perturbations.to(traces.device)
    insert_loc = torch.argsort(perturbations[:, 0].cpu()).to(perturbations.device)
    perturbations = perturbations[insert_loc]

    insert_positions = perturbations[:, 0].long()
    insert_counts = perturbations[:, 1].long()
    total_insert = insert_counts.sum().item()
    if total_insert == 0:
        return traces

    B, seq_len, _ = traces.shape
    new_traces = torch.zeros((B, seq_len + total_insert, 1), device=traces.device)

    prev_pos, offset = 0, 0
    for i in range(len(insert_positions)):
        pos, num_insert = insert_positions[i].item(), insert_counts[i].item()
        new_traces[:, prev_pos+offset:pos+offset, :] = traces[:, prev_pos:pos, :]
        insert_val = traces[:, pos if pos < seq_len else -1, :]
        new_traces[:, pos+offset:pos+offset+num_insert, :] = insert_val.unsqueeze(1).expand(-1, num_insert, -1)
        prev_pos, offset = pos, offset + num_insert
    new_traces[:, prev_pos+offset:, :] = traces[:, prev_pos:, :]

    return new_traces[:, :seq_len, :]


class HAAD:
    """
    简化的混合蚁群优化 - 离散信息素 + 连续高斯采样
    
    核心功能:
    1. 从历史优质解构建高斯混合分布
    2. 混合采样：部分从连续分布，部分从信息素
    3. 自适应带宽调整
    """
    def __init__(self, model: nn.Module, num_ants: int, max_insert: int = 6,
                 patches: int = 8, max_iters: int = 10,
                 rho: float = 0.1, gamma: float = 0.35,
                 local_trials: int = 30, epsilon: float = 0.1,
                 # 连续域参数
                 continuous_ratio: float = 0.3,
                 gaussian_bandwidth: float = 5.0,
                 device=device):
        self.model = model.to(device)
        self.num_ants = num_ants
        self.max_insert = max_insert
        self.patches = patches
        self.max_iters = max_iters
        self.rho = rho
        self.gamma = gamma
        self.local_trials = local_trials
        self.epsilon = epsilon
        self.device = device
        
        # 连续域参数
        self.continuous_ratio = continuous_ratio
        self.gaussian_bandwidth = gaussian_bandwidth
        
        # 优质位置历史（用于构建连续分布）
        self.good_positions = []  # 存储 (position, score) 列表

    def _sample_from_gaussian_mixture(self, seq_len, n_samples):
        """
        从历史优质位置构建的高斯混合分布中采样
        
        返回: [n_samples] 的位置tensor
        """
        if len(self.good_positions) == 0:
            # 没有历史，均匀随机采样
            return torch.randint(0, seq_len, (n_samples,))
        
        # 提取位置和分数
        positions = torch.tensor([p for p, _ in self.good_positions], dtype=torch.float32)
        scores = torch.tensor([s for _, s in self.good_positions], dtype=torch.float32)
        
        # 分数归一化为权重
        weights = F.softmax(scores / (scores.std() + 1e-6), dim=0)
        
        # 自适应带宽：根据位置分布调整
        pos_std = positions.std().item()
        bandwidth = max(2.0, pos_std * 0.3)  # 至少为2
        
        # 采样：选择中心 + 高斯噪声
        centers_idx = torch.multinomial(weights, n_samples, replacement=True)
        centers = positions[centers_idx]
        noise = torch.randn(n_samples) * bandwidth
        sampled = centers + noise
        
        # 截断到有效范围
        sampled = torch.clamp(sampled, 0, seq_len - 1).long()
        return sampled

    def sample_paths(self, pheromone: torch.Tensor, epsilon: float = None):
        """
        混合采样：连续 + 离散
        
        参数:
            pheromone: [seq_len, max_insert+1]
            epsilon: 探索率
        返回:
            [patches, 2] tensor
        """
        if epsilon is None:
            epsilon = self.epsilon
        
        seq_len = pheromone.size(0)
        n_continuous = int(self.patches * self.continuous_ratio)
        n_discrete = self.patches - n_continuous
        
        positions = []
        counts = []
        
        # === 连续采样 ===
        if n_continuous > 0:
            cont_positions = self._sample_from_gaussian_mixture(seq_len, n_continuous)
            for pos in cont_positions:
                pos_int = int(pos.item())
                # 从信息素中采样插入数量
                count_prob = pheromone[pos_int, :]
                count_prob = count_prob / (count_prob.sum() + 1e-12)
                count = torch.multinomial(count_prob, 1).item()
                positions.append(pos_int)
                counts.append(count)
        
        # === 离散采样 ===
        if n_discrete > 0:
            prob = pheromone.view(-1)
            prob = prob / (prob.sum() + 1e-12)
            
            # epsilon-greedy
            uniform = torch.ones_like(prob) / prob.numel()
            prob = (1 - epsilon) * prob + epsilon * uniform
            
            # 避免重复
            used = set(positions)
            mask = torch.ones_like(prob, dtype=torch.bool)
            for p in used:
                mask[p * pheromone.size(1):(p + 1) * pheromone.size(1)] = False
            
            if mask.sum() > 0:
                masked_prob = prob.clone()
                masked_prob[~mask] = 0
                masked_prob = masked_prob / (masked_prob.sum() + 1e-12)
                
                idx = torch.multinomial(masked_prob, min(n_discrete, mask.sum()), replacement=False)
                positions.extend((idx // pheromone.size(1)).tolist())
                counts.extend((idx % pheromone.size(1)).tolist())
        
        return torch.tensor([[p, c] for p, c in zip(positions, counts)], dtype=torch.long)

    def evaluate_paths(self, path_solutions, traces, labels, chunk_size=64):
        """批量评估"""
        B = traces.size(0)
        adv_nums = [0] * len(path_solutions)

        for i in range(0, B, chunk_size):
            chunk_x = traces[i:i+chunk_size].to(self.device)
            chunk_y = labels[i:i+chunk_size].to(self.device)

            with torch.no_grad():
                for idx, sol in enumerate(path_solutions):
                    adv_x = generate_adv_trace(sol.to(self.device), chunk_x)
                    preds = self.model(adv_x)
                    adv_nums[idx] += (preds.argmax(-1) != chunk_y.argmax(-1)).sum().item()

        return torch.tensor(adv_nums, dtype=torch.float32)

    def _eval_single(self, sol, traces, labels, chunk_size=64):
        """评估单个解"""
        B = traces.size(0)
        total = 0
        for i in range(0, B, chunk_size):
            chunk_x = traces[i:i+chunk_size].to(self.device)
            chunk_y = labels[i:i+chunk_size].to(self.device)
            with torch.no_grad():
                adv_x = generate_adv_trace(sol.to(self.device), chunk_x)
                preds = self.model(adv_x)
                total += (preds.argmax(-1) != chunk_y.argmax(-1)).sum().item()
        return total

    def local_search(self, base, traces, labels, max_trials_per_pos=10, chunk_size=64):
        """对每个位置依次进行局部搜索"""
        best = base.clone().cpu()
        best_score = self._eval_single(best.to(self.device), traces, labels, chunk_size)
        seq_len = traces.size(1)
        
        # 对每个patch位置进行局部搜索
        for patch_idx in range(len(best)):
            improved = True
            trial = 0
            
            while improved and trial < max_trials_per_pos:
                improved = False
                trial += 1
                
                # 只修改当前patch
                cand = best.clone()
                
                if torch.rand(1) < 0.5:
                    # 调整位置
                    delta = torch.randint(-3, 4, (1,)).item()
                    cand[patch_idx, 0] = torch.clamp(cand[patch_idx, 0] + delta, 0, seq_len - 1)
                else:
                    # 调整数量
                    cand[patch_idx, 1] = torch.clamp(
                        cand[patch_idx, 1] + (1 if torch.rand(1) < 0.5 else -1),
                        0, self.max_insert
                    )
                
                score = self._eval_single(cand.to(self.device), traces, labels, chunk_size)
                
                if score > best_score:
                    best, best_score = cand.clone(), score
                    improved = True  # 继续优化这个位置
        
        return best, best_score

    def _update_pheromone(self, pheromone, solutions, scores, seq_len):
        """更新信息素"""
        # 标准化分数
        scores_arr = np.array(scores)
        mean_s, std_s = scores_arr.mean(), scores_arr.std() + 1e-9
        rewards = 1.0 / (1.0 + np.exp(-(scores_arr - mean_s) / std_s))
        
        # 计算增量
        delta = torch.zeros((seq_len, self.max_insert + 1))
        for sol, r in zip(solutions, rewards):
            for pos, cnt in sol.tolist():
                delta[int(pos), int(cnt)] += r
        
        if delta.sum() > 0:
            delta = delta / delta.sum()
        
        # 更新
        pheromone = (1 - self.rho) * pheromone + self.gamma * delta
        pheromone = pheromone.clamp(min=1e-9)
        pheromone = pheromone / pheromone.sum(dim=1, keepdim=True)
        return pheromone

    def run(self, traces, labels, chunk_size=64):
        """
        主流程
        
        返回: (best_solution, best_score)
        """
        seq_len, B = traces.size(1), traces.size(0)
        
        # 初始化
        pheromone = torch.ones((seq_len, self.max_insert + 1))
        pheromone = pheromone / pheromone.sum(dim=1, keepdim=True)
        
        best_global = None
        best_score_global = -1
        no_improve = 0
        self.good_positions = []  # 清空历史

        for it in range(self.max_iters):
            # 1. 采样
            solutions = [self.sample_paths(pheromone) for _ in range(self.num_ants)]
            
            # 2. 评估
            scores = self.evaluate_paths(solutions, traces, labels, chunk_size)
            
            # 3. 选top-5并局部搜索
            top5_idx = scores.argsort(descending=True)[:5]
            refined = []
            refined_scores = []
            
            for idx in top5_idx:
                sol, score = self.local_search(solutions[idx], traces, labels, 
                                              max_trials_per_pos=max(10, self.local_trials//3), 
                                              chunk_size=chunk_size)
                refined.append(sol)
                refined_scores.append(score)
                
                # 更新历史（用于连续采样）
                for pos in sol[:, 0].tolist():
                    self.good_positions.append((pos, score))
            
            # 保持历史大小
            if len(self.good_positions) > 100:
                self.good_positions.sort(key=lambda x: x[1], reverse=True)
                self.good_positions = self.good_positions[:50]
            
            # 4. 更新全局最优
            best_idx = int(np.argmax(refined_scores))
            if refined_scores[best_idx] > best_score_global:
                best_score_global = refined_scores[best_idx]
                best_global = refined[best_idx].clone()
                no_improve = 0
                print(f"[Iter {it}] Best: {best_score_global:.0f}/{B}, "
                      f"pos={[p for p in best_global[:, 0].tolist()]}")
            else:
                no_improve += 1
            
            # 5. 更新信息素
            pheromone = self._update_pheromone(pheromone, refined, refined_scores, seq_len)
            
            # 6. 动态调整连续比例
            self.continuous_ratio = min(0.5, 0.3 + 0.2 * (it / self.max_iters))
            
            # 7. 重启机制
            if no_improve >= max(5, self.max_iters // 5):
                pheromone = (pheromone + torch.ones_like(pheromone) / (self.max_insert + 1)) / 2
                pheromone = pheromone / pheromone.sum(dim=1, keepdim=True)
                self.good_positions = self.good_positions[:len(self.good_positions)//2]
                no_improve = 0
                print(f"[Iter {it}] Restart")
            
            # 8. 提前终止
            if best_score_global >= B * 0.98:
                print(f"[Iter {it}] Early stop")
                break
        
        if best_global is None:
            best_global = self.sample_paths(pheromone, epsilon=1.0)
            best_score_global = 0
        
        return best_global, best_score_global