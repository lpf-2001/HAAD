import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HAAD:
    """
    改进版HAAD - 解耦信息素设计
    
    核心改进:
    1. 位置信息素和数量信息素分离
    2. 信息素更新时考虑邻域扩散
    3. 统一的采样机制
    """
    def __init__(self, model: nn.Module, num_ants: int, max_insert: int = 6,
                 patches: int = 8, max_iters: int = 10,
                 rho: float = 0.1, gamma: float = 0.35,
                 local_trials: int = 30, epsilon: float = 0.1,
                 diffusion_radius: int = 5,  # 新增: 信息素扩散半径
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
        self.diffusion_radius = diffusion_radius
        self.device = device
        
        # 历史最优解
        self.best_history = []  # 存储 (solution, score)

    def _create_diffusion_kernel(self, radius):
        """创建高斯扩散核"""
        x = torch.arange(-radius, radius + 1, dtype=torch.float32)
        kernel = torch.exp(-x**2 / (2 * (radius/2)**2))
        return kernel / kernel.sum()

    def sample_paths(self, pos_pheromone, count_pheromone, epsilon=None):
        """
        分离采样策略
        
        参数:
            pos_pheromone: [seq_len] 位置信息素
            count_pheromone: [max_insert+1] 数量信息素
            epsilon: 探索率
        """
        if epsilon is None:
            epsilon = self.epsilon
        
        seq_len = pos_pheromone.size(0)
        positions = []
        counts = []
        
        # === 采样位置 ===
        pos_prob = pos_pheromone.clone()
        pos_prob = pos_prob / (pos_prob.sum() + 1e-12)
        
        # epsilon-greedy
        uniform_pos = torch.ones_like(pos_prob) / seq_len
        pos_prob = (1 - epsilon) * pos_prob + epsilon * uniform_pos
        
        # 无放回采样patches个位置
        sampled_positions = torch.multinomial(pos_prob, self.patches, replacement=False)
        positions = sampled_positions.tolist()
        
        # === 采样数量 ===
        count_prob = count_pheromone.clone()
        count_prob = count_prob / (count_prob.sum() + 1e-12)
        
        uniform_count = torch.ones_like(count_prob) / (self.max_insert + 1)
        count_prob = (1 - epsilon) * count_prob + epsilon * uniform_count
        
        # 为每个位置独立采样数量
        for _ in range(self.patches):
            count = torch.multinomial(count_prob, 1).item()
            counts.append(count)
        
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

    def local_search_parallel(self, base, traces, labels, chunk_size=64):
        """并行局部搜索 - 评估所有邻域"""
        best = base.clone().cpu()
        best_score = self._eval_single(best.to(self.device), traces, labels, chunk_size)
        seq_len = traces.size(1)
        
        improved = True
        while improved:
            improved = False
            candidates = []
            
            # 生成所有邻域解
            for patch_idx in range(len(best)):
                # 位置邻域 (±1, ±2, ±3)
                for delta in [-3, -2, -1, 1, 2, 3]:
                    cand = best.clone()
                    new_pos = torch.clamp(cand[patch_idx, 0] + delta, 0, seq_len - 1)
                    if new_pos != cand[patch_idx, 0]:
                        cand[patch_idx, 0] = new_pos
                        candidates.append(cand)
                
                # 数量邻域 (±1)
                for delta in [-1, 1]:
                    cand = best.clone()
                    new_cnt = torch.clamp(cand[patch_idx, 1] + delta, 0, self.max_insert)
                    if new_cnt != cand[patch_idx, 1]:
                        cand[patch_idx, 1] = new_cnt
                        candidates.append(cand)
            
            # 批量评估
            if candidates:
                scores = self.evaluate_paths(candidates, traces, labels, chunk_size)
                best_idx = scores.argmax()
                
                if scores[best_idx] > best_score:
                    best = candidates[best_idx].clone()
                    best_score = scores[best_idx].item()
                    improved = True
        
        return best, best_score

    def _update_pheromone_with_diffusion(self, pos_pheromone, count_pheromone, 
                                         solutions, scores, seq_len):
        """
        更新信息素 - 带邻域扩散
        """
        # 标准化分数
        scores_arr = np.array(scores)
        if scores_arr.std() < 1e-6:
            rewards = np.ones_like(scores_arr) / len(scores_arr)
        else:
            mean_s, std_s = scores_arr.mean(), scores_arr.std()
            rewards = 1.0 / (1.0 + np.exp(-(scores_arr - mean_s) / std_s))
            rewards = rewards / rewards.sum()
        
        # === 更新位置信息素 (带扩散) ===
        pos_delta = torch.zeros(seq_len)
        for sol, r in zip(solutions, rewards):
            for pos, _ in sol.tolist():
                pos = int(pos)
                # 中心位置
                pos_delta[pos] += r
                
                # 邻域扩散 (高斯衰减)
                for offset in range(1, self.diffusion_radius + 1):
                    weight = r * np.exp(-offset**2 / (2 * (self.diffusion_radius/2)**2))
                    if pos - offset >= 0:
                        pos_delta[pos - offset] += weight
                    if pos + offset < seq_len:
                        pos_delta[pos + offset] += weight
        
        if pos_delta.sum() > 0:
            pos_delta = pos_delta / pos_delta.sum()
        
        # === 更新数量信息素 ===
        count_delta = torch.zeros(self.max_insert + 1)
        for sol, r in zip(solutions, rewards):
            for _, cnt in sol.tolist():
                cnt = int(cnt)
                count_delta[cnt] += r
                # 邻域扩散
                if cnt > 0:
                    count_delta[cnt - 1] += r * 0.3
                if cnt < self.max_insert:
                    count_delta[cnt + 1] += r * 0.3
        
        if count_delta.sum() > 0:
            count_delta = count_delta / count_delta.sum()
        
        # === 信息素更新 ===
        pos_pheromone = (1 - self.rho) * pos_pheromone + self.gamma * pos_delta
        pos_pheromone = pos_pheromone.clamp(min=1e-9)
        pos_pheromone = pos_pheromone / pos_pheromone.sum()
        
        count_pheromone = (1 - self.rho) * count_pheromone + self.gamma * count_delta
        count_pheromone = count_pheromone.clamp(min=1e-9)
        count_pheromone = count_pheromone / count_pheromone.sum()
        
        return pos_pheromone, count_pheromone

    def run(self, traces, labels, chunk_size=64):
        """
        主流程
        """
        seq_len, B = traces.size(1), traces.size(0)
        
        # 初始化: 分离的信息素
        pos_pheromone = torch.ones(seq_len) / seq_len
        count_pheromone = torch.ones(self.max_insert + 1) / (self.max_insert + 1)
        
        best_global = None
        best_score_global = -1
        no_improve = 0
        self.best_history = []

        for it in range(self.max_iters):
            # 1. 采样
            solutions = [self.sample_paths(pos_pheromone, count_pheromone) 
                        for _ in range(self.num_ants)]
            
            # 2. 评估
            scores = self.evaluate_paths(solutions, traces, labels, chunk_size)
            
            # 3. 选top-5并局部搜索
            top5_idx = scores.argsort(descending=True)[:5]
            refined = []
            refined_scores = []
            
            for idx in top5_idx:
                sol, score = self.local_search_parallel(solutions[idx], traces, labels, chunk_size)
                refined.append(sol)
                refined_scores.append(score)
                
                # 记录历史
                self.best_history.append((sol.clone(), score))
            
            # 保持历史大小
            if len(self.best_history) > 50:
                self.best_history.sort(key=lambda x: x[1], reverse=True)
                self.best_history = self.best_history[:25]
            
            # 4. 更新全局最优
            best_idx = int(np.argmax(refined_scores))
            if refined_scores[best_idx] > best_score_global:
                best_score_global = refined_scores[best_idx]
                best_global = refined[best_idx].clone()
                no_improve = 0
                print(f"[Iter {it}] Best: {best_score_global:.0f}/{B} ({best_score_global/B*100:.1f}%)")
                print(f"  Positions: {best_global[:, 0].tolist()}")
                print(f"  Counts:    {best_global[:, 1].tolist()}")
            else:
                no_improve += 1
            
            # 5. 更新信息素 (使用历史最优解)
            if len(self.best_history) >= 5:
                # 使用历史top-10进行更新
                hist_sorted = sorted(self.best_history, key=lambda x: x[1], reverse=True)[:10]
                hist_sols = [s for s, _ in hist_sorted]
                hist_scores = [sc for _, sc in hist_sorted]
                pos_pheromone, count_pheromone = self._update_pheromone_with_diffusion(
                    pos_pheromone, count_pheromone, hist_sols, hist_scores, seq_len
                )
            else:
                pos_pheromone, count_pheromone = self._update_pheromone_with_diffusion(
                    pos_pheromone, count_pheromone, refined, refined_scores, seq_len
                )
            
            # 6. 动态调整探索率
            self.epsilon = max(0.05, 0.1 * (1 - it / self.max_iters))
            
            # 7. 重启机制
            if no_improve >= max(5, self.max_iters // 5):
                print(f"[Iter {it}] Restart - resetting pheromone")
                pos_pheromone = (pos_pheromone + torch.ones_like(pos_pheromone) / seq_len) / 2
                count_pheromone = (count_pheromone + torch.ones_like(count_pheromone) / (self.max_insert+1)) / 2
                self.best_history = self.best_history[:len(self.best_history)//2]
                no_improve = 0
            
            # 8. 提前终止
            if best_score_global >= B * 0.98:
                print(f"[Iter {it}] Early stop - achieved 98% success rate")
                break
        
        if best_global is None:
            best_global = self.sample_paths(pos_pheromone, count_pheromone, epsilon=1.0)
            best_score_global = 0
        
        return best_global, best_score_global


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