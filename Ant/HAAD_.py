import torch
import torch.nn as nn

# 全局设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 扰动插入函数
# ============================================================
def generate_adv_trace(perturbations: torch.Tensor, traces: torch.Tensor) -> torch.Tensor:
    """
    在 traces 中插入对抗扰动。
    参数:
        perturbations: [m,2] tensor, 每行 [插入位置, 插入数量]
        traces: [B, seq_len, 1] 原始流
    返回:
        new_traces: [B, seq_len, 1] 添加扰动后的流 (保持原长度)
    """
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


# ============================================================
# 改进版 HAAD
# ============================================================
class HAAD:
    def __init__(self, model: nn.Module, num_ants: int, max_insert: int = 6,
                 patches: int = 8, max_iters: int = 10):
        self.model = model.to(device)
        self.num_ants = num_ants
        self.max_insert = max_insert
        self.patches = patches
        self.max_iters = max_iters

    def sample_paths(self, pheromone: torch.Tensor, epsilon: float = 0.1):
        """ 根据信息素采样一条路径 (加入 ε-探索避免陷入局部极值) """
        prob = pheromone.view(-1)
        prob = prob / (prob.sum() + 1e-12)

        # ε-greedy: 加入均匀分布噪声
        uniform = torch.ones_like(prob) / prob.numel()
        prob = (1 - epsilon) * prob + epsilon * uniform

        idx = torch.multinomial(prob, num_samples=self.patches, replacement=False)
        insert_pos = idx // pheromone.size(1)
        insert_cnt = idx % pheromone.size(1)
        return torch.stack([insert_pos.long(), insert_cnt.long()], dim=1)  # [patches,2]

    def evaluate_paths(self, path_solutions, traces, labels):
        """ 批量评估所有蚂蚁 """
        B = traces.size(0)
        adv_traces = [generate_adv_trace(paths.to(device), traces) for paths in path_solutions]
        big_batch = torch.cat(adv_traces, dim=0)  # [num_ants*B, seq_len, 1]

        with torch.no_grad():
            preds = self.model(big_batch)

        adv_nums = []
        for i, paths in enumerate(path_solutions):
            preds_i = preds[i*B:(i+1)*B]
            adv_num = (preds_i.argmax(-1) != labels.argmax(-1)).sum().item()
            adv_nums.append(adv_num)
        return torch.tensor(adv_nums, dtype=torch.float32, device="cpu")

    def local_search(self, base_paths, traces, labels, max_trials=5):
        """ 局部搜索（hill climbing） """
        best_paths = base_paths.clone().to(device)
        with torch.no_grad():
            adv = generate_adv_trace(best_paths, traces)
            preds = self.model(adv)
            best_score = (preds.argmax(-1) != labels.argmax(-1)).sum().item()

        for _ in range(max_trials):
            cand = best_paths.clone()
            idx = torch.randint(0, cand.size(0), (1,)).item()
            if torch.rand(1).item() < 0.5:  # 调整位置
                delta = 1 if torch.rand(1).item() < 0.5 else -1
                cand[idx, 0] = torch.clamp(cand[idx, 0] + delta, 0, traces.size(1)-1)
            else:  # 调整插入数量
                delta = 1 if torch.rand(1).item() < 0.5 else -1
                cand[idx, 1] = torch.clamp(cand[idx, 1] + delta, 0, self.max_insert)

            with torch.no_grad():
                adv = generate_adv_trace(cand, traces)
                preds = self.model(adv)
                score = (preds.argmax(-1) != labels.argmax(-1)).sum().item()

            if score > best_score:
                best_paths, best_score = cand.clone(), score

        return best_paths.cpu(), float(best_score)

    def run(self, traces, labels, rho=0.15, elite_k=5, epsilon=0.1, restart_patience=10):
        """
        主流程：避免局部最优的改进版 HAAD
        """
        seq_len = traces.size(1)
        B = traces.size(0)

        # 初始化信息素
        pheromone_sum = torch.ones((seq_len, self.max_insert + 1), dtype=torch.float32) / (self.max_insert + 1)
        pheromone_all = pheromone_sum.unsqueeze(0).repeat(self.num_ants, 1, 1)

        best_score, best_paths = -1, None
        no_improve_rounds = 0

        for it in range(self.max_iters):
            # 1) 采样路径
            path_solutions = [self.sample_paths(pheromone_sum, epsilon=epsilon) for _ in range(self.num_ants)]

            # 2) 批量评估
            adv_nums = self.evaluate_paths(path_solutions, traces.to(device), labels.to(device))

            # 3) 局部搜索 top-k
            topk = min(elite_k, self.num_ants)
            top_vals, top_idx = torch.topk(adv_nums, topk)
            for idx in top_idx.tolist():
                new_paths, new_score = self.local_search(path_solutions[idx], traces.to(device), labels.to(device))
                if new_score > adv_nums[idx]:
                    path_solutions[idx] = new_paths
                    adv_nums[idx] = new_score

            # 4) 更新信息素
            delta_all = torch.zeros_like(pheromone_all)
            for ant_idx, paths in enumerate(path_solutions):
                adv_rate = adv_nums[ant_idx].item() / B
                for pos, cnt in paths.tolist():
                    delta_all[ant_idx, pos, cnt] += adv_rate

            pheromone_all = (1 - rho) * pheromone_all + delta_all
            pheromone_sum = pheromone_all.mean(dim=0)

            # 5) 记录最优解
            best_idx = adv_nums.argmax().item()
            if adv_nums[best_idx] > best_score:
                best_score = adv_nums[best_idx].item()
                best_paths = path_solutions[best_idx].clone()
                no_improve_rounds = 0
            else:
                no_improve_rounds += 1

            # 6) 动态重启机制：长时间没有改进则重置部分信息素
            if no_improve_rounds >= restart_patience:
                pheromone_sum = torch.ones_like(pheromone_sum) / (self.max_insert + 1)
                pheromone_all = pheromone_sum.unsqueeze(0).repeat(self.num_ants, 1, 1)
                no_improve_rounds = 0
                print(f"[Restart] Iter {it}: 重置信息素避免局部最优")

            print(f"[Iter {it}] Best adv_num={best_score}/{B}")

            # 早停
            if best_score >= B * 0.98:
                break

        return best_paths
