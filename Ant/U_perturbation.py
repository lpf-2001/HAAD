import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm


# ==============================
# 工具函数
# ==============================
def patch_key(patch):
    if isinstance(patch, torch.Tensor):
        patch = patch.cpu().tolist()
    return f"{int(patch[0])}_{int(patch[1])}"

def key_patch(key):
    a, b = key.split('_')
    return [int(a), int(b)]

def combo_key_from_patches(patches):
    keys = [patch_key(p) for p in patches]
    return "|".join(sorted(keys))


# ==============================
# 改进的通用扰动选择器
# ==============================
class ImprovedUniversalPatchSelector:
    """
    改进版通用扰动选择器，解决以下问题：
    1. 避免陷入单一强patch
    2. 考虑patch之间的冲突和协同
    3. 平衡全局和局部效果
    """
    def __init__(self, model, device, generate_adv_trace_fn):
        self.model = model
        self.device = device
        self.generate_adv_trace = generate_adv_trace_fn

        # 候选组合
        self.candidate_combos = {}  # combo_key -> {'patches': [[p]], 'seen_in_batches': set()}
        self.eval_batches = []      # [(x_cpu, y_cpu), ...]
        
        # 增益矩阵
        self.combo_batch_gains = defaultdict(dict)  # combo_key -> {batch_idx: gain}
        self.patch_batch_gains = defaultdict(dict)  # patch_key -> {batch_idx: gain}
        
        # 全局最优
        self.global_best_combo = None
        self.global_best_gain = -float('inf')
        
        # 局部最优
        self.batch_best_combos = {}  # batch_idx -> combo_key
        
        # 冲突检测相关
        self.patch_stats = {}  # patch_key -> {mean, std, CI, S_pos, S_neg, score}
        self.pairwise_conflicts = {}  # (p1, p2) -> conflict_score
        
        self._eps = 1e-8

    # ----------------------------------------------------------
    # 数据收集
    # ----------------------------------------------------------
    def add_candidate_combo(self, patches, origin_batch_id=None):
        """添加候选组合"""
        patches_list = [p.cpu().tolist() if isinstance(p, torch.Tensor) else list(p) 
                       for p in patches]
        ck = combo_key_from_patches(patches_list)
        if ck not in self.candidate_combos:
            self.candidate_combos[ck] = {
                'patches': patches_list, 
                'seen_in_batches': set()
            }
        if origin_batch_id is not None:
            self.candidate_combos[ck]['seen_in_batches'].add(origin_batch_id)
        return ck

    def add_eval_batch(self, x_batch, y_batch):
        """加入评估batch"""
        self.eval_batches.append((
            x_batch.detach().cpu().clone(), 
            y_batch.detach().cpu().clone()
        ))
        return len(self.eval_batches) - 1

    # ----------------------------------------------------------
    # 评估函数
    # ----------------------------------------------------------
    def _eval_patches_on_batch(self, patches_list, x_batch_cpu, y_batch_cpu):
        """在单batch上评估patches的gain"""
        x = x_batch_cpu.float().to(self.device)
        y = y_batch_cpu.to(self.device)
        
        with torch.no_grad():
            base_pred = self.model(x)
        base_acc = (y.argmax(1) == base_pred.argmax(1)).float().mean().item()

        p_tensor = torch.tensor(patches_list, dtype=torch.int64, device=self.device)
        with torch.no_grad():
            adv = self.generate_adv_trace(p_tensor, x)
            adv_pred = self.model(adv)
        adv_acc = (y.argmax(1) == adv_pred.argmax(1)).float().mean().item()

        return float(base_acc - adv_acc)

    # ----------------------------------------------------------
    # 步骤1: 评估所有组合，找到全局和局部最优
    # ----------------------------------------------------------
    def evaluate_all_candidates(self, verbose=False):
        """
        评估所有candidate combo在每个batch上的表现
        同时记录：
        1. 全局最优组合
        2. 每个batch的局部最优组合
        """
        print("\n[Step 1] 评估所有候选组合...")
        
        for ck, meta in tqdm(self.candidate_combos.items(), desc='评估组合'):
            patches = meta['patches']
            gains = []
            
            for eval_idx, (x_b, y_b) in enumerate(self.eval_batches):
                if eval_idx in self.combo_batch_gains[ck]:
                    gain = self.combo_batch_gains[ck][eval_idx]
                else:
                    gain = self._eval_patches_on_batch(patches, x_b, y_b)
                    self.combo_batch_gains[ck][eval_idx] = gain
                gains.append(gain)
                
                # 更新batch局部最优
                if eval_idx not in self.batch_best_combos or \
                   gain > self.combo_batch_gains[self.batch_best_combos[eval_idx]][eval_idx]:
                    self.batch_best_combos[eval_idx] = ck
            
            # 更新全局最优（使用平均gain）
            avg_gain = np.mean(gains)
            if avg_gain > self.global_best_gain:
                self.global_best_gain = avg_gain
                self.global_best_combo = ck
            
            if verbose:
                print(f"  {ck}: mean={np.mean(gains):.4f}, min={np.min(gains):.4f}")
        
        print(f"\n全局最优组合: {self.global_best_combo}")
        print(f"全局最优平均gain: {self.global_best_gain:.4f}")
        print(f"\n各batch局部最优:")
        for bid, ck in sorted(self.batch_best_combos.items()):
            gain = self.combo_batch_gains[ck][bid]
            print(f"  Batch {bid}: {ck} (gain={gain:.4f})")

    # ----------------------------------------------------------
    # 步骤2: 评估单个patch的表现
    # ----------------------------------------------------------
    def evaluate_individual_patches(self, verbose=False):
        """
        对每个patch单独评估在每个batch上的gain
        这是为了计算patch的独立效果（非组合效果）
        """
        print("\n[Step 2] 评估单个patch...")
        
        # 收集所有unique patches
        all_patches = set()
        for meta in self.candidate_combos.values():
            for p in meta['patches']:
                all_patches.add(tuple(p))
        
        for patch_tuple in tqdm(all_patches, desc='评估单patch'):
            pk = patch_key(list(patch_tuple))
            
            for eval_idx, (x_b, y_b) in enumerate(self.eval_batches):
                if eval_idx in self.patch_batch_gains[pk]:
                    continue
                
                # 评估单个patch
                gain = self._eval_patches_on_batch([list(patch_tuple)], x_b, y_b)
                self.patch_batch_gains[pk][eval_idx] = gain
            
            if verbose:
                gains = list(self.patch_batch_gains[pk].values())
                print(f"  {pk}: mean={np.mean(gains):.4f}, std={np.std(gains):.4f}")

    # ----------------------------------------------------------
    # 步骤3: 计算patch得分（全局+局部+冲突检测）
    # ----------------------------------------------------------
    def compute_patch_scores_v2(self, 
                                 global_weight=0.4,
                                 local_weight=0.3, 
                                 individual_weight=0.3,
                                 use_conflict_penalty=True,
                                 mu_threshold=0.0,
                                 ci_threshold=0.15):
        """
        改进的patch评分方法，综合考虑：
        1. 全局最优组合中的贡献 (global_weight)
        2. 局部最优组合中的贡献 (local_weight)
        3. 单独作用效果 (individual_weight)
        4. 冲突指数惩罚
        
        返回: patch_scores = {patch_key: {score, details}}
        """
        print("\n[Step 3] 计算patch综合得分...")
        
        # 1. 计算单patch统计指标（μ, σ, CI）
        self._compute_patch_statistics()
        
        # 2. 收集所有patches
        all_patches = set()
        for meta in self.candidate_combos.values():
            for p in meta['patches']:
                all_patches.add(patch_key(p))
        
        patch_scores = {}
        
        for pk in tqdm(all_patches, desc='计算综合得分'):
            patch_stats = self.patch_stats.get(pk, {})
            
            # === 分量1: 全局贡献 ===
            global_contribution = 0.0
            if self.global_best_combo:
                global_patches = self.candidate_combos[self.global_best_combo]['patches']
                if any(patch_key(p) == pk for p in global_patches):
                    # 该patch在全局最优中
                    # 计算去掉它后的效果下降
                    global_contribution = self._compute_removal_impact(
                        self.global_best_combo, pk
                    )
            
            # === 分量2: 局部贡献 ===
            local_contributions = []
            for bid, best_ck in self.batch_best_combos.items():
                local_patches = self.candidate_combos[best_ck]['patches']
                if any(patch_key(p) == pk for p in local_patches):
                    impact = self._compute_removal_impact(best_ck, pk, batch_idx=bid)
                    local_contributions.append(impact)
            local_contribution = np.mean(local_contributions) if local_contributions else 0.0
            
            # === 分量3: 独立效果 ===
            individual_gains = list(self.patch_batch_gains.get(pk, {}).values())
            individual_mean = np.mean(individual_gains) if individual_gains else 0.0
            
            # === 综合得分（加权） ===
            base_score = (global_weight * global_contribution + 
                         local_weight * local_contribution +
                         individual_weight * individual_mean)
            
            # === 冲突惩罚 ===
            conflict_penalty = 0.0
            if use_conflict_penalty:
                CI = patch_stats.get('CI', 0)
                mu = patch_stats.get('mean', 0)
                
                # CI惩罚
                if CI > ci_threshold:
                    conflict_penalty += (CI - ci_threshold) * 0.5
                
                # 负向惩罚
                if mu < mu_threshold:
                    conflict_penalty += abs(mu) * 0.3
            
            final_score = base_score - conflict_penalty
            
            # 保存详细信息
            patch_scores[pk] = {
                'score': final_score,
                'global_contribution': global_contribution,
                'local_contribution': local_contribution,
                'individual_mean': individual_mean,
                'conflict_penalty': conflict_penalty,
                'CI': patch_stats.get('CI', 0),
                'mean': patch_stats.get('mean', 0),
                'std': patch_stats.get('std', 0),
            }
        
        self._patch_scores = patch_scores
        return patch_scores

    def _compute_removal_impact(self, combo_key, patch_to_remove, batch_idx=None):
        """
        计算从组合中移除某个patch的影响
        impact = combo_gain - (combo_without_patch_gain)
        
        如果batch_idx指定，只在该batch上计算；否则计算平均
        """
        combo_patches = self.candidate_combos[combo_key]['patches']
        remaining = [p for p in combo_patches if patch_key(p) != patch_to_remove]
        
        if len(remaining) == 0:
            # 如果移除后为空，impact就是原combo的gain
            if batch_idx is not None:
                return self.combo_batch_gains[combo_key].get(batch_idx, 0)
            else:
                gains = self.combo_batch_gains[combo_key].values()
                return np.mean(list(gains)) if gains else 0
        
        # 尝试找到remaining的combo
        remaining_key = combo_key_from_patches(remaining)
        
        if batch_idx is not None:
            # 单batch计算
            full_gain = self.combo_batch_gains[combo_key].get(batch_idx, 0)
            
            if remaining_key in self.combo_batch_gains:
                remaining_gain = self.combo_batch_gains[remaining_key].get(batch_idx, 0)
            else:
                # 需要临时评估
                x_b, y_b = self.eval_batches[batch_idx]
                remaining_gain = self._eval_patches_on_batch(remaining, x_b, y_b)
            
            return full_gain - remaining_gain
        else:
            # 跨所有batch平均
            impacts = []
            for bid in self.combo_batch_gains[combo_key].keys():
                full_gain = self.combo_batch_gains[combo_key][bid]
                
                if remaining_key in self.combo_batch_gains:
                    remaining_gain = self.combo_batch_gains[remaining_key].get(bid, 0)
                else:
                    x_b, y_b = self.eval_batches[bid]
                    remaining_gain = self._eval_patches_on_batch(remaining, x_b, y_b)
                
                impacts.append(full_gain - remaining_gain)
            
            return np.mean(impacts) if impacts else 0

    def _compute_patch_statistics(self):
        """计算每个patch的统计指标：μ, σ, CI"""
        for pk, batch_gains_dict in self.patch_batch_gains.items():
            gains = np.array(list(batch_gains_dict.values()))
            
            if len(gains) == 0:
                continue
            
            mu = np.mean(gains)
            std = np.std(gains)
            var = np.var(gains)
            
            # S+ 和 S-
            S_pos = np.sum(np.maximum(gains, 0))
            S_neg = np.sum(np.maximum(-gains, 0))
            
            # 冲突指数
            CI = (S_pos * S_neg) / ((S_pos + S_neg) ** 2 + self._eps)
            
            self.patch_stats[pk] = {
                'mean': mu,
                'std': std,
                'variance': var,
                'CI': CI,
                'S_pos': S_pos,
                'S_neg': S_neg,
                'sample_count': len(gains)
            }

    # ----------------------------------------------------------
    # 步骤4: 多样性选择top-k
    # ----------------------------------------------------------
    def select_topk_diverse(self, k=10, overlap_threshold=0.5, 
                           strategy='greedy_diverse'):
        """
        从高分patches中选择top-k，同时保证多样性
        
        策略:
        - 'greedy_diverse': 贪心选择，避免高重叠
        - 'conflict_aware': 考虑成对冲突
        """
        if not hasattr(self, '_patch_scores'):
            raise RuntimeError("请先调用 compute_patch_scores_v2()")
        
        print(f"\n[Step 4] 选择top-{k} patches (策略: {strategy})...")
        
        # 按分数排序
        sorted_patches = sorted(
            self._patch_scores.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )
        
        if strategy == 'greedy_diverse':
            selected = self._greedy_diverse_selection(
                sorted_patches, k, overlap_threshold
            )
        elif strategy == 'conflict_aware':
            selected = self._conflict_aware_selection(
                sorted_patches, k, overlap_threshold
            )
        else:
            # 简单top-k
            selected = [key_patch(pk) for pk, _ in sorted_patches[:k]]
        
        print(f"\n选中的{len(selected)}个patches:")
        for i, patch in enumerate(selected, 1):
            pk = patch_key(patch)
            info = self._patch_scores[pk]
            print(f"  {i}. {patch}: score={info['score']:.4f}, "
                  f"μ={info['mean']:.4f}, CI={info['CI']:.4f}")
        
        return selected

    def _calculate_overlap(self, patch1, patch2):
        """计算两个patch的位置重叠度"""
        pos1, num1 = patch1[0], patch1[1]
        pos2, num2 = patch2[0], patch2[1]
        
        start1, end1 = pos1, pos1 + num1
        start2, end2 = pos2, pos2 + num2
        
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)
        overlap_len = max(0, overlap_end - overlap_start)
        
        total_len = max(end1, end2) - min(start1, start2)
        return overlap_len / total_len if total_len > 0 else 0

    def _greedy_diverse_selection(self, sorted_patches, k, overlap_threshold):
        """贪心多样性选择"""
        selected = []
        
        for pk, info in sorted_patches:
            if len(selected) >= k:
                break
            
            patch = key_patch(pk)
            
            # 检查与已选择patches的重叠
            has_high_overlap = False
            for selected_pk in selected:
                selected_patch = key_patch(selected_pk)
                overlap = self._calculate_overlap(patch, selected_patch)
                
                if overlap > overlap_threshold:
                    has_high_overlap = True
                    break
            
            if not has_high_overlap:
                selected.append(pk)
        
        return [key_patch(pk) for pk in selected]

    def _conflict_aware_selection(self, sorted_patches, k, overlap_threshold):
        """考虑冲突的选择（可以在这里加入成对冲突检测）"""
        # 简化版：先用greedy_diverse，然后检查成对冲突
        candidates = self._greedy_diverse_selection(
            sorted_patches, k*2, overlap_threshold
        )
        
        # TODO: 可以在这里加入成对冲突检测逻辑
        # 目前简化为直接返回前k个
        return candidates[:k]

    # ----------------------------------------------------------
    # 报告生成
    # ----------------------------------------------------------
    def print_analysis_report(self):
        """打印详细的分析报告"""
        print("\n" + "="*80)
        print("Patch分析报告")
        print("="*80)
        
        # 全局统计
        print(f"\n【全局统计】")
        print(f"  候选组合数: {len(self.candidate_combos)}")
        print(f"  评估batch数: {len(self.eval_batches)}")
        print(f"  唯一patches数: {len(self.patch_stats)}")
        
        # 全局最优
        if self.global_best_combo:
            print(f"\n【全局最优组合】")
            print(f"  组合: {self.global_best_combo}")
            global_patches = self.candidate_combos[self.global_best_combo]['patches']
            print(f"  包含patches: {global_patches}")
            print(f"  平均gain: {self.global_best_gain:.4f}")
        
        # Patch统计摘要
        if self.patch_stats:
            print(f"\n【Patch统计摘要】")
            means = [s['mean'] for s in self.patch_stats.values()]
            CIs = [s['CI'] for s in self.patch_stats.values()]
            stds = [s['std'] for s in self.patch_stats.values()]
            
            print(f"  平均gain: {np.mean(means):.4f} (min={np.min(means):.4f}, max={np.max(means):.4f})")
            print(f"  平均CI: {np.mean(CIs):.4f}")
            print(f"  平均std: {np.mean(stds):.4f}")
            
            high_conflict = sum(1 for ci in CIs if ci > 0.15)
            negative = sum(1 for m in means if m < 0)
            print(f"  高冲突patches (CI>0.15): {high_conflict} ({high_conflict/len(CIs)*100:.1f}%)")
            print(f"  负向patches (μ<0): {negative} ({negative/len(means)*100:.1f}%)")
        
        print("="*80)


# ==============================
# 主函数接口
# ==============================
def run_improved_universal_selector(S_model, device, generate_adv_trace_fn, 
                                    haad_results, eval_batches, 
                                    k=10, 
                                    global_weight=0.4,
                                    local_weight=0.3,
                                    individual_weight=0.3,
                                    overlap_threshold=0.5,
                                    strategy='greedy_diverse'):
    """
    改进的通用patch选择流程
    
    参数:
        haad_results: list of patches，每个是某batch的最优组合
        eval_batches: list of (x, y)
        k: 选择的patch数量
        global_weight: 全局贡献权重
        local_weight: 局部贡献权重
        individual_weight: 独立效果权重
        overlap_threshold: 位置重叠阈值
        strategy: 选择策略 ('greedy_diverse' 或 'conflict_aware')
    
    返回:
        top_patches: list of [pos, num]
    """
    selector = ImprovedUniversalPatchSelector(S_model, device, generate_adv_trace_fn)

    # Step 1: 添加所有候选组合
    for i, patches in enumerate(haad_results):
        selector.add_candidate_combo(patches, origin_batch_id=i)

    # Step 2: 添加评估batches
    for xb, yb in eval_batches:
        selector.add_eval_batch(xb, yb)

    # Step 3: 评估所有组合，找全局和局部最优
    selector.evaluate_all_candidates(verbose=False)
    
    # Step 4: 评估单个patches
    selector.evaluate_individual_patches(verbose=False)
    
    # Step 5: 计算综合得分
    selector.compute_patch_scores_v2(
        global_weight=global_weight,
        local_weight=local_weight,
        individual_weight=individual_weight,
        use_conflict_penalty=True
    )
    
    # Step 6: 多样性选择top-k
    top_patches = selector.select_topk_diverse(
        k=k,
        overlap_threshold=overlap_threshold,
        strategy=strategy
    )
    
    # 打印分析报告
    selector.print_analysis_report()
    
    return top_patches