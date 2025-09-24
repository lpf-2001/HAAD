import torch

def patch_key(patch): 
    return f"{int(patch[0])}_{int(patch[1])}"
def key_patch(key):
    parts = key.split('_')
    return [int(parts[0]), int(parts[1])]    

class PerturbationEvaluator:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.position = {}  # 全局 patch -> score
    

    def _accuracy(self, pred, label):
        return (label.argmax(1) == pred.argmax(1)).float().mean().item()

    
    def evaluate_batch(self, x, y, patches, generate_adv_trace):
        batch_x = x.float().to(self.device)
        batch_y = y.float().to(self.device)

        # baseline
        origin_pred = self.model(batch_x)
        base_acc = self._accuracy(origin_pred, batch_y)

        patch_scores = {}

        # 1) 单 patch 效果
        for patch in patches:
            single_tensor = torch.tensor(patch, dtype=torch.int64, device=self.device).unsqueeze(0)
            adv_pred = self.model(generate_adv_trace(single_tensor, batch_x))
            adv_acc = self._accuracy(adv_pred, batch_y)
            delta = base_acc - adv_acc
            patch_scores[patch_key(patch)] = delta

        # 2) 组合效果
        if len(patches) > 1:
            combo_tensor = torch.tensor(patches, dtype=torch.int64, device=self.device)
            adv_pred_combo = self.model(generate_adv_trace(combo_tensor, batch_x))
            adv_acc_combo = self._accuracy(adv_pred_combo, batch_y)
            combo_delta = base_acc - adv_acc_combo

            # 按比例分摊
            total = sum(max(s, 0.0) for s in patch_scores.values()) + 1e-8
            for patch in patches:
                key = patch_key(patch)
                w = max(patch_scores[key], 0.0) / total if total > 0 else 1.0 / len(patches)
                patch_scores[key] += w * (combo_delta - total)
        else:
            combo_delta = 0.0

        # 3) 更新全局统计（带衰减）
        for patch, score in patch_scores.items():
            old = self.position.get(patch, 0.0)
            self.position[patch] = old * 0.9 + score  # 衰减避免过拟合单 batch

        return combo_delta


    def top_k(self, k):
        """返回前 k 个 patch (list of [pos, insert_num])"""
        sorted_item = sorted(self.position.items(), key=lambda item: item[1], reverse=True)
        return [key_patch(p) for p, _ in sorted_item[:k]]
