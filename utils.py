import torch
import numpy as np
import random
from sklearn.metrics import r2_score


def set_seed(seed=42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def weighted_r2_score(y_true_dict, y_pred_dict, weights):
    """
    计算加权R²分数

    Args:
        y_true_dict: {target_name: array of true values}
        y_pred_dict: {target_name: array of predictions}
        weights: {target_name: weight}

    Returns:
        weighted R² score
    """
    # 展平所有预测和真实值
    all_true = []
    all_pred = []
    all_weights = []

    for target_name in weights.keys():
        true_vals = y_true_dict[target_name]
        pred_vals = y_pred_dict[target_name]
        weight = weights[target_name]

        all_true.extend(true_vals)
        all_pred.extend(pred_vals)
        all_weights.extend([weight] * len(true_vals))

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    all_weights = np.array(all_weights)

    # 加权均值
    weighted_mean = np.sum(all_weights * all_true) / np.sum(all_weights)

    # SS_res 和 SS_tot
    ss_res = np.sum(all_weights * (all_true - all_pred) ** 2)
    ss_tot = np.sum(all_weights * (all_true - weighted_mean) ** 2)

    # R²
    r2 = 1 - ss_res / (ss_tot + 1e-8)

    return r2


class WeightedR2Loss(torch.nn.Module):
    """加权R2损失函数（负R2 + Clover MSE 辅助项）"""

    def __init__(self, target_weights):
        super().__init__()
        self.target_weights = target_weights
        self.mse_loss = torch.nn.MSELoss()

    def forward(self, predictions, targets):
        # 原来的R2损失（负R2）
        r2_loss = self._compute_r2_loss(predictions, targets)

        # 为Clover添加额外的MSE loss（因为它很稀疏）
        clover_mse = self.mse_loss(
            predictions['Dry_Clover_g'],
            targets['Dry_Clover_g'],
        )

        return r2_loss + 0.1 * clover_mse

    def _compute_r2_loss(self, predictions, targets):
        # 展平
        all_preds = []
        all_targets = []
        all_weights = []

        for target_name in self.target_weights.keys():
            preds = predictions[target_name]
            targs = targets[target_name]
            weight = self.target_weights[target_name]

            all_preds.append(preds)
            all_targets.append(targs)
            all_weights.append(torch.full_like(targs, weight))

        preds = torch.cat(all_preds)
        targs = torch.cat(all_targets)
        ws = torch.cat(all_weights)

        # 加权均值
        weighted_mean = (ws * targs).sum() / ws.sum()

        # SS_res 和 SS_tot
        ss_res = (ws * (targs - preds) ** 2).sum()
        ss_tot = (ws * (targs - weighted_mean) ** 2).sum()

        # R2
        r2 = 1 - ss_res / (ss_tot + 1e-8)

        # 返回负值（最小化）
        return -r2

class ConsistencyLoss(torch.nn.Module):
    """物理一致性损失：Total ≈ Green + Dead + Clover"""

    def __init__(self, weight=0.1):
        super().__init__()
        self.weight = weight

    def forward(self, predictions):
        """
        Args:
            predictions: dict {target_name: (B,)}
        """
        predicted_total = predictions["Dry_Total_g"]
        sum_components = (
            predictions["Dry_Green_g"]
            + predictions["Dry_Dead_g"]
            + predictions["Dry_Clover_g"]
        )

        # L1 loss
        consistency = torch.abs(predicted_total - sum_components).mean()

        return self.weight * consistency


class AverageMeter:
    """追踪平均值和当前值"""

    def __init__(self):
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


class EarlyStopping:
    """Early stopping handler"""

    def __init__(self, patience=5, min_delta=1e-4, mode="max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "max":
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        else:
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
            return True

        return False
