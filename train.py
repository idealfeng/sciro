import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
import pandas as pd
from pathlib import Path
import gc

from config import Config
from dataset import get_dataloaders
from models import BiomassPredictor
from utils import (
    set_seed,
    weighted_r2_score,
    WeightedR2Loss,
    ConsistencyLoss,
    AverageMeter,
    EarlyStopping,
)


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = config.DEVICE

        # 设置随机种子
        set_seed(config.SEED)

        # 初始化模型
        self.model = BiomassPredictor(
            model_name=config.MODEL_NAME,
            num_tabular_features=10,  # 根据dataset.py中的特征数
            pretrained=config.PRETRAINED,
            dropout_rate=0.4,
        ).to(self.device)

        # 损失函数
        self.r2_loss = WeightedR2Loss(config.TARGET_WEIGHTS)
        self.consistency_loss = ConsistencyLoss(weight=0.15)

        # 优化器和调度器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,  # 10个epoch后第一次重启
            T_mult=2,  # 每次重启周期翻倍
            eta_min=1e-6,
        )

        # 混合精度训练
        self.scaler = GradScaler()

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.PATIENCE, min_delta=config.MIN_DELTA, mode="max"
        )

        # 最佳模型追踪
        self.best_score = -np.inf

        # DataLoaders
        self.train_loader, self.val_loader = get_dataloaders(config)

        print(f"Training on fold {config.FOLD}/{config.N_FOLDS}")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")
        print(f"Device: {self.device}")

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()

        # 损失追踪
        r2_losses = AverageMeter()
        consistency_losses = AverageMeter()
        total_losses = AverageMeter()

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.NUM_EPOCHS}")

        for batch_idx, batch in enumerate(pbar):
            images = batch["image"].to(self.device)
            tabular = batch["tabular"].to(self.device)
            targets = {k: v.to(self.device) for k, v in batch["targets"].items()}

            # 前向传播（混合精度）
            with autocast():
                predictions = self.model(images, tabular)

                # 计算损失
                r2_loss = self.r2_loss(predictions, targets)
                consistency_loss = self.consistency_loss(predictions)

                total_loss = r2_loss + consistency_loss

            # 反向传播
            self.optimizer.zero_grad()
            self.scaler.scale(total_loss).backward()

            # 梯度裁剪
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # 更新追踪
            batch_size = images.size(0)
            r2_losses.update(r2_loss.item(), batch_size)
            consistency_losses.update(consistency_loss.item(), batch_size)
            total_losses.update(total_loss.item(), batch_size)

            # 更新进度条
            pbar.set_postfix(
                {
                    "r2_loss": f"{r2_losses.avg:.4f}",
                    "cons_loss": f"{consistency_losses.avg:.4f}",
                    "total": f"{total_losses.avg:.4f}",
                    "lr": f'{self.optimizer.param_groups[0]["lr"]:.6f}',
                }
            )

        return {
            "r2_loss": r2_losses.avg,
            "consistency_loss": consistency_losses.avg,
            "total_loss": total_losses.avg,
        }

    @torch.no_grad()
    def validate(self, epoch):
        """验证"""
        self.model.eval()

        # 收集所有预测和真实值
        all_predictions = {name: [] for name in self.config.TARGET_NAMES}
        all_targets = {name: [] for name in self.config.TARGET_NAMES}

        r2_losses = AverageMeter()

        pbar = tqdm(self.val_loader, desc=f"Validation")

        for batch in pbar:
            images = batch["image"].to(self.device)
            tabular = batch["tabular"].to(self.device)
            targets = {k: v.to(self.device) for k, v in batch["targets"].items()}

            # 前向传播
            predictions = self.model(images, tabular)

            # 计算损失
            r2_loss = self.r2_loss(predictions, targets)
            r2_losses.update(r2_loss.item(), images.size(0))

            # 收集预测结果（转到CPU）
            for target_name in self.config.TARGET_NAMES:
                all_predictions[target_name].extend(
                    predictions[target_name].cpu().numpy()
                )
                all_targets[target_name].extend(targets[target_name].cpu().numpy())

            pbar.set_postfix({"r2_loss": f"{r2_losses.avg:.4f}"})

        # 计算加权R²分数
        weighted_r2 = weighted_r2_score(
            all_targets, all_predictions, self.config.TARGET_WEIGHTS
        )

        # 计算每个target的R²（用于分析）
        individual_r2 = {}
        for target_name in self.config.TARGET_NAMES:
            from sklearn.metrics import r2_score

            r2 = r2_score(all_targets[target_name], all_predictions[target_name])
            individual_r2[target_name] = r2

        print(f"\nValidation Results (Epoch {epoch+1}):")
        print(f"Weighted R²: {weighted_r2:.4f}")
        print("Individual R² scores:")
        for name, score in individual_r2.items():
            weight = self.config.TARGET_WEIGHTS[name]
            print(f"  {name:20s}: {score:.4f} (weight={weight})")

        return {
            "weighted_r2": weighted_r2,
            "individual_r2": individual_r2,
            "r2_loss": r2_losses.avg,
        }

    def save_checkpoint(self, epoch, val_metrics, is_best=False):
        """保存检查点"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "val_metrics": val_metrics,
            "config": self.config.__dict__,
        }

        # 保存最新的checkpoint
        checkpoint_path = (
            self.config.OUTPUT_DIR / f"checkpoint_fold{self.config.FOLD}_latest.pth"
        )
        torch.save(checkpoint, checkpoint_path)

        # 如果是最佳模型，额外保存
        if is_best:
            best_path = (
                self.config.OUTPUT_DIR / f"model_fold{self.config.FOLD}_best.pth"
            )
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")

    def train(self):
        """完整训练流程"""
        print("\n" + "=" * 50)
        print("Starting Training")
        print("=" * 50 + "\n")

        for epoch in range(self.config.NUM_EPOCHS):
            print(f"\nEpoch {epoch+1}/{self.config.NUM_EPOCHS}")
            print("-" * 50)

            # 训练
            train_metrics = self.train_epoch(epoch)

            # 验证
            val_metrics = self.validate(epoch)

            # 学习率调度
            self.scheduler.step()

            # 检查是否为最佳模型
            current_score = val_metrics["weighted_r2"]
            is_best = current_score > self.best_score

            if is_best:
                self.best_score = current_score
                print(f"\n🎉 New best score: {self.best_score:.4f}")

            # 保存checkpoint
            self.save_checkpoint(epoch, val_metrics, is_best=is_best)

            # Early stopping
            if self.early_stopping(current_score):
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break

            # 清理内存
            gc.collect()
            torch.cuda.empty_cache()

        print("\n" + "=" * 50)
        print(f"Training completed!")
        print(f"Best weighted R²: {self.best_score:.4f}")
        print("=" * 50 + "\n")

        return self.best_score


def main():
    config = Config()

    # 创建输出目录
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 训练
    trainer = Trainer(config)
    best_score = trainer.train()

    print(f"\nFold {config.FOLD} completed with best score: {best_score:.4f}")


if __name__ == "__main__":
    main()
