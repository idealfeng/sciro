"""
完整训练+推理流程
训练所有5个fold，然后ensemble预测
"""

import subprocess
import sys
from pathlib import Path
from config import Config


def train_all_folds():
    """训练所有fold"""
    config = Config()

    # 创建输出目录（本地运行需要）
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("TRAINING ALL FOLDS")
    print("=" * 70 + "\n")

    for fold in range(config.N_FOLDS):
        print(f"\n{'='*70}")
        print(f"TRAINING FOLD {fold+1}/{config.N_FOLDS}")
        print(f"{'='*70}\n")

        # 修改config的fold
        config.FOLD = fold

        # 训练这个fold
        from train import Trainer

        trainer = Trainer(config)
        best_score = trainer.train()

        print(f"\nFold {fold+1} completed with score: {best_score:.4f}")

        # 清理内存
        import gc
        import torch

        del trainer
        gc.collect()
        torch.cuda.empty_cache()


def run_inference():
    """运行推理"""
    print("\n" + "=" * 70)
    print("RUNNING INFERENCE")
    print("=" * 70 + "\n")

    from inference import main as inference_main

    inference_main()


def main():
    # 1. 训练所有fold
    train_all_folds()

    # 2. 推理
    run_inference()

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
