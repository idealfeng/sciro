import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
import gc

from config import Config
from dataset import BiomassDataset, get_transforms
from models import BiomassPredictor
from utils import set_seed


class Inferencer:
    def __init__(self, config, model_paths):
        """
        Args:
            config: Config对象
            model_paths: list of model checkpoint paths (for ensemble)
        """
        self.config = config
        self.device = config.DEVICE
        self.model_paths = model_paths

        set_seed(config.SEED)

        # 加载所有模型
        self.models = self._load_models()

        print(f"Loaded {len(self.models)} models for ensemble")

    def _load_models(self):
        """加载所有模型"""
        models = []

        for model_path in self.model_paths:
            # 创建模型
            model = BiomassPredictor(
                model_name=self.config.MODEL_NAME,
                num_tabular_features=10,
                pretrained=False,  # 不需要预训练权重
                dropout_rate=0.4,
            ).to(self.device)

            # 加载权重
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()

            models.append(model)

            print(f"Loaded model from {model_path}")
            if "val_metrics" in checkpoint:
                print(f"  Val R²: {checkpoint['val_metrics']['weighted_r2']:.4f}")

        return models

    @torch.no_grad()
    def predict(self, test_loader, use_tta=True, tta_times=5):
        """
        预测测试集

        Args:
            test_loader: DataLoader for test set
            use_tta: 是否使用Test-Time Augmentation
            tta_times: TTA次数

        Returns:
            dict of predictions {image_id: {target_name: value}}
        """
        all_predictions = {}

        pbar = tqdm(test_loader, desc="Predicting")

        for batch in pbar:
            images = batch["image"].to(self.device)
            tabular = batch["tabular"].to(self.device)
            image_ids = batch["image_id"]

            batch_size = images.size(0)

            # 存储所有模型的预测
            batch_preds = {
                target_name: torch.zeros(batch_size, device=self.device)
                for target_name in self.config.TARGET_NAMES
            }

            # Ensemble所有模型
            for model in self.models:
                if use_tta:
                    # Test-Time Augmentation
                    tta_preds = {
                        target_name: [] for target_name in self.config.TARGET_NAMES
                    }

                    for _ in range(tta_times):
                        # 应用不同的augmentation
                        aug_images = self._apply_tta(images)
                        preds = model(aug_images, tabular)

                        for target_name in self.config.TARGET_NAMES:
                            tta_preds[target_name].append(preds[target_name])

                    # 平均TTA结果
                    for target_name in self.config.TARGET_NAMES:
                        avg_pred = torch.mean(
                            torch.stack(tta_preds[target_name]), dim=0
                        )
                        batch_preds[target_name] += avg_pred
                else:
                    # 不使用TTA
                    preds = model(images, tabular)
                    for target_name in self.config.TARGET_NAMES:
                        batch_preds[target_name] += preds[target_name]

            # 平均所有模型的预测
            num_models = len(self.models)
            for target_name in self.config.TARGET_NAMES:
                batch_preds[target_name] /= num_models

            # 存储结果
            for i, image_id in enumerate(image_ids):
                all_predictions[image_id] = {
                    target_name: batch_preds[target_name][i].cpu().item()
                    for target_name in self.config.TARGET_NAMES
                }

        return all_predictions

    def _apply_tta(self, images):
        """应用Test-Time Augmentation"""
        # 随机水平翻转
        if torch.rand(1).item() > 0.5:
            images = torch.flip(images, dims=[3])

        # 随机垂直翻转
        if torch.rand(1).item() > 0.5:
            images = torch.flip(images, dims=[2])

        # 随机90度旋转
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            images = torch.rot90(images, k, dims=[2, 3])

        return images

    def create_submission(self, predictions, output_path):
        """
        创建提交文件

        Args:
            predictions: dict {image_id: {target_name: value}}
            output_path: 保存路径
        """
        submission_data = []

        for image_id, targets in predictions.items():
            for target_name in self.config.TARGET_NAMES:
                sample_id = f"{image_id}__{target_name}"
                value = targets[target_name]

                submission_data.append({"sample_id": sample_id, "target": value})

        # 创建DataFrame
        submission_df = pd.DataFrame(submission_data)

        # 按sample_id排序（确保顺序一致）
        submission_df = submission_df.sort_values("sample_id").reset_index(drop=True)

        # 保存
        submission_df.to_csv(output_path, index=False)

        print(f"\nSubmission saved to {output_path}")
        print(f"Total samples: {len(submission_df)}")
        print(f"Unique images: {len(predictions)}")

        # 显示预测统计
        print("\nPrediction statistics:")
        for target_name in self.config.TARGET_NAMES:
            values = [p[target_name] for p in predictions.values()]
            print(
                f"{target_name:20s}: "
                f"mean={np.mean(values):.2f}, "
                f"std={np.std(values):.2f}, "
                f"min={np.min(values):.2f}, "
                f"max={np.max(values):.2f}"
            )

        return submission_df


def main():
    config = Config()

    # 准备测试数据
    test_df = pd.read_csv(config.TEST_CSV)

    # 创建测试集Dataset
    test_dataset = BiomassDataset(
        test_df,
        config.TEST_IMG_DIR,
        transform=get_transforms(is_train=False, image_size=config.IMAGE_SIZE),
        is_test=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # 查找所有fold的best model
    model_paths = []
    for fold in range(config.N_FOLDS):
        model_path = config.OUTPUT_DIR / f"model_fold{fold}_best.pth"
        if model_path.exists():
            model_paths.append(model_path)

    if not model_paths:
        print("No trained models found! Please train models first.")
        return

    print(f"Found {len(model_paths)} trained models")

    # 创建Inferencer
    inferencer = Inferencer(config, model_paths)

    # 预测
    print("\n" + "=" * 50)
    print("Starting Inference")
    print("=" * 50 + "\n")

    predictions = inferencer.predict(test_loader, use_tta=True, tta_times=5)

    # 创建提交文件
    submission_df = inferencer.create_submission(predictions, config.SUBMISSION_PATH)

    print("\n" + "=" * 50)
    print("Inference completed!")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
