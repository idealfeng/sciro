"""
数据探索分析
运行这个脚本来了解数据分布
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image

from config import Config


def explore_data():
    config = Config()

    # 加载数据
    train_df = pd.read_csv(config.TRAIN_CSV)
    test_df = pd.read_csv(config.TEST_CSV)

    print("=" * 70)
    print("DATA EXPLORATION")
    print("=" * 70)

    # 基本信息
    print("\n1. BASIC INFO")
    print("-" * 70)
    print(f"Train samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Train images: {train_df['image_path'].nunique()}")
    print(f"Test images: {test_df['image_path'].nunique()}")

    # 检查是否有缺失值
    print("\n2. MISSING VALUES")
    print("-" * 70)
    print(train_df.isnull().sum())

    # 目标分布
    print("\n3. TARGET DISTRIBUTIONS")
    print("-" * 70)
    for target_name in config.TARGET_NAMES:
        target_data = train_df[train_df["target_name"] == target_name]["target"]
        print(f"\n{target_name}:")
        print(f"  Mean: {target_data.mean():.2f}")
        print(f"  Std:  {target_data.std():.2f}")
        print(f"  Min:  {target_data.min():.2f}")
        print(f"  Max:  {target_data.max():.2f}")
        print(f"  Median: {target_data.median():.2f}")

    # 特征分布
    print("\n4. FEATURE DISTRIBUTIONS")
    print("-" * 70)

    # 获取unique的图片级别数据
    img_df = train_df.drop_duplicates(subset=["image_path"])

    print(f"\nNDVI:")
    print(f"  Mean: {img_df['Pre_GSHH_NDVI'].mean():.3f}")
    print(f"  Std:  {img_df['Pre_GSHH_NDVI'].std():.3f}")
    print(f"  Min:  {img_df['Pre_GSHH_NDVI'].min():.3f}")
    print(f"  Max:  {img_df['Pre_GSHH_NDVI'].max():.3f}")

    print(f"\nHeight (cm):")
    print(f"  Mean: {img_df['Height_Ave_cm'].mean():.2f}")
    print(f"  Std:  {img_df['Height_Ave_cm'].std():.2f}")
    print(f"  Min:  {img_df['Height_Ave_cm'].min():.2f}")
    print(f"  Max:  {img_df['Height_Ave_cm'].max():.2f}")

    # 类别分布
    print(f"\nState distribution:")
    print(img_df["State"].value_counts())

    print(f"\nSpecies distribution:")
    print(img_df["Species"].value_counts())

    # 日期分布
    img_df["Sampling_Date"] = pd.to_datetime(img_df["Sampling_Date"])
    print(f"\nDate range:")
    print(f"  From: {img_df['Sampling_Date'].min()}")
    print(f"  To:   {img_df['Sampling_Date'].max()}")

    print(f"\nYear distribution:")
    print(img_df["Sampling_Date"].dt.year.value_counts().sort_index())

    print(f"\nMonth distribution:")
    print(img_df["Sampling_Date"].dt.month.value_counts().sort_index())

    # 检查物理约束
    print("\n5. PHYSICAL CONSISTENCY CHECK")
    print("-" * 70)

    # 获取每个图片的所有target
    consistency_check = []
    for img_path in train_df["image_path"].unique():
        img_data = train_df[train_df["image_path"] == img_path]

        dry_green = img_data[img_data["target_name"] == "Dry_Green_g"]["target"].values[
            0
        ]
        dry_dead = img_data[img_data["target_name"] == "Dry_Dead_g"]["target"].values[0]
        dry_clover = img_data[img_data["target_name"] == "Dry_Clover_g"][
            "target"
        ].values[0]
        dry_total = img_data[img_data["target_name"] == "Dry_Total_g"]["target"].values[
            0
        ]

        sum_components = dry_green + dry_dead + dry_clover
        diff = abs(dry_total - sum_components)

        consistency_check.append(diff)

    consistency_check = np.array(consistency_check)
    print(f"Dry_Total vs Sum(Green+Dead+Clover) difference:")
    print(f"  Mean abs diff: {consistency_check.mean():.2f}g")
    print(f"  Max abs diff:  {consistency_check.max():.2f}g")
    print(f"  % with diff < 1g: {(consistency_check < 1).mean() * 100:.1f}%")

    # 查看一些样本图片
    print("\n6. SAMPLE IMAGES")
    print("-" * 70)

    sample_images = img_df["image_path"].sample(3, random_state=42).values

    for img_path in sample_images:
        img_id = img_path.split("/")[-1].replace(".jpg", "")
        full_path = config.TRAIN_IMG_DIR / f"{img_id}.jpg"

        if full_path.exists():
            img = Image.open(full_path)
            print(f"\n{img_path}:")
            print(f"  Size: {img.size}")
            print(f"  Mode: {img.mode}")

            # 获取这张图片的metadata
            img_meta = img_df[img_df["image_path"] == img_path].iloc[0]
            print(f"  NDVI: {img_meta['Pre_GSHH_NDVI']:.3f}")
            print(f"  Height: {img_meta['Height_Ave_cm']:.1f}cm")
            print(f"  State: {img_meta['State']}")
            print(f"  Species: {img_meta['Species']}")
            print(f"  Date: {img_meta['Sampling_Date']}")

    print("\n" + "=" * 70)
    print("EXPLORATION COMPLETED")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    explore_data()
