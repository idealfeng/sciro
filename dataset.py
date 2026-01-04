import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
import warnings

warnings.filterwarnings("ignore")


class BiomassDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, is_test=False):
        """
        Args:
            df: DataFrame with columns [sample_id, image_path, target_name, target(optional)]
            img_dir: 图片目录
            transform: albumentations transform
            is_test: 是否为测试集
        """
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test

        # 按图片ID分组（每个图片有5个target）
        self.image_ids = df["image_path"].unique()

        # 预计算每个图片的元数据
        self._prepare_metadata()

    def _prepare_metadata(self):
        """预处理表格特征"""
        # 获取unique的图片级别数据
        img_df = self.df.drop_duplicates(subset=['image_path']).copy()

        # 提取图片ID
        img_df["image_id"] = img_df["image_path"].apply(
            lambda x: x.split("/")[-1].replace(".jpg", "")
        )

        # 日期特征
        img_df["Sampling_Date"] = pd.to_datetime(img_df["Sampling_Date"])
        img_df["year"] = img_df["Sampling_Date"].dt.year
        img_df["month"] = img_df["Sampling_Date"].dt.month
        img_df["day_of_year"] = img_df["Sampling_Date"].dt.dayofyear
        img_df["season"] = img_df["month"].apply(self._get_season)

        # 类别编码
        self.state_encoder = LabelEncoder()
        self.species_encoder = LabelEncoder()

        img_df['State_encoded'] = self.state_encoder.fit_transform(img_df['State'])

        # 物种分组（合并稀有类别）
        species_mapping = {
            'Ryegrass_Clover': 'Ryegrass_Clover',
            'Ryegrass': 'Ryegrass',
            'Phalaris_Clover': 'Phalaris_Clover',
            'Clover': 'Clover',
            'Fescue': 'Fescue',
            'Lucerne': 'Lucerne',
            # 其他稀有物种都归为 'Other'
        }

        img_df['Species_grouped'] = img_df['Species'].apply(lambda x: species_mapping.get(x, 'Other'))
        img_df['Species_encoded'] = self.species_encoder.fit_transform(img_df['Species_grouped'])

        # NDVI衍生特征
        img_df["ndvi_height"] = img_df["Pre_GSHH_NDVI"] * img_df["Height_Ave_cm"]
        img_df["ndvi_squared"] = img_df["Pre_GSHH_NDVI"] ** 2
        img_df["height_squared"] = img_df["Height_Ave_cm"] ** 2

        # 存储为字典便于查找
        self.metadata = img_df.set_index("image_id").to_dict("index")

    @staticmethod
    def _get_season(month):
        """澳大利亚季节（南半球）"""
        if month in [12, 1, 2]:
            return 0  # Summer
        elif month in [3, 4, 5]:
            return 1  # Autumn
        elif month in [6, 7, 8]:
            return 2  # Winter
        else:
            return 3  # Spring

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_path = self.image_ids[idx]
        image_id = image_path.split("/")[-1].replace(".jpg", "")

        # 加载图片
        img_path = self.img_dir / f"{image_id}.jpg"
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)

        # 数据增强
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # 获取表格特征
        meta = self.metadata[image_id]
        tabular_features = np.array(
            [
                meta["Pre_GSHH_NDVI"],
                meta["Height_Ave_cm"],
                meta["State_encoded"],
                meta["Species_encoded"],
                meta["month"],
                meta["day_of_year"],
                meta["season"],
                meta["ndvi_height"],
                meta["ndvi_squared"],
                meta["height_squared"],
            ],
            dtype=np.float32,
        )

        if self.is_test:
            return {
                "image": image,
                "tabular": torch.tensor(tabular_features, dtype=torch.float32),
                "image_id": image_id,
            }

        # 获取所有5个target
        img_data = self.df[self.df["image_path"] == image_path]
        targets = {row["target_name"]: row["target"] for _, row in img_data.iterrows()}

        # 转换为tensor
        targets = {k: torch.tensor(v, dtype=torch.float32) for k, v in targets.items()}

        return {
            "image": image,
            "tabular": torch.tensor(tabular_features, dtype=torch.float32),
            "targets": targets,
            "image_id": image_id,
        }


def get_transforms(is_train=True, image_size=384):
    """获取数据增强pipeline"""
    if is_train:
        return A.Compose([
            # 先处理长宽比问题（小图先pad再裁剪，避免报错）
            A.PadIfNeeded(min_height=1000, min_width=1000, border_mode=4, p=1.0),
            A.OneOf([
                A.RandomCrop(height=1000, width=1000, p=0.3),
                A.CenterCrop(height=1000, width=1000, p=0.7),
            ], p=1.0),

            A.Resize(image_size, image_size),

            # 更激进的几何变换
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.2, rotate_limit=30, p=0.6),

            # 更强的颜色增强（应对不同光照/季节）
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
                A.HueSaturationValue(hue_shift_limit=25, sat_shift_limit=40, val_shift_limit=25, p=1.0),
                A.ColorJitter(p=1.0),
            ], p=0.7),

            # 噪声和模糊
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 70.0), p=1.0),
                A.GaussianBlur(blur_limit=(3, 9), p=1.0),
                A.MotionBlur(blur_limit=7, p=1.0),
            ], p=0.4),

            # Cutout（防止过拟合）
            A.CoarseDropout(max_holes=12, max_height=48, max_width=48, p=0.5),

            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.PadIfNeeded(min_height=1000, min_width=1000, border_mode=4, p=1.0),
            A.CenterCrop(height=1000, width=1000),
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])


def prepare_folds(train_csv_path, n_folds=5, seed=42):
    """准备K-fold交叉验证"""
    df = pd.read_csv(train_csv_path)

    # 获取unique的图片ID
    image_ids = df["image_path"].unique()

    # K-fold split
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    # 为每个图片分配fold
    folds = {}
    for fold, (train_idx, val_idx) in enumerate(kfold.split(image_ids)):
        train_images = image_ids[train_idx]
        val_images = image_ids[val_idx]

        folds[fold] = {
            "train": df[df["image_path"].isin(train_images)],
            "val": df[df["image_path"].isin(val_images)],
        }

    return folds


def get_dataloaders(config):
    """获取训练和验证的DataLoader"""
    # 准备fold数据
    folds = prepare_folds(config.TRAIN_CSV, config.N_FOLDS, config.SEED)
    fold_data = folds[config.FOLD]

    # 创建Dataset
    train_dataset = BiomassDataset(
        fold_data["train"],
        config.TRAIN_IMG_DIR,
        transform=get_transforms(is_train=True, image_size=config.IMAGE_SIZE),
        is_test=False,
    )

    val_dataset = BiomassDataset(
        fold_data["val"],
        config.TRAIN_IMG_DIR,
        transform=get_transforms(is_train=False, image_size=config.IMAGE_SIZE),
        is_test=False,
    )

    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader
