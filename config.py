import os
import torch
from pathlib import Path

class Config:
    # 路径配置
    _REPO_DIR = Path(__file__).resolve().parent
    _KAGGLE_DATA_DIR = Path('/kaggle/input/csiro-image2biomass-prediction')
    _LOCAL_DATA_DIR = (_REPO_DIR / 'data_full') if (_REPO_DIR / 'data_full').exists() else (_REPO_DIR / 'data')

    DATA_DIR = Path(os.environ.get('SCIRO_DATA_DIR', str(_KAGGLE_DATA_DIR if _KAGGLE_DATA_DIR.exists() else _LOCAL_DATA_DIR)))
    TRAIN_IMG_DIR = DATA_DIR / 'train'
    TEST_IMG_DIR = DATA_DIR / 'test'
    TRAIN_CSV = DATA_DIR / 'train.csv'
    TEST_CSV = DATA_DIR / 'test.csv'
    
    # 模型配置
    MODEL_NAME = 'tf_efficientnetv2_m'  # timm模型名
    IMAGE_SIZE = 384  # EfficientNetV2推荐384
    PRETRAINED = True
    
    # 训练配置
    BATCH_SIZE = 12  # 根据GPU调整
    NUM_EPOCHS = 50
    LEARNING_RATE = 5e-5
    WEIGHT_DECAY = 1e-4
    NUM_WORKERS = 2
    
    # 交叉验证
    N_FOLDS = 5
    FOLD = 0  # 当前训练的fold
    
    # Target权重（官方定义）
    TARGET_WEIGHTS = {
        'Dry_Green_g': 0.1,
        'Dry_Dead_g': 0.1,
        'Dry_Clover_g': 0.1,
        'GDM_g': 0.2,
        'Dry_Total_g': 0.5,
    }
    
    TARGET_NAMES = list(TARGET_WEIGHTS.keys())
    
    # 设备配置
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 随机种子
    SEED = 42
    
    # Early stopping
    PATIENCE = 10
    MIN_DELTA = 1e-5
    
    # 数据增强配置
    AUG_PROB = 0.5
    
    # 输出
    _KAGGLE_WORKING_DIR = Path('/kaggle/working')
    _LOCAL_OUTPUT_DIR = _REPO_DIR / 'output'
    OUTPUT_DIR = Path(os.environ.get('SCIRO_OUTPUT_DIR', str(_KAGGLE_WORKING_DIR if _KAGGLE_WORKING_DIR.exists() else _LOCAL_OUTPUT_DIR)))
    MODEL_SAVE_PATH = OUTPUT_DIR / f'model_fold{FOLD}.pth'
    SUBMISSION_PATH = OUTPUT_DIR / 'submission.csv'
