import os
from pathlib import Path

import torch

_REPO_DIR = Path(__file__).resolve().parent
_KAGGLE_DATA_DIR = Path("/kaggle/input/csiro-image2biomass-prediction")
_KAGGLE_WORKING_DIR = Path("/kaggle/working")


def _looks_like_dataset_dir(path: Path) -> bool:
    return (path / "train.csv").exists() and (path / "test.csv").exists()


def _resolve_data_dir() -> Path:
    env_dir = os.environ.get("SCIRO_DATA_DIR")
    if env_dir:
        return Path(env_dir)

    candidates = [
        _KAGGLE_DATA_DIR,
        _REPO_DIR / "data_full" / "csiro-biomass",
        _REPO_DIR / "data" / "csiro-biomass",
        _REPO_DIR / "data_full",
        _REPO_DIR / "data",
    ]

    for candidate in candidates:
        if candidate.exists() and _looks_like_dataset_dir(candidate):
            return candidate

    found = next(_REPO_DIR.glob("**/train.csv"), None)
    if found is not None:
        return found.parent

    return _REPO_DIR / "data"


class Config:
    # paths
    DATA_DIR = _resolve_data_dir()
    TRAIN_IMG_DIR = DATA_DIR / "train"
    TEST_IMG_DIR = DATA_DIR / "test"
    TRAIN_CSV = DATA_DIR / "train.csv"
    TEST_CSV = DATA_DIR / "test.csv"

    # model
    MODEL_NAME = "tf_efficientnetv2_m"
    IMAGE_SIZE = 384
    PRETRAINED = True

    # training
    BATCH_SIZE = 12
    NUM_EPOCHS = 50
    LEARNING_RATE = 5e-5
    WEIGHT_DECAY = 1e-4
    NUM_WORKERS = 2

    # CV
    N_FOLDS = 5
    FOLD = 0

    # target weights
    TARGET_WEIGHTS = {
        "Dry_Green_g": 0.1,
        "Dry_Dead_g": 0.1,
        "Dry_Clover_g": 0.1,
        "GDM_g": 0.2,
        "Dry_Total_g": 0.5,
    }
    TARGET_NAMES = list(TARGET_WEIGHTS.keys())

    # device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # seed
    SEED = 42

    # early stopping
    PATIENCE = 10
    MIN_DELTA = 1e-5

    # augmentation
    AUG_PROB = 0.5

    # output
    OUTPUT_DIR = Path(
        os.environ.get(
            "SCIRO_OUTPUT_DIR",
            str(_KAGGLE_WORKING_DIR if _KAGGLE_WORKING_DIR.exists() else (_REPO_DIR / "output")),
        )
    )
    MODEL_SAVE_PATH = OUTPUT_DIR / f"model_fold{FOLD}.pth"
    SUBMISSION_PATH = OUTPUT_DIR / "submission.csv"
