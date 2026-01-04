# CSIRO Image2Biomass Prediction

预测牧草生物量的深度学习解决方案

## 项目结构
```
biomass-prediction/
├── config.py           # 配置文件
├── dataset.py          # 数据加载和预处理
├── models.py           # 模型定义
├── train.py            # 训练脚本
├── inference.py        # 推理脚本
├── utils.py            # 工具函数
├── eda.py              # 数据探索
├── run_full_pipeline.py  # 完整流程
└── requirements.txt    # 依赖包
```

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 数据探索（可选但推荐）
```bash
python eda.py
```

### 3. 训练单个fold
```bash
# 修改config.py中的FOLD参数，然后运行
python train.py
```

### 4. 训练所有fold + 推理
```bash
python run_full_pipeline.py
```

这会：
- 训练所有5个fold
- 对每个fold保存最佳模型
- Ensemble所有模型进行推理
- 生成submission.csv

### 5. 仅推理（假设已训练）
```bash
python inference.py
```

## 模型架构

- **Backbone**: EfficientNetV2-L (ImageNet预训练)
- **Tabular Features**: NDVI, Height, State, Species, Date等
- **Fusion**: Image features + Tabular features
- **Multi-task**: 5个独立回归头

## 训练策略

- **K-Fold**: 5-fold交叉验证
- **Loss**: Weighted R² + Consistency Loss
- **Optimizer**: AdamW with Cosine Annealing
- **Mixed Precision**: AMP for faster training
- **Data Augmentation**: 翻转、旋转、颜色增强等

## 推理策略

- **Ensemble**: 所有5个fold的模型
- **TTA**: Test-Time Augmentation (5次)

## 配置说明

主要参数在`config.py`中：
```python
MODEL_NAME = 'tf_efficientnetv2_l'  # 可改为其他timm模型
IMAGE_SIZE = 384                     # 图片大小
BATCH_SIZE = 16                      # 批次大小
NUM_EPOCHS = 30                      # 训练轮数
LEARNING_RATE = 1e-4                 # 学习率
```

## 预期性能

- **Baseline** (单模型): top 40-50%
- **优化版** (特征工程): top 25-30%
- **Ensemble** (5-fold + TTA): top 15-20%

## 改进方向

1. **模型升级**: ConvNeXt, Swin Transformer, DINOv2
2. **更多特征**: NDVI时序、土壤数据等
3. **Domain Adaptation**: 处理跨州/季节的domain shift
4. **Pseudo Labeling**: 使用测试集进行半监督学习