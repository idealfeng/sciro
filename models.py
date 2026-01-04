import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class BiomassPredictor(nn.Module):
    def __init__(
        self,
        model_name='tf_efficientnetv2_l',
        num_tabular_features=10,
        pretrained=True,
        dropout_rate=0.4
    ):
        super().__init__()
        
        # 图像特征提取器
        self.image_encoder = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # 移除分类头
            global_pool='avg'
        )
        
        # 获取图像特征维度
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            image_feat_dim = self.image_encoder(dummy_input).shape[1]
        
        print(f"Image feature dimension: {image_feat_dim}")
        
        # 表格特征处理网络
        self.tabular_encoder = nn.Sequential(
            nn.Linear(num_tabular_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
        
        # 融合层
        fusion_input_dim = image_feat_dim + 256
        self.fusion_layers = nn.Sequential(
            nn.Linear(fusion_input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )
        
        # 多任务回归头
        # 使用ModuleDict以便按名称访问
        self.regression_heads = nn.ModuleDict({
            'Dry_Green_g': self._make_head(512, dropout_rate),
            'Dry_Dead_g': self._make_head(512, dropout_rate),
            'Dry_Clover_g': self._make_head(512, dropout_rate),
            'GDM_g': self._make_head(512, dropout_rate),
            'Dry_Total_g': self._make_head(512, dropout_rate),
        })
    
    def _make_head(self, input_dim, dropout_rate):
        """创建单个回归头"""
        return nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(256, 1)
        )
    
    def forward(self, images, tabular_features):
        """
        Args:
            images: (B, 3, H, W)
            tabular_features: (B, num_features)
        
        Returns:
            dict of predictions: {target_name: (B,)}
        """
        # 图像特征
        image_features = self.image_encoder(images)  # (B, image_feat_dim)
        
        # 表格特征
        tabular_features = self.tabular_encoder(tabular_features)  # (B, 256)
        
        # 融合
        combined = torch.cat([image_features, tabular_features], dim=1)
        shared_features = self.fusion_layers(combined)  # (B, 512)
        
        # 多头预测
        predictions = {}
        for target_name, head in self.regression_heads.items():
            pred = head(shared_features).squeeze(-1)  # (B,)

            # Clover经常是0，用sigmoid压缩范围
            if target_name == 'Dry_Clover_g':
                pred = torch.sigmoid(pred) * 100.0  # 压缩到0-100范围
            else:
                # 确保非负（生物量不能为负）
                pred = F.softplus(pred)

            predictions[target_name] = pred
        
        return predictions