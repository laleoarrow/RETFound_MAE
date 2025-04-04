import torch
import torch.nn as nn
import torchvision
from torchvision.models.resnet import Bottleneck

class ResNet50(torchvision.models.ResNet):
    """直接继承官方ResNet，保持参数名一致性"""
    def __init__(self):
        # 调用官方ResNet50初始化（block, layers等参数固定）
        super().__init__(
            block=Bottleneck,
            layers=[3, 4, 6, 3],  # ResNet50标准配置
            num_classes=1000       # 原始分类头尺寸（后续会覆盖）
        )

        # 替换全连接层
        self.fc = nn.Sequential(
            nn.Linear(2048, 496),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(496, 2)
        )

        # 初始化新添加的层
        self._init_fc_layers()

    def _init_fc_layers(self):
        """自定义全连接层初始化"""
        # 中间层 (2048→496)
        nn.init.kaiming_normal_(self.fc[0].weight, 
                               mode='fan_out', 
                               nonlinearity='relu')
        nn.init.constant_(self.fc[0].bias, 0)
        
        # 输出层 (496→num_classes)
        nn.init.normal_(self.fc[3].weight, std=0.01)
        nn.init.constant_(self.fc[3].bias, 0)

    def forward_features(self, x):
        """特征提取方法（不包含全连接层）"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, x):
        """完整分类流程"""
        features = self.forward_features(x)
        return self.fc(features)