# cnn_model.py
import torch
import torch.nn as nn
import torchvision.models as models

class CNNFeatureExtractor(nn.Module):
    def __init__(self, output_dim=512):
        super(CNNFeatureExtractor, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-2])  # Remove avgpool and fc
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(resnet.fc.in_features, output_dim)

    def forward(self, x):
        # x: (batch_size * sequence_length, 3, H, W)
        x = self.features(x)                     # Feature maps
        x = self.pool(x)                         # (batch_size * seq_len, 512, 1, 1)
        x = x.view(x.size(0), -1)                # Flatten
        x = self.fc(x)                           # Project to output_dim
        return x