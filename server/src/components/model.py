import torch
import torch.nn as nn


# Residual Block implementation (building block of ResNet architecture)
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)  # Batch normalization

        # Second convolutional layer
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection for residual learning
        self.shortcut = nn.Sequential()
        self.use_shortcut = stride != 1 or in_channels != out_channels
        if self.use_shortcut:
            self.shortcut = nn.Sequential(
                # 1x1 convolution for dimension matching
                nn.Conv2d(in_channels, out_channels, 1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels))

    def forward(self, x, fmap_dict=None, prefix=""):
        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)  # ReLU activation

        out = self.conv2(out)
        out = self.bn2(out)

        # Shortcut path
        shortcut = self.shortcut(x) if self.use_shortcut else x

        # Residual connection
        out_add = out + shortcut

        # Store feature maps if requested (for visualization)
        if fmap_dict is not None:
            fmap_dict[f"{prefix}.conv"] = out_add

        out = torch.relu(out_add)
        if fmap_dict is not None:
            fmap_dict[f"{prefix}.relu"] = out

        return out


# Main CNN model for audio classification
class AudioCNN(nn.Module):
    def __init__(self, num_classes=50):
        super().__init__()
        # Initial convolutional block
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False),  # 7x7 conv
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)  # 3x3 max pooling
        )

        # Residual blocks organized in layers
        # Layer 1: 3 blocks with 64 channels
        self.layer1 = nn.ModuleList([ResidualBlock(64, 64) for i in range(3)])

        # Layer 2: 4 blocks with 128 channels (first block downsamples)
        self.layer2 = nn.ModuleList(
            [ResidualBlock(64 if i == 0 else 128, 128, stride=2 if i == 0 else 1)
             for i in range(4)])

        # Layer 3: 6 blocks with 256 channels (first block downsamples)
        self.layer3 = nn.ModuleList(
            [ResidualBlock(128 if i == 0 else 256, 256, stride=2 if i == 0 else 1)
             for i in range(6)])

        # Layer 4: 3 blocks with 512 channels (first block downsamples)
        self.layer4 = nn.ModuleList(
            [ResidualBlock(256 if i == 0 else 512, 512, stride=2 if i == 0 else 1)
             for i in range(3)])

        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        self.dropout = nn.Dropout(0.5)  # Regularization
        self.fc = nn.Linear(512, num_classes)  # Final fully-connected layer

    def forward(self, x, return_feature_maps=False):
        if not return_feature_maps:
            # Standard forward pass (for inference)
            x = self.conv1(x)
            for block in self.layer1:
                x = block(x)
            for block in self.layer2:
                x = block(x)
            for block in self.layer3:
                x = block(x)
            for block in self.layer4:
                x = block(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)  # Flatten
            x = self.dropout(x)
            x = self.fc(x)
            return x
        else:
            # Forward pass with feature map collection (for visualization)
            feature_maps = {}

            # Initial block
            x = self.conv1(x)
            feature_maps["conv1"] = x  # Store first conv output

            # Layer 1 with feature map storage
            for i, block in enumerate(self.layer1):
                x = block(x, feature_maps, prefix=f"layer1.block{i}")
            feature_maps["layer1"] = x  # Store full layer output

            # Layer 2 with feature map storage
            for i, block in enumerate(self.layer2):
                x = block(x, feature_maps, prefix=f"layer2.block{i}")
            feature_maps["layer2"] = x

            # Layer 3 with feature map storage
            for i, block in enumerate(self.layer3):
                x = block(x, feature_maps, prefix=f"layer3.block{i}")
            feature_maps["layer3"] = x

            # Layer 4 with feature map storage
            for i, block in enumerate(self.layer4):
                x = block(x, feature_maps, prefix=f"layer4.block{i}")
            feature_maps["layer4"] = x

            # Classification head
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.dropout(x)
            x = self.fc(x)

            # Return both predictions and feature maps
            return x, feature_maps
