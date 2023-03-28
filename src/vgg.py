import torch
import torch.nn as nn
import torchvision.models as models

class VGG16UNet(nn.Module):
    def __init__(self):
        super(VGG16UNet, self).__init__()
        self.features = models.vgg16(weights='VGG16_Weights.DEFAULT').features
        for param in self.features.parameters():
            param.requires_grad = False

        # Create decoder layers with concatenation
        self.layer1 = nn.Conv2d(64, 35, 3, stride=1, padding=1)

        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.5),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.5),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(512, 256, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.5),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(1024, 512, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.5),
        )

        self.up_sample_54 = nn.ConvTranspose2d(512, 512, 2, stride=2)

        self.up_sample_43 = nn.ConvTranspose2d(512, 256, 2, stride=2)

        self.up_sample_32 = nn.ConvTranspose2d(256, 128, 2, stride=2)

        self.up_sample_21 = nn.ConvTranspose2d(128, 64, 2, stride=2)  # 35 classes
        
    def forward(self, x):
        # Encoder path
        X1 = self.features[:4](x)
        # 64
        X2 = self.features[4:9](X1)
        # 128
        X3 = self.features[9:16](X2)
        # 256
        X4 = self.features[16:23](X3)
        # 512
        X5 = self.features[23:30](X4)
        
        # Decoder path
        X = self.up_sample_54(X5)
        X4 = torch.cat([X, X4], dim=1)
        X4 = self.layer5(X4)

        X = self.up_sample_43(X4)
        X3 = torch.cat([X, X3], dim=1)
        X3 = self.layer4(X3)

        X = self.up_sample_32(X3)
        X2 = torch.cat([X, X2], dim=1)
        X2 = self.layer3(X2)

        X = self.up_sample_21(X2)
        X1 = torch.cat([X, X1], dim=1)
        X1 = self.layer2(X1)

        X = self.layer1(X1)

        return X


class VGG19UNet(nn.Module):
    def __init__(self, num_classes=35):
        super(VGG19UNet, self).__init__()
        self.features = models.vgg19(weights="VGG19_Weights.DEFAULT").features
        assert num_classes > 0 and isinstance(num_classes, int)
        if num_classes == 2:
            self.num_classes = 1
        else:
            self.num_classes = num_classes
        for param in self.features.parameters():
            param.requires_grad = False

        # Create decoder layers with concatenation
        self.layer1 = nn.Conv2d(
            in_channels=64,
            out_channels=self.num_classes,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.25),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.25),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(512, 256, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.25),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(1024, 512, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.25),
        )

        self.up_sample_54 = nn.ConvTranspose2d(512, 512, 2, stride=2)

        self.up_sample_43 = nn.ConvTranspose2d(512, 256, 2, stride=2)

        self.up_sample_32 = nn.ConvTranspose2d(256, 128, 2, stride=2)

        self.up_sample_21 = nn.ConvTranspose2d(128, 64, 2, stride=2)  # 35 classes

    def forward(self, x):
        # Encoder path
        X1 = self.features[:4](x)
        # 64
        X2 = self.features[4:9](X1)
        # 128
        X3 = self.features[9:18](X2)
        # 256
        X4 = self.features[18:27](X3)
        # 512
        X5 = self.features[27:-1](X4)

        # Decoder path
        X = self.up_sample_54(X5)
        X4 = torch.cat([X, X4], dim=1)
        X4 = self.layer5(X4)

        X = self.up_sample_43(X4)
        X3 = torch.cat([X, X3], dim=1)
        X3 = self.layer4(X3)

        X = self.up_sample_32(X3)
        X2 = torch.cat([X, X2], dim=1)
        X2 = self.layer3(X2)

        X = self.up_sample_21(X2)
        X1 = torch.cat([X, X1], dim=1)
        X1 = self.layer2(X1)

        X = self.layer1(X1)

        return X
