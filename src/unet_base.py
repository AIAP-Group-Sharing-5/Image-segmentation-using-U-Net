import torch
import torch.nn as nn

# NOT IN USE
# This is the base model created from scratch

class contracting(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Dropout2d(0.25),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Dropout2d(0.25),
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(
                in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Dropout2d(0.25),
        )

        self.down_sample = nn.MaxPool2d(2, stride=2)

    def forward(self, X):
        X1 = self.layer1(X)
        X2 = self.layer2(self.down_sample(X1))
        X3 = self.layer3(self.down_sample(X2))
        X4 = self.layer4(self.down_sample(X3))
        X5 = self.layer5(self.down_sample(X4))
        return X5, X4, X3, X2, X1


class expansive(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        assert isinstance(num_classes, int) and num_classes > 0, "num_classes must be a positive integer"
        if num_classes == 2:
            self._num_classes = 1
        else:
            self._num_classes = num_classes
        self.layer1 = nn.Conv2d(64, self._num_classes, 3, stride=1, padding=1)

        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(512, 256, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(1024, 512, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.up_sample_54 = nn.ConvTranspose2d(1024, 512, 2, stride=2)

        self.up_sample_43 = nn.ConvTranspose2d(512, 256, 2, stride=2)

        self.up_sample_32 = nn.ConvTranspose2d(256, 128, 2, stride=2)

        self.up_sample_21 = nn.ConvTranspose2d(128, 64, 2, stride=2)  # 35 classes 

    def forward(self, X5, X4, X3, X2, X1):
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


class unet(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.down = contracting()
        self.up = expansive()

    def forward(self, X):
        X5, X4, X3, X2, X1 = self.down(X)
        X = self.up(X5, X4, X3, X2, X1)
        return X
