import torch.nn as nn


class Dropout(nn.Module):

    def __init__(self, max_classes):
        super(Dropout, self).__init__()

        self.feature = nn.Sequential(  # input: 64 * 64 * 3
            nn.Conv2d(3, 64, 3, stride=1, padding=1),  # out: 64 * 64 * 64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32 * 32 * 64

            nn.Conv2d(64, 64 * 2, 3, stride=1, padding=1),  # 32 * 32 * 128
            nn.BatchNorm2d(64 * 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16 * 16 * 128

            nn.Conv2d(64 * 2, 64 * 4, 3, stride=1, padding=1),  # 16 * 16 * 256
            nn.BatchNorm2d(64 * 4),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 8 * 8 * 256

            nn.Conv2d(64 * 4, 64 * 8, 3, stride=1, padding=1),  # 8 * 8 * 512
            nn.BatchNorm2d(64 * 8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 4 * 4 * 512

            nn.Conv2d(64 * 8, 64 * 16, 3, stride=1, padding=1),  # 4 * 4 * 1024
            nn.BatchNorm2d(64 * 16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 2 * 2 * 1024
        )

        self.classifier = nn.Sequential(
            nn.Linear(1024 * 2 * 2, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, max_classes)
        )

    def forward(self, x):
        x = self.feature(x)
        x = x.view(-1, 1024 * 2 * 2)
        x = self.classifier(x)
        return x
