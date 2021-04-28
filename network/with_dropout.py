import torch.nn as nn


class Dropout(nn.Module):

    def __init__(self, max_classes):
        super(Dropout, self).__init__()

        self.feature = nn.Sequential(  # input: 96 * 96 * 3
            nn.Conv2d(3, 64, 3, stride=1, padding=1),  # out: 96 * 96 * 64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 48 * 48 * 64

            nn.Conv2d(64, 64 * 2, 3, stride=1, padding=1),  # 48 * 48 * 128
            nn.BatchNorm2d(64 * 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 24 * 24 * 128

            nn.Conv2d(64 * 2, 64 * 4, 3, stride=1, padding=1),  # 24 * 24 * 256
            nn.BatchNorm2d(64 * 4),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 12 * 12 * 256

            nn.Conv2d(64 * 4, 64 * 8, 3, stride=1, padding=1),  # 12 * 12 * 512
            nn.BatchNorm2d(64 * 8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 6 * 6 * 512

            nn.Conv2d(64 * 8, 64 * 16, 3, stride=1, padding=1),  # 6 * 6 * 1024
            nn.BatchNorm2d(64 * 16),
            nn.ReLU(),
            nn.MaxPool2d(3, 3)  # 2 * 2 * 1024
        )

        self.classifier = nn.Sequential(
            nn.Linear(1024 * 2 * 2, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, max_classes)
        )

    def forward(self, x):
        x = self.feature(x)
        x = x.view(-1, 1024 * 2 * 2)
        x = self.classifier(x)
        return x
