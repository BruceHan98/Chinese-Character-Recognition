import torch.nn as nn


class Network(nn.Module):

    def __init__(self, max_classes):
        super(Network, self).__init__()

        self.feature = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 64 * 2, 3, stride=1, padding=1),
            nn.BatchNorm2d(64 * 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64 * 2, 64 * 4, 3, stride=1, padding=1),
            nn.BatchNorm2d(64 * 4),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64 * 4, 64 * 8, 3, stride=1, padding=1),
            nn.BatchNorm2d(64 * 8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64 * 8, 64 * 8, 3, stride=1, padding=1),
            nn.BatchNorm2d(64 * 8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 2 * 2, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, max_classes)
        )

    def forward(self, x):
        x = self.feature(x)
        x = x.view(-1, 512 * 2 * 2)
        x = self.classifier(x)
        return x
