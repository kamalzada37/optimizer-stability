# model.py
import torch.nn as nn

class SmallCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,32,3,1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,1), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*5*5,128), nn.ReLU(),
            nn.Linear(128,num_classes)
        )
    def forward(self,x):
        return self.fc(self.conv(x))
