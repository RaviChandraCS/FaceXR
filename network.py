import torch
from torch.nn import Sequential, Module
from torch.nn import Conv2d, MaxPool2d, BatchNorm2d, ReLU
from torch.nn import Flatten, Linear, Dropout2d, Dropout
from torch import cuda

class Net(Module):
    
    def __init__(self):
        super().__init__()
        
        self.conv1 = Conv2d(1, 8, 5, 1, 2)
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2d(2, 2)
        self.conv2 = Conv2d(8, 16, 5, 1, 2)
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool2d(2, 2)
        self.conv3 = Conv2d(16, 120, 5, 1, 2)
        self.relu3 = ReLU()
        self.drop1 = Dropout2d(0.5)
        self.flat = Flatten()
        self.fc1 = Linear(120 * 16 * 16, 128)
        self.drop2 = Dropout(0.5)
        self.fc2 = Linear(128, 7)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.drop1(x)
        x = self.flat(x)
        x = self.fc1(x)
        x = self.drop2(x)
        x = self.fc2(x)
        return x
