import torch.nn as nn
import math
import torch

def conv3x3(in_planes, out_planes, ksize=3):
    return nn.Sequential(
            nn.Conv3d(in_planes, out_planes, ksize),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(inplace=True))

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.branch1 = nn.Sequential(
                conv3x3(4, 30),
                conv3x3(30, 30),
                conv3x3(30, 40),
                conv3x3(40, 40),
                conv3x3(40, 40),
                conv3x3(40, 40),
                conv3x3(40, 50),
                conv3x3(50, 50))

        self.branch2 = nn.Sequential(
                conv3x3(4, 30),
                conv3x3(30, 30),
                conv3x3(30, 40),
                conv3x3(40, 40),
                conv3x3(40, 40),
                conv3x3(40, 40),
                conv3x3(40, 50),
                conv3x3(50, 50))

        self.fc = nn.Sequential(
                conv3x3(100, 150, 1),
                conv3x3(150, 150, 1),
                nn.Conv3d(150, 5, 1))

    def forward(self, inputs):
        x1, x2 = inputs
        x1 = self.branch1(x1)
        x2 = self.branch2(x2)
        x2 = x2.repeat(1, 1, 3, 3, 3)
        x = torch.cat([x1, x2], 1)
        x = self.fc(x)
        return x
