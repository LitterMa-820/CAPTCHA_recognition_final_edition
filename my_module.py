import os

import torch
from torch import nn
import torch.nn.functional as F


class CapNet(nn.Module):
    def __init__(self):
        super(CapNet, self).__init__()
        self.c1 = nn.Conv2d(3, 32, (3, 3), padding=1)
        self.c2 = nn.Conv2d(32, 32, (3, 3), padding=1)
        self.c4 = nn.Conv2d(32, 64, (3, 3), padding=1)
        self.c5 = nn.Conv2d(64, 64, (3, 3), padding=1)
        self.c7 = nn.Conv2d(64, 128, (3, 3), padding=1)
        self.c8 = nn.Conv2d(128, 128, (3, 3), padding=1)
        self.c10 = nn.Conv2d(128, 128, (3, 3), padding=1)
        self.c11 = nn.Conv2d(128, 128, (3, 3), padding=1)
        self.f13 = nn.Linear(1152, 512)
        # 这是四个用于输出四个字符的线性层
        self.fc41 = nn.Linear(512, 62)
        self.fc42 = nn.Linear(512, 62)
        self.fc43 = nn.Linear(512, 62)
        self.fc44 = nn.Linear(512, 62)
        self.fc45 = nn.Linear(512, 62)

    def forward(self, x):
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = F.max_pool2d(x, (2, 2))  # p3
        x = F.relu(self.c4(x))
        x = F.relu(self.c5(x))
        x = F.max_pool2d(x, (2, 2))  # p6
        x = F.relu(self.c7(x))
        x = F.relu(self.c8(x))
        x = F.max_pool2d(x, (2, 2))  # p9
        x = F.relu(self.c10(x))
        x = F.relu(self.c11(x))
        x = F.max_pool2d(x, (2, 2))  # p12
        x = x.view(-1, 1152)  # flatten
        x = self.f13(x)
        x1 = self.fc41(x)
        x2 = self.fc42(x)
        x3 = self.fc43(x)
        x4 = self.fc44(x)
        x5 = self.fc45(x)
        return x1, x2, x3, x4, x5


def save(model, path):
    torch.save(model.state_dict(), path)


def load_param(model, path):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
