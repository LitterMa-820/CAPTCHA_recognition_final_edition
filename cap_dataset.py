import os

import torch
from PIL import Image
from torch.utils import data
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms as T

import my_module
from parameters import *

nums = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
lower_char = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
              'v', 'w', 'x', 'y', 'z']
upper_char = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
              'V', 'W', 'X', 'Y', 'Z']


def StrToLabel(Str):
    # print(Str)
    res_label = []
    for i in range(0, charNumber):
        if '0' <= Str[i] <= '9':  # 数字
            res_label.append(ord(Str[i]) - ord('0'))
        elif 'a' <= Str[i] <= 'z':  # 小写字母
            res_label.append(ord(Str[i]) - ord('a') + 10)
        else:  # 大写字母
            res_label.append(ord(Str[i]) - ord('A') + 36)
    return res_label


def LabelToStr(Label):
    Str = ""
    for i in Label:
        if i <= 9:
            Str += chr(ord('0') + i)
        elif i <= 35:
            Str += chr(ord('a') + i - 10)
        else:
            Str += chr(ord('A') + i - 36)
    return Str


class Captcha(data.Dataset):
    def __init__(self, root, train=True):
        self.imgPath = [os.path.join(root, img) for img in os.listdir(root)]
        self.transform = T.Compose([
            T.Resize((ImageHeight, ImageWidth)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __getitem__(self, index):
        img_path = self.imgPath[index]
        label = img_path.split('\\')[-1].split('.')[0]
        label_tensor = torch.Tensor(StrToLabel(label))
        data = Image.open(img_path)
        data = self.transform(data)  # 使用PLT打开图片文件
        return data, label_tensor

    def __len__(self):
        return len(self.imgPath)


# dataset class test
if __name__ == '__main__':
    img_data = Captcha("./data/train/train", train=True)
    trainDataLoader = DataLoader(img_data, batch_size=1,
                                 shuffle=False, num_workers=4)
    it = trainDataLoader.__iter__()
    data, label = it.next()
    print(data)
    print(label)
    # print(LabelToStr(label.long().numpy()))
    # capNet = my_module.CapNet()
    # my_module.load_param(capNet, 'model_weight.pkl')
    # y1, y2, y3, y4, y5 = capNet(data)
    # print(y1)
    # print(y2)
    # print(y3)
    # print(y4)
    # print(y5)
