import torch.cuda
from torch import nn, optim
from torch.utils.data import DataLoader

import cap_dataset
import my_module
from parameters import learningRate, batch_size, totalEpoch
from train import train

if __name__ == '__main__':
    train_data = cap_dataset.Captcha("./data/train/train", train=True)
    trainDataLoader = DataLoader(train_data, batch_size=batch_size,
                                 shuffle=True, num_workers=4)
    test_data = cap_dataset.Captcha("./data/test/test", train=True)
    testDataLoader = DataLoader(test_data, batch_size=batch_size,
                                shuffle=True, num_workers=4)
    CUDA = torch.cuda.is_available()
    if CUDA:
        captcha = my_module.CapNet().cuda()
    else:
        captcha = my_module.CapNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(captcha.parameters(), lr=learningRate)

    train(captcha, criterion, optimizer, trainDataLoader, testloader=testDataLoader, epochs=totalEpoch)
    my_module.save(captcha, 'model_weight_after4-12.pkl')
