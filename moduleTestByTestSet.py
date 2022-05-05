import torch
from torch.utils.data import DataLoader

import cap_dataset
import my_module

img_data = cap_dataset.Captcha("./data/test/test", train=True)
testloader = DataLoader(img_data, batch_size=1,
                        shuffle=False, num_workers=4)
if __name__ == '__main__':
    captcha = my_module.CapNet()
    my_module.load_param(captcha, 'model_weight180accuracy=0.921875.pkl')
    captcha = captcha.cuda()
    for i, data in enumerate(testloader, 0):
        data, label = data
        # print(label)

        data = data.cuda()
        label = label.cuda()
        y1, y2, y3, y4, y5 = captcha(data)
        y1, y2, y3, y4, y5 = y1.topk(1, dim=1)[1].view(label.size(0), 1), y2.topk(1, dim=1)[1].view(label.size(0), 1), \
                             y3.topk(1, dim=1)[1].view(label.size(0), 1), y4.topk(1, dim=1)[1].view(label.size(0), 1), \
                             y5.topk(1, dim=1)[1].view(label.size(0), 1)

        y = torch.cat((y1, y2, y3, y4, y5), dim=1)
        # print(cap_dataset.LabelToStr(label.long()[0].numpy()))

        if y.equal(label):
            print("\033[32m标签值:%s   预测值:%s\033[0m" % (
                cap_dataset.LabelToStr(label.cpu().long()[0].numpy()),
                cap_dataset.LabelToStr(y.cpu().long()[0].numpy())))
        else:
            print("\033[31m标签值:%s   预测值:%s\033[0m" % (
                cap_dataset.LabelToStr(label.cpu().long()[0].numpy()),
                cap_dataset.LabelToStr(y.cpu().long()[0].numpy())))
