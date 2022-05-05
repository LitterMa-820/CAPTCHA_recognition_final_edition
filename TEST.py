import hiddenlayer
import torch
from torch.utils.data import DataLoader

import cap_dataset
import my_module

if __name__ == '__main__':
    # train_data = cap_dataset.Captcha("./data/train/train", train=True)
    # trainDataLoader = DataLoader(train_data, batch_size=1,
    #                              shuffle=True, num_workers=4)
    model = my_module.CapNet()
    my_module.load_param(model, 'model_weight119')
    # data_iter = iter(trainDataLoader)
    # hl_graph = hiddenlayer.build_graph(model, (next(data_iter)[0]))
    # hl_graph.save(path='CaptchaNet-b0.png', format='png')
    print( model)


