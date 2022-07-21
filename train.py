import torch.cuda
from tensorboardX import SummaryWriter

import my_module
from parameters import batch_size


def train(module, criterion, optimizer, trainloader, testloader, epochs):
    writer = SummaryWriter()
    counter_train = 1
    counter_test = 1
    CUDA = torch.cuda.is_available()
    if CUDA:
        module = module.cuda()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, label = data
            label = label.long()
            if CUDA:
                inputs, label = inputs.cuda(), label.cuda()
            optimizer.zero_grad()
            y1, y2, y3, y4, y5 = module(inputs)
            label1, label2, label3, label4, label5 = label[:, 0], label[:, 1], label[:, 2], label[:, 3], label[:, 4]
            loss1, loss2, loss3, loss4, loss5 = criterion(y1, label1), criterion(y2, label2), criterion(y3,
                                                                                                        label3), criterion(
                y4, label4), criterion(y5, label5)
            loss = loss1 + loss2 + loss3 + loss4 + loss5
            loss.backward()
            optimizer.step()
            running_loss += loss.item()  # 取出张量的值
            # 每10个batch 打印一次
            if i % 10 == 9:
                print('[epoch:%d],Batch:%5d] Loss:%.3f' % (epoch + 1, i + 1, running_loss / 10))
                writer.add_scalar('loss/per 10 batches', running_loss / 10, counter_train)
                counter_train += 1
                running_loss = 0.0
    #         每10个epoch将模型进行一次保存并进行一轮测试
        if epoch % 10 == 9:
            print("save and test")
            current_accuracy = test(module, testloader=testloader)
            writer.add_scalar('accuracy/per 10 epoch', current_accuracy, counter_test)
            counter_test += 1
            module_name = 'model_weight' + str(epoch + 1)+'accuracy='+str(current_accuracy)+'.pkl'
            my_module.save(module, module_name)
    writer.close()
    print('Finished Training')


def test(module, testloader):
    total = 0
    rightNum = 0
    CUDA = torch.cuda.is_available()
    for i, data in enumerate(testloader, 0):
        images, labels = data
        if CUDA:
            images = images.cuda()
            labels = labels.cuda()
        y1, y2, y3, y4, y5 = module(images)
        y1, y2, y3, y4, y5 = y1.topk(1, dim=1)[1].view(labels.size(0), 1), y2.topk(1, dim=1)[1].view(labels.size(0), 1), \
                             y3.topk(1, dim=1)[1].view(labels.size(0), 1), y4.topk(1, dim=1)[1].view(labels.size(0), 1), \
                             y5.topk(1, dim=1)[1].view(labels.size(0), 1)
        total += labels.size(0)

        y = torch.cat((y1, y2, y3, y4, y5), dim=1)
        # print(y[0], " ", labels.size(0))
        diff = (y != labels)
        diff = diff.sum(1)
        diff = (diff != 0)
        res = diff.sum(0).item()
        rightNum += (labels.size(0) - res)
    print('Accuracy:%d%%' % (100 * rightNum / total))
    return rightNum / total
