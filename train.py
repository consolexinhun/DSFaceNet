import os
import torch.utils.data
from torch import nn
from config import *
import model
from dataload.CASIA_Face_loader import CASIA_Face
from dataload.LFW_loader import LFW
from torch.optim import lr_scheduler
import torch.optim as optim
from eval_flw import parseList, evaluation_10_fold
import numpy as np
import scipy.io

torch.manual_seed(1234)

# other init
start_epoch = 1

# define trainloader and testloader
trainset = CASIA_Face(root=CASIA_DATA_DIR)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=8, drop_last=False)

# nl: left_image_path
# nr: right_image_path
nl, nr, folds, flags = parseList(root=LFW_DATA_DIR)
testdataset = LFW(nl, nr)
testloader = torch.utils.data.DataLoader(testdataset, batch_size=32,
                                         shuffle=False, num_workers=8, drop_last=False)
# GPU_list = ''
# if isinstance(GPU, int): # 如果只有一个整型的值
#     GPU_list = str(GPU)
# else :
#     if isinstance(GPU, str): # 如果本身就是一个字符串
#         GPU_list = GPU
#     else: # 如果是一组整型的值
#         for i, gpu in enumerate(GPU):
#             GPU_list += str(gpu)
#             if i != len(GPU) -1:
#                 GPU_list += ','


device = DEVICE
# define model
net = model.VarGFaceNet(128)
arcmargin = model.ArcMarginProduct(128, trainset.class_nums)

# define optimizers
# ignored_params = list(map(id, net.linear1.parameters()))
# ignored_params += list(map(id, arcmargin.weight))
# prelu_params_id = []
# prelu_params = []
# for m in net.modules():
#     if isinstance(m, nn.PReLU):
#         ignored_params += list(map(id, m.parameters()))
#         prelu_params += m.parameters()
# base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())

# optimizer_ft = optim.SGD([
#     {'params': base_params, 'weight_decay': 4e-5},
#     {'params': net.linear1.parameters(), 'weight_decay': 4e-4},
#     {'params': arcmargin.weight, 'weight_decay': 4e-4},
#     {'params': prelu_params, 'weight_decay': 0.0}
# ], lr=0.1, momentum=0.9, nesterov=True)

optimizer_ft = torch.optim.Adam(net.parameters(),lr=1e-3)

# 学习率衰减
exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[36, 52, 58], gamma=0.1)


net = net.to(device)
arcmargin = arcmargin.to(device)
criterion = torch.nn.CrossEntropyLoss()

best_acc = 0.0
best_epoch = 0
for epoch in range(start_epoch, TOTAL_EPOCH+1):

    # train model
    print('Train Epoch: {}/{} ...'.format(epoch, TOTAL_EPOCH))
    net.train()

    train_total_loss = 0.0
    total = 0
    for data in trainloader:
        img, label = data[0].to(device), data[1].to(device)
        batch_size = img.size(0)
        optimizer_ft.zero_grad()

        raw_logits = net(img)

        output = arcmargin(raw_logits, label)
        total_loss = criterion(output, label)
        total_loss.backward()
        optimizer_ft.step()

        train_total_loss += total_loss.item() * batch_size
        total += batch_size

    train_total_loss = train_total_loss / total
    print('total_loss: {:.4f} '.format(train_total_loss))

    # test model on lfw
    if epoch % TEST_FREQ == 0:
        net.eval()
        featureLs = None
        featureRs = None
        print('Test Epoch: {} ...'.format(epoch))
        for data in testloader:
            for i in range(len(data)):
                data[i] = data[i].to(device)
            res = [net(d).data.cpu().numpy() for d in data]
            featureL = np.concatenate((res[0], res[1]), 1)
            featureR = np.concatenate((res[2], res[3]), 1)
            if featureLs is None:
                featureLs = featureL
            else:
                featureLs = np.concatenate((featureLs, featureL), 0)
            if featureRs is None:
                featureRs = featureR
            else:
                featureRs = np.concatenate((featureRs, featureR), 0)

        result = {'fl': featureLs, 'fr': featureRs, 'fold': folds, 'flag': flags}
        # save tmp_result
        scipy.io.savemat('tmp_result.mat', result)
        accs = evaluation_10_fold('tmp_result.mat')
        print('ave: {:.4f}'.format(np.mean(accs) * 100))

    # save model
    if epoch % SAVE_FREQ == 0:
        print('Saving checkpoint: {}'.format(epoch))
        net_state_dict = net.state_dict()
        if not os.path.exists(SAVE_DIR):
            os.mkdir(SAVE_DIR)
        torch.save({
            'epoch': epoch,
            'net_state_dict': net_state_dict},
            os.path.join(SAVE_DIR, '%03d.ckpt' % epoch))

    exp_lr_scheduler.step()
print('finishing training')
