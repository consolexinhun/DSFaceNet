import os
import torch.utils.data
from config import *
from model import model_se_avg
from dataload.CASIA_Face_loader import CASIA_Face
from dataload.LFW_loader import LFW
from torch.optim import lr_scheduler
from eval_flw import parseList, evaluation_10_fold
import numpy as np
import pandas as pd
import scipy.io
from tqdm import tqdm
from torch import nn
from torch import optim

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-v","--vis", type=bool,default=False, help="是否要使用visdom工具")
args = parser.parse_args()

torch.manual_seed(1234)



# 训练集
trainset = CASIA_Face(root=CASIA_DATA_DIR)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,shuffle=True, num_workers=8, drop_last=False)

# 人脸路径
nl, nr, folds, flags = parseList(root=LFW_DATA_DIR)
# 验证集
testdataset = LFW(nl, nr)
testloader = torch.utils.data.DataLoader(testdataset, batch_size=32,shuffle=False, num_workers=8, drop_last=False)

print("dataset loaded!")


device = DEVICE
net = model_se_avg.DSFaceNet()
arcmargin = model_se_avg.ArcMarginProduct(128, trainset.class_nums)

ignored_params = list(map(id, net.linear1.parameters()))
ignored_params += list(map(id, arcmargin.weight))
prelu_params_id = []
prelu_params = []
for m in net.modules():
    if isinstance(m, nn.PReLU):
        ignored_params += list(map(id, m.parameters()))
        prelu_params += m.parameters()
base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())

optimizer_ft = optim.SGD([
    {'params': base_params, 'weight_decay': 4e-5},
    {'params': net.linear1.parameters(), 'weight_decay': 4e-4},
    {'params': arcmargin.weight, 'weight_decay': 4e-4},
    {'params': prelu_params, 'weight_decay': 0.0}
], lr=0.1, momentum=0.9, nesterov=True)

# optimizer_ft = torch.optim.Adam(net.parameters(),lr=1e-3)

# 学习率衰减
exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[36, 52, 58], gamma=0.1)


net = net.to(device)
arcmargin = arcmargin.to(device)
criterion = torch.nn.CrossEntropyLoss()

start_epoch = 1
if RESUME: # 从上次中断的地方继续训练
    ckpt = torch.load(RESUME)
    net.load_state_dict(ckpt['net_state_dict'])
    start_epoch = ckpt['epoch'] + 1

best_acc = 0.0
best_epoch = 0
if args.vis:
    from visdom import Visdom
    vis = Visdom()
    vis.line([0.], [0.], win="train_loss", opts=dict(title="train loss"))
    vis.line([0.], [0.], win="val_acc", opts=dict(title="test accuracy"))

# train_loss, train_epoch = [], []
# val_acc, val_epoch = [],[]
train_loss, val_acc = [],[]
for epoch in tqdm(range(start_epoch, TOTAL_EPOCH+1)):
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

    print('total_loss: {:.4f} '.format(train_total_loss)) # 训练平均损失
    if args.vis:
        vis.line([train_total_loss], [epoch], win="train_loss", update="append")
    train_loss.append({"epoch":epoch, "train_loss":train_total_loss})
    # train_loss.append(train_total_loss)
    # train_epoch.append(epoch)

    if epoch % TEST_FREQ == 0: # 测试
        net.eval()
        featureLs = None
        featureRs = None
        print('Test Epoch: {}/{} ...'.format(epoch, (TOTAL_EPOCH // TEST_FREQ)))
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
        scipy.io.savemat(SAVE_FEATURE_FILENAME, result)
        accs = evaluation_10_fold(SAVE_FEATURE_FILENAME)

        print('ave: {:.4f}'.format(np.mean(accs) * 100))
        if args.vis:
            vis.line([np.mean(accs)*100], [epoch], win="val_acc", update="append")
        val_acc.append({"epoch":epoch, "val_acc":np.mean(accs)*100})


        if np.mean(accs) > best_acc:
            best_epoch = epoch
            best_acc = np.mean(accs)

    if epoch % SAVE_FREQ == 0:
        print('Saving checkpoint: {}'.format(epoch))
        net_state_dict = net.state_dict()
        if not os.path.exists(SAVE_DIR):
            os.mkdir(SAVE_DIR)
        torch.save({'epoch': epoch,'net_state_dict': net_state_dict},
                   os.path.join(SAVE_DIR, '%03d.ckpt' % epoch))

    exp_lr_scheduler.step()

print("best epoch is {}, best average acc is {}".format(best_epoch, best_acc))
pd_train_loss = pd.DataFrame(train_loss)
pd_val_acc = pd.DataFrame(val_acc)
pd_train_loss.to_csv(os.path.join(SAVE_DIR, "train_loss.csv"), index=False)
pd_val_acc.to_csv(os.path.join(SAVE_DIR, "val_acc.csv"), index=False)
print('finishing training')










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
