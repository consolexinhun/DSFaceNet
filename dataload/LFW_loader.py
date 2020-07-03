import numpy as np
import sys
sys.path.append("..")
import torch
import imageio
import os
class LFW(object):
    def __init__(self, imgl, imgr):
        self.imgl_list = imgl
        self.imgr_list = imgr

    def __getitem__(self, index):
        imgl = imageio.imread(self.imgl_list[index])

        if len(imgl.shape) == 2:
            imgl = np.stack([imgl] * 3, 2)
        imgr = imageio.imread(self.imgr_list[index])

        if len(imgr.shape) == 2:
            imgr = np.stack([imgr] * 3, 2)

        imglist = [imgl, imgl[:, ::-1, :], imgr, imgr[:, ::-1, :]]
        for i in range(len(imglist)):
            imglist[i] = (imglist[i] - 127.5) / 128.0
            imglist[i] = imglist[i].transpose(2, 0, 1)
        imgs = [torch.from_numpy(i).float() for i in imglist] # 成了一个 (4, 3, 112, 96)
        # imgs:左脸张量，对称左脸张量，右脸张量，对称右脸张量
        return imgs

    def __len__(self):
        return len(self.imgl_list)

def parseList(root):
    with open(os.path.join(root, 'pairs.txt')) as f:
        pairs = f.read().splitlines()[1:] # 'Abel_Pacheco\t1\t4'
    folder_name = 'lfw-112X96'
    nameLs = [] # 左边人脸路径
    nameRs = [] # 右边人脸路径
    folds = [] # 后面10折交叉验证的时候用的，6000数据分成了10份，每份600样本
    flags = [] # 标签： 1相同 -1不同 ；一维的向量
    for i, p in enumerate(pairs):
        p = p.split('\t')
        if len(p) == 3:# 同一个人
            nameL = os.path.join(root, folder_name, p[0], p[0] + '_' + '{:04}.jpg'.format(int(p[1])))
            nameR = os.path.join(root, folder_name, p[0], p[0] + '_' + '{:04}.jpg'.format(int(p[2])))
            fold = i // 600
            flag = 1
        elif len(p) == 4: # 不同的人
            nameL = os.path.join(root, folder_name, p[0], p[0] + '_' + '{:04}.jpg'.format(int(p[1])))
            nameR = os.path.join(root, folder_name, p[2], p[2] + '_' + '{:04}.jpg'.format(int(p[3])))
            fold = i // 600
            flag = -1
        nameLs.append(nameL)
        nameRs.append(nameR)
        folds.append(fold)
        flags.append(flag)
    return [nameLs, nameRs, folds, flags]

if __name__ == '__main__':
    nl, nr, flods, flags = parseList("../LFW")
    lfw_dataset = LFW(nl, nr)








# lfw_loader = torch.utils.data.DataLoader(lfw_dataset, batch_size=32,shuffle=False, num_workers=8, drop_last=False)
# sample = next(iter(lfw_loader)) # (4, 32, 3, 112, 96)
# print(sample[0].shape, sample[1].shape)
# lfw_dataset.__getitem__(0)
# for i in range(len(imgs)):
#     transform = transforms.Compose([
#         transforms.ToPILImage(),
#         transforms.Resize((112, 112)),
#         transforms.ToTensor(),
#     ])
#     imgs[i] = transform(imgs[i])
# print(len(imgs),imgs[0].shape)