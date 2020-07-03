import numpy as np
import scipy.io
import os
import torch.utils.data
import model
from dataload.LFW_loader import LFW
from config import *
import argparse
from tqdm import tqdm
import time
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

device = DEVICE

# 根据路径，拿到左边的人脸和右边的人脸的路径，以及标签，folds是做交叉验证用的
def parseList(root):
    with open(os.path.join(root, 'pairs.txt')) as f:
        pairs = f.read().splitlines()[1:] # 'Abel_Pacheco\t1\t4'
    folder_name = 'lfw-112X96'
    nameLs = [] # 左边人脸路径
    nameRs = [] # 右边人脸路径
    folds = [] # 后面10折交叉验证的时候用的，6000数据分成了10份，每份600样本
    flags = [] # 标签 1相同 -1不同
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



def getAccuracy(scores, flags, threshold):
    p = np.sum(scores[flags == 1] > threshold)
    n = np.sum(scores[flags == -1] < threshold)
    return 1.0 * (p + n) / len(scores)


def getThreshold(scores, flags, thrNum):
    accuracys = np.zeros((2 * thrNum + 1, 1))
    thresholds = np.arange(-thrNum, thrNum + 1) * 1.0 / thrNum
    for i in range(2 * thrNum + 1):
        accuracys[i] = getAccuracy(scores, flags, thresholds[i])

    max_index = np.squeeze(accuracys == np.max(accuracys))
    bestThreshold = np.mean(thresholds[max_index])
    return bestThreshold


def evaluation_10_fold(root='./result/pytorch_result.mat'):
    ACCs = np.zeros(10)
    result = scipy.io.loadmat(root)
    for i in range(10):
        fold = result['fold']
        flags = result['flag']
        featureLs = result['fl']
        featureRs = result['fr']

        valFold = fold != i
        testFold = fold == i
        flags = np.squeeze(flags)

        mu = np.mean(np.concatenate((featureLs[valFold[0], :], featureRs[valFold[0], :]), 0), 0)
        mu = np.expand_dims(mu, 0)
        featureLs = featureLs - mu
        featureRs = featureRs - mu
        featureLs = featureLs / np.expand_dims(np.sqrt(np.sum(np.power(featureLs, 2), 1)), 1)
        featureRs = featureRs / np.expand_dims(np.sqrt(np.sum(np.power(featureRs, 2), 1)), 1)

        scores = np.sum(np.multiply(featureLs, featureRs), 1)
        threshold = getThreshold(scores[valFold[0]], flags[valFold[0]], 10000)
        ACCs[i] = getAccuracy(scores[testFold[0]], flags[testFold[0]], threshold)
    return ACCs



def getFeatureFromTorch(lfw_dir, feature_save_dir, resume=None, gpu=True):
    net = model.VarGFaceNet(128)
    if gpu:
        net = net.to(device)
    if resume:
        if gpu:
            ckpt = torch.load(resume)
        else:
            ckpt = torch.load(resume, map_location='cpu')
        net.load_state_dict(ckpt['net_state_dict'])

    net.eval()
    nl, nr, flods, flags = parseList(lfw_dir)
    lfw_dataset = LFW(nl, nr)
    lfw_loader = torch.utils.data.DataLoader(lfw_dataset, batch_size=32,
                                              shuffle=False, num_workers=8, drop_last=False)

    featureLs = None
    featureRs = None
    count = 0

    is_calc_time = True
    ans = 0

    for data in tqdm(lfw_loader):
        if gpu:
            for i in range(len(data)):
                data[i] = data[i].to(device)
        count += data[0].size(0) # batch的大小
        print('extracing deep features from the face pair {}...'.format(count))

        if is_calc_time:
            start_time = time.time()
            res = [net(d).data.cpu().numpy() for d in data]
            end_time = time.time()
            is_calc_time = False
            # 算出平均每张图片的毫秒数量
            count = len(data) * len(data[0])  # 图片总数量
            ans = int(((end_time - start_time) * 1000 + 0.5) / count)
        else:
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

    result = {'fl': featureLs, 'fr': featureRs, 'fold': flods, 'flag': flags}
    scipy.io.savemat(feature_save_dir, result)

    return  ans

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--lfw_dir', type=str, default=LFW_DATA_DIR, help='The path of lfw data')
    parser.add_argument('--resume', type=str, default='./model/best/068.ckpt',
                        help='The path pf save model')
    parser.add_argument('--feature_save_dir', type=str, default='./best_result.mat',
                        help='The path of the extract features save, must be .mat file')
    args = parser.parse_args()

    avg_time = getFeatureFromTorch(args.lfw_dir, args.feature_save_dir, args.resume, gpu=False)
    print("平均每张照片的推理速度:{} ms".format(avg_time))
    ACCs = evaluation_10_fold(args.feature_save_dir)
    for i in range(len(ACCs)):
        print('{}    {:.2f}'.format(i+1, ACCs[i] * 100))
    print('--------')
    print('AVE    {:.2f}'.format(np.mean(ACCs) * 100))


