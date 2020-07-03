import numpy as np
import scipy.io
import os
import torch.utils.data
from model import model_se_avg
from dataload.LFW_loader import LFW
from config import *
import argparse
from tqdm import tqdm
import time
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




# 从文本中拿到的人脸向量的路径，加载到数据集中，并通过网络得到人脸向量
def getFeatureFromTorch(lfw_dir, feature_save_dir, resume=None, gpu=True):
    net = model_se_avg.DSFaceNet()
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
    lfw_loader = torch.utils.data.DataLoader(lfw_dataset, batch_size=32,shuffle=False, num_workers=8, drop_last=False)

    featureLs = None
    featureRs = None

    is_calc_time = True
    ans = 0

    for data in tqdm(lfw_loader): # data (4, batch_size, C, H ,W)
        if gpu:
            for i in range(len(data)):
                data[i] = data[i].to(device)

        if is_calc_time:
            print("len_data:{}\t data.shape:{}".format(len(data), data[0].shape))
            start_time = time.time()
            res = [net(d).data.cpu().numpy() for d in data]
            end_time = time.time()
            is_calc_time = False
            # 算出平均每张图片的毫秒数量
            cnt = len(data) * len(data[0])  # 图片总数量
            ans = int(((end_time - start_time) * 1000 + 0.5) / cnt)
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


# score为左右人脸向量内积（归一化后的）， 如果向量内积大于阈值，则为同一个人，否则不是同一个人
def getAccuracy(scores, flags, threshold):
    p = np.sum(scores[flags == 1] > threshold) # 预测是同一个人并且对了
    n = np.sum(scores[flags == -1] < threshold) # 预测不是同一个人并且对了
    return 1.0 * (p + n) / len(scores)

'''
thrNum可以理解为精度，如果为10k
阈值为-1到1范围，中间有20k+1个数，遍历阈值，找到能使得精度最大的阈值
'''
def getThreshold(scores, flags, thrNum):
    accuracys = np.zeros((2 * thrNum + 1, 1))
    thresholds = np.arange(-thrNum, thrNum + 1) * 1.0 / thrNum
    for i in range(2 * thrNum + 1):
        accuracys[i] = getAccuracy(scores, flags, thresholds[i])

    max_index = np.squeeze(accuracys == np.max(accuracys)) # 找到能使精度最大的索引
    bestThreshold = np.mean(thresholds[max_index]) # 找到最好的阈值
    return bestThreshold

# 十折交叉验证
def evaluation_10_fold(root):
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

        # mu为当前验证样本的人脸向量的均值,在行上拼接，再对行求均值
        mu = np.mean(np.concatenate((featureLs[valFold[0], :], featureRs[valFold[0], :]), 0), 0)
        mu = np.expand_dims(mu, 0)
        featureLs = featureLs - mu
        featureRs = featureRs - mu
        featureLs = featureLs / np.expand_dims(np.sqrt(np.sum(np.power(featureLs, 2), 1)), 1)
        featureRs = featureRs / np.expand_dims(np.sqrt(np.sum(np.power(featureRs, 2), 1)), 1)
        # 上面四行，减去均值除以平方和开根号，很明显的归一化

        scores = np.sum(np.multiply(featureLs, featureRs), 1) # 左右脸向量内积
        threshold = getThreshold(scores[valFold[0]], flags[valFold[0]], 10000)# 在验证集上找最好的阈值
        ACCs[i] = getAccuracy(scores[testFold[0]], flags[testFold[0]], threshold) # 在测试集上得到精度
    return ACCs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--lfw_dir', type=str, default=LFW_DATA_DIR,
                        help='The path of lfw data')
    parser.add_argument('--resume', type=str, default='./model/best/068.ckpt',
                        help='The path pf save model')
    parser.add_argument('--feature_save_dir', type=str, default=SAVE_FEATURE_FILENAME,
                        help='The path of the extract features save, must be .mat file')
    args = parser.parse_args()

    avg_time = getFeatureFromTorch(args.lfw_dir, args.feature_save_dir, args.resume, gpu=False)
    print("平均每张照片的推理速度:{} ms".format(avg_time))
    ACCs = evaluation_10_fold(args.feature_save_dir)
    for i in range(len(ACCs)):
        print('{}    {:.2f}'.format(i+1, ACCs[i] * 100))
    print('--------')
    print('AVE    {:.2f}'.format(np.mean(ACCs) * 100))


