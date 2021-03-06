import numpy as np
# import scipy.misc
import os
import torch
import sys
sys.path.append('..')
import imageio
from PIL import Image

class CASIA_Face(object):
    def __init__(self, root):
        self.root = root

        # 从文本中读取训练数据的图片路径和标签
        img_txt_dir = os.path.join(root, 'CASIA-WebFace-112X96.txt')
        image_list = []
        label_list = []
        with open(img_txt_dir) as f:
            img_label_list = f.read().splitlines()
        for info in img_label_list:
            image_dir, label_name = info.split(' ')
            image_list.append(os.path.join(root, 'CASIA-WebFace-112X96', image_dir))
            label_list.append(int(label_name))

        self.image_list = image_list
        self.label_list = label_list
        self.class_nums = len(np.unique(self.label_list)) # 类别总数

    def __getitem__(self, index):
        img_path = self.image_list[index]
        target = self.label_list[index]
        img = imageio.imread(img_path) #  (H:112, W:96, C:3)

        if len(img.shape) == 2: # 如果灰度图，那么变为彩图
            img = np.stack([img] * 3, 2)
        flip = np.random.choice(2)*2-1
        img = img[:, ::flip, :] # 一半概率为左右对称图片
        img = (img - 127.5) / 128.0 # 归一化
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float() # (C:3, H:112, W:96)

        return img, target

    def __len__(self):
        return len(self.image_list)

if __name__ == '__main__': 
    dataset = CASIA_Face("../CASIA") # dataset.__getitem__(0)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=8, drop_last=False)

    sample = next(iter(trainloader))
    print(sample[0].shape, sample[1].shape)










# to_img = transforms.ToPILImage()
# img = to_img(sample[0][0])# C H W .numpy()
# img.show()

# img = np.transpose(sample[0][0].numpy(), (1, 2, 0))
# plt.imshow(img)
# plt.show()

# for data in trainloader:
#     print(data[0].shape)

# transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((112, 112)),
#     transforms.ToTensor(),
# ])
# img = transform(img)
