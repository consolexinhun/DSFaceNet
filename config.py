import torch

BATCH_SIZE = 100
SAVE_FREQ = 1
TEST_FREQ = 5
TOTAL_EPOCH = 60

DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

RESUME = ''
SAVE_DIR = './save_model' # 保存模型的目录


CASIA_DATA_DIR = 'CASIA'
LFW_DATA_DIR = 'LFW'

GPU = 2

