import torch

BATCH_SIZE = 100
SAVE_FREQ = 2 # 保存模型的频率
TEST_FREQ = 1 # 验证的频率
TOTAL_EPOCH = 60

DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

RESUME = '' # 断点保存文件的文件路径
SAVE_DIR = './save_model' # 保存模型的目录
SAVE_FEATURE_FILENAME = "best_result.mat"

CASIA_DATA_DIR = 'CASIA' #
LFW_DATA_DIR = 'LFW' #

