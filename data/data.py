from torchvision.transforms import Compose, ToTensor, RandomCrop, RandomHorizontalFlip, RandomVerticalFlip
from data.LOLdataset import *
from data.LoLIStreetDataset import *
# [新增] 导入 LSRW 数据集类
from data.LSRWDataset import *
from data.eval_sets import *
from data.SICE_blur_SID import *
from data.fivek import *

def transform1(size=256):
    # [修改] 如果 size 为 None 或 0，说明需要全分辨率，直接去掉 RandomCrop
    if size is None or size <= 0:
        return Compose([
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            ToTensor(),
        ])
    else:
        # 否则，保留裁剪
        return Compose([
            RandomCrop((size, size)),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            ToTensor(),
        ])

def transform2():
    return Compose([ToTensor()])


def get_lol_training_set(data_dir,size):
    return LOLDatasetFromFolder(data_dir, transform=transform1(size))


def get_lol_v2_training_set(data_dir,size):
    return LOLv2DatasetFromFolder(data_dir, transform=transform1(size))

def get_loli_street_training_set(data_dir, size):
    return LoLIStreetDatasetFromFolder(data_dir, transform=transform1(size))

# [新增] LSRW 训练集获取函数 (Huawei 和 Nikon 通用)
def get_lsrw_training_set(data_dir, size):
    return LSRWDatasetFromFolder(data_dir, transform=transform1(size))


def get_training_set_blur(data_dir,size):
    return LOLBlurDatasetFromFolder(data_dir, transform=transform1(size))


def get_lol_v2_syn_training_set(data_dir,size):
    return LOLv2SynDatasetFromFolder(data_dir, transform=transform1(size))


def get_SID_training_set(data_dir,size):
    return SIDDatasetFromFolder(data_dir, transform=transform1(size))


def get_SICE_training_set(data_dir,size):
    return SICEDatasetFromFolder(data_dir, transform=transform1(size))

def get_SICE_eval_set(data_dir):
    return SICEDatasetFromFolderEval(data_dir, transform=transform2())

def get_eval_set(data_dir):
    return DatasetFromFolderEval(data_dir, transform=transform2())

def get_fivek_training_set(data_dir,size):
    return FiveKDatasetFromFolder(data_dir, transform=transform1(size))

def get_fivek_eval_set(data_dir):
    return SICEDatasetFromFolderEval(data_dir, transform=transform2())