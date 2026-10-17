import os
import argparse
import glob
import numpy as np
import scipy  # 引入 scipy 以便进行补丁
from PIL import Image
from tqdm import tqdm
import torch

# ==============================================================================
# 1. OpenMP 冲突修复
# ==============================================================================
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ==============================================================================
# 2. Scipy 兼容性补丁 (关键修复: 解决 'scipy' has no attribute 'ndarray')
# ==============================================================================
# 问题: 新版 scipy 移除了对 numpy.ndarray 的直接引用，但旧版库(如 imquality/libsvm)仍在使用
if not hasattr(scipy, 'ndarray'):
    scipy.ndarray = np.ndarray

# ==============================================================================
# 3. Skimage 兼容性补丁 (针对新版 skimage 和 旧版 imquality)
# 必须在 import imquality 之前执行！
# ==============================================================================
import skimage.transform
import skimage.color

# --- 补丁 A: 修复 rescale 函数 ---
if hasattr(skimage.transform, 'rescale'):
    _original_rescale = skimage.transform.rescale


    def _patched_rescale(*args, **kwargs):
        # 1. 拦截并移除过时的 multichannel 参数
        if 'multichannel' in kwargs:
            is_multichannel = kwargs.pop('multichannel')

            # 2. 智能判断 channel_axis
            if 'channel_axis' not in kwargs:
                # 获取输入图像
                img_input = args[0] if len(args) > 0 else kwargs.get('image')

                # 如果输入图像是 2D (ndim==2)，说明已经是灰度图，强制 channel_axis=None
                if hasattr(img_input, 'ndim') and img_input.ndim == 2:
                    kwargs['channel_axis'] = None
                else:
                    kwargs['channel_axis'] = -1 if is_multichannel else None

        return _original_rescale(*args, **kwargs)


    skimage.transform.rescale = _patched_rescale

# --- 补丁 B: 修复 rgb2gray 函数 ---
if hasattr(skimage.color, 'rgb2gray'):
    _original_rgb2gray = skimage.color.rgb2gray


    def _patched_rgb2gray(image, *args, **kwargs):
        # 如果输入已经是 2D 数组（灰度），直接返回，不调用原函数避免报错
        if hasattr(image, 'ndim') and image.ndim == 2:
            return image
        return _original_rgb2gray(image, *args, **kwargs)


    skimage.color.rgb2gray = _patched_rgb2gray

# ==============================================================================
# 4. 导入业务库 (补丁生效后再导入)
# ==============================================================================
# 注意：imquality 可能会在导入时检查 scipy/skimage，所以必须放在补丁之后
import imquality.brisque as brisque
from loss.niqe_utils import *


def metrics(im_dir):
    avg_niqe = 0
    n = 0
    avg_brisque = 0

    # 获取文件列表
    files = sorted(glob.glob(im_dir))
    if len(files) == 0:
        print(f"[Warning] No files found in path: {im_dir}")
        return 0, 0

    print(f"Start processing {len(files)} images in {im_dir}...")

    for item in tqdm(files):
        try:
            # 读取图片并转为 RGB
            pil_img = Image.open(item).convert('RGB')
            im_array = np.array(pil_img)

            # 计算 BRISQUE
            # 注意: imquality 可能会抛出 index out of bounds 等其他错误，这里尽量捕获
            score_brisque = brisque.score(im_array)

            # 计算 NIQE
            score_niqe = calculate_niqe(im_array)

            avg_brisque += score_brisque
            avg_niqe += score_niqe
            n += 1

            # 清理显存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            # 捕获异常，打印文件名和错误信息
            # 有时候 brisque 对纯黑或纯白图片计算会失败，这是正常现象，跳过即可
            print(f"\n[Error] Failed to process {os.path.basename(item)}: {str(e)}")

    if n == 0:
        print("No images were successfully processed.")
        return 0, 0

    avg_brisque = avg_brisque / n
    avg_niqe = avg_niqe / n
    return avg_niqe, avg_brisque


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Quality Metrics Evaluation')

    # 数据集开关
    parser.add_argument('--DICM', action='store_true', help='Evaluate DICM dataset')
    parser.add_argument('--LIME', action='store_true', help='Evaluate LIME dataset')
    parser.add_argument('--MEF', action='store_true', help='Evaluate MEF dataset')
    parser.add_argument('--NPE', action='store_true', help='Evaluate NPE dataset')
    parser.add_argument('--VV', action='store_true', help='Evaluate VV dataset')

    # 自定义路径
    parser.add_argument('--input', type=str, default=None, help='Custom input path pattern (e.g. ./my_data/*.png)')

    args = parser.parse_args()

    # 确定图片路径
    im_dir = None
    if args.DICM:
        im_dir = './output/DICM/*.jpg'
    elif args.LIME:
        im_dir = './output/LIME/*.bmp'
    elif args.MEF:
        im_dir = './output/MEF/*.png'
    elif args.NPE:
        im_dir = './output/NPE/*.jpg'
    elif args.VV:
        im_dir = './output/VV/*.jpg'
    elif args.input:
        im_dir = args.input

    if im_dir:
        niqe_val, brisque_val = metrics(im_dir)
        print("-" * 30)
        print(f"Dataset: {im_dir}")
        print(f"Avg NIQE:    {niqe_val:.4f}")
        print(f"Avg BRISQUE: {brisque_val:.4f}")
        print("-" * 30)
    else:
        print("Please specify a dataset (e.g., --DICM) or use --input to specify a path.")