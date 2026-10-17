import os
import argparse
import glob
import numpy as np
import scipy  # 引入 scipy 以便进行补丁
from PIL import Image
from tqdm import tqdm
import torch

# ==============================================================================
# 1. 现代 IQA 指标库导入 (MUSIQ, CLIPIQA)
# ==============================================================================
try:
    import pyiqa
except ImportError:
    print("\n[Error] 未检测到 pyiqa 库。请先执行: pip install pyiqa")
    exit(1)

# ==============================================================================
# 2. OpenMP 冲突修复
# ==============================================================================
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ==============================================================================
# 3. Scipy 兼容性补丁 (关键修复: 解决 'scipy' has no attribute 'ndarray')
# ==============================================================================
if not hasattr(scipy, 'ndarray'):
    scipy.ndarray = np.ndarray

# ==============================================================================
# 4. Skimage 兼容性补丁 (针对新版 skimage 和 旧版 imquality)
# 必须在 import imquality 之前执行！
# ==============================================================================
import skimage.transform
import skimage.color

# --- 补丁 A: 修复 rescale 函数 ---
if hasattr(skimage.transform, 'rescale'):
    _original_rescale = skimage.transform.rescale


    def _patched_rescale(*args, **kwargs):
        if 'multichannel' in kwargs:
            is_multichannel = kwargs.pop('multichannel')
            if 'channel_axis' not in kwargs:
                img_input = args[0] if len(args) > 0 else kwargs.get('image')
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
        if hasattr(image, 'ndim') and image.ndim == 2:
            return image
        return _original_rgb2gray(image, *args, **kwargs)


    skimage.color.rgb2gray = _patched_rgb2gray

# ==============================================================================
# 5. 导入业务库
# ==============================================================================
import imquality.brisque as brisque
from loss.niqe_utils import *


def metrics(im_dir):
    avg_niqe = 0
    avg_brisque = 0
    avg_musiq = 0
    avg_clipiqa = 0
    n = 0

    files = sorted(glob.glob(im_dir))
    if len(files) == 0:
        print(f"[Warning] No files found in path: {im_dir}")
        return 0, 0, 0, 0

    print(f"Start processing {len(files)} images in {im_dir}...")

    # 自动获取设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device} for MUSIQ and CLIPIQA")

    # 初始化现代 IQA 模型 (只会下载/加载一次)
    print("Loading MUSIQ model...")
    musiq_metric = pyiqa.create_metric('musiq', device=device)
    print("Loading CLIPIQA model...")
    clipiqa_metric = pyiqa.create_metric('clipiqa', device=device)

    for item in tqdm(files):
        try:
            # 1. 加载图像
            pil_img = Image.open(item).convert('RGB')
            im_array = np.array(pil_img)

            # 2. 计算传统指标 (NIQE, BRISQUE)
            score_brisque = brisque.score(im_array)
            score_niqe = calculate_niqe(im_array)

            # 3. 图像预处理 for 现代深度学习指标 (PyTorch Tensor [1, C, H, W] in range [0, 1])
            img_tensor = torch.from_numpy(im_array).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            img_tensor = img_tensor.to(device)

            # 4. 计算现代指标 (MUSIQ, CLIPIQA)
            with torch.no_grad():
                score_musiq = musiq_metric(img_tensor).item()
                score_clipiqa = clipiqa_metric(img_tensor).item()

            # 5. 累加得分
            avg_brisque += score_brisque
            avg_niqe += score_niqe
            avg_musiq += score_musiq
            avg_clipiqa += score_clipiqa
            n += 1

            # 清理显存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"\n[Error] Failed to process {os.path.basename(item)}: {str(e)}")

    if n == 0:
        print("No images were successfully processed.")
        return 0, 0, 0, 0

    avg_brisque = avg_brisque / n
    avg_niqe = avg_niqe / n
    avg_musiq = avg_musiq / n
    avg_clipiqa = avg_clipiqa / n

    return avg_niqe, avg_brisque, avg_musiq, avg_clipiqa


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Quality Metrics Evaluation (including MUSIQ & CLIPIQA)')

    parser.add_argument('--DICM', action='store_true', help='Evaluate DICM dataset')
    parser.add_argument('--LIME', action='store_true', help='Evaluate LIME dataset')
    parser.add_argument('--MEF', action='store_true', help='Evaluate MEF dataset')
    parser.add_argument('--NPE', action='store_true', help='Evaluate NPE dataset')
    parser.add_argument('--VV', action='store_true', help='Evaluate VV dataset')
    parser.add_argument('--input', type=str, default=None, help='Custom input path pattern (e.g. ./my_data/*.png)')

    args = parser.parse_args()

    im_dir = None
    if args.DICM:
        im_dir = './output/DICM/*.jpg'
    elif args.LIME:
        im_dir = './results_SCFR_WLMB/LIME/*.png'
    elif args.MEF:
        im_dir = './results_SCFR_WLMB/MEF/*.png'
    elif args.NPE:
        im_dir = './results_SCFR_WLMB/NPE/*.png'
    elif args.VV:
        im_dir = './results_SCFR_WLMB/VV/*.png'
    elif args.input:
        im_dir = args.input

    if im_dir:
        niqe_val, brisque_val, musiq_val, clipiqa_val = metrics(im_dir)
        print("=" * 40)
        print(f"Dataset: {im_dir}")
        print("-" * 40)
        print("Traditional Metrics (Lower is better ↓):")
        print(f"Avg NIQE:    {niqe_val:.4f}")
        print(f"Avg BRISQUE: {brisque_val:.4f}")
        print("-" * 40)
        print("Modern AI Metrics (Higher is better ↑):")
        print(f"Avg MUSIQ:   {musiq_val:.4f}")
        print(f"Avg CLIPIQA: {clipiqa_val:.4f}")
        print("=" * 40)
    else:
        print("Please specify a dataset (e.g., --DICM) or use --input to specify a path.")