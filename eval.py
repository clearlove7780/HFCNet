import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
from tqdm import tqdm
from data.data import *
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

# 导入模型

try:
    from net.HVI_FGNet import HVI_FGNet
except ImportError:
    HVI_FGNet = None


def eval(model, testing_data_loader, model_path, output_folder, norm_size=True, LOL=False, v2=False, unpaired=False,
         alpha=1.0, gamma=1.0):
    torch.set_grad_enabled(False)

    # 加载权重
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        print(f'Pre-trained model loaded from {model_path}')
    else:
        print(f'Warning: Model path {model_path} does not exist. Using random weights or skipping load.')

    model.eval()
    print('Evaluation...')

    # HVI 参数调整 (如果模型支持)
    if hasattr(model, 'trans'):
        if hasattr(model.trans, 'gated'):
            if LOL:
                model.trans.gated = True
            elif v2 or unpaired:
                if hasattr(model.trans, 'gated2'):
                    model.trans.gated2 = True
                    model.trans.alpha = alpha

    for batch in tqdm(testing_data_loader):
        with torch.no_grad():
            if norm_size:
                input, name = batch[0], batch[1]
            else:
                input, name, h, w = batch[0], batch[1], batch[2], batch[3]

            input = input.cuda()

            # --- OOM 保护逻辑: 如果图片过大，进行下采样 ---
            _, _, origin_h, origin_w = input.shape
            # 设定最大边长，防止Attention显存爆炸 (通常设置为 960, 1024 或 1200)
            MAX_DIM = 1024
            scale = 1.0
            if max(origin_h, origin_w) > MAX_DIM:
                scale = MAX_DIM / max(origin_h, origin_w)
                new_h, new_w = int(origin_h * scale), int(origin_w * scale)
                # 确保尺寸是 16 的倍数 (对齐网络下采样)
                new_h = (new_h // 16) * 16
                new_w = (new_w // 16) * 16
                input = F.interpolate(input, size=(new_h, new_w), mode='bilinear', align_corners=False)
            # -------------------------------------------

            # 前向推理
            # 支持 gamma 矫正增强
            if gamma != 1.0:
                output = model(input ** gamma)
            else:
                output = model(input)

            if isinstance(output, dict):
                output = output['rgb']

        if not os.path.exists(output_folder):
            os.makedirs(output_folder, exist_ok=True)

        output = torch.clamp(output, 0, 1)

        # --- 恢复原始尺寸 ---
        if scale != 1.0:
            output = F.interpolate(output, size=(origin_h, origin_w), mode='bilinear', align_corners=False)
        # -------------------

        if not norm_size:
            output = output[:, :, :h, :w]

        output_img = transforms.ToPILImage()(output.squeeze(0).cpu())
        # 处理保存路径
        save_name = name[0]
        output_img.save(os.path.join(output_folder, save_name))

        # 清理显存
        torch.cuda.empty_cache()

    print('===> End evaluation')

    # 恢复状态
    if hasattr(model, 'trans'):
        if hasattr(model.trans, 'gated'):
            model.trans.gated = False
        if hasattr(model.trans, 'gated2'):
            model.trans.gated2 = False

    torch.set_grad_enabled(True)


if __name__ == '__main__':

    eval_parser = argparse.ArgumentParser(description='Eval')

    # 模型选择
    eval_parser.add_argument('--model_type', type=str, default='fgnet', choices=['cid', 'fgnet'],
                             help='Choose model architecture: cid or fgnet (default)')

    # 数据集选择开关
    eval_parser.add_argument('--lol', action='store_true', help='eval lolv1 dataset')
    eval_parser.add_argument('--lol_v2_real', action='store_true', help='eval lol_v2_real dataset')
    eval_parser.add_argument('--lol_v2_syn', action='store_true', help='eval lol_v2_syn dataset')
    eval_parser.add_argument('--loli_street', action='store_true', help='eval LoLI-Street dataset')

    # LSRW 开关
    eval_parser.add_argument('--lsrw_huawei', action='store_true', help='eval LSRW Huawei dataset')
    eval_parser.add_argument('--lsrw_nikon', action='store_true', help='eval LSRW Nikon dataset')

    eval_parser.add_argument('--SICE_grad', action='store_true', help='eval SICE_grad dataset')
    eval_parser.add_argument('--SICE_mix', action='store_true', help='eval SICE_mix dataset')
    eval_parser.add_argument('--fivek', action='store_true', help='eval FiveK dataset')

    # 无配对数据集开关 (参考 eval11.py 添加)
    eval_parser.add_argument('--unpaired', action='store_true', help='eval unpaired dataset')
    eval_parser.add_argument('--DICM', action='store_true', help='output DICM dataset')
    eval_parser.add_argument('--LIME', action='store_true', help='output LIME dataset')
    eval_parser.add_argument('--MEF', action='store_true', help='output MEF dataset')
    eval_parser.add_argument('--NPE', action='store_true', help='output NPE dataset')
    eval_parser.add_argument('--VV', action='store_true', help='output VV dataset')

    eval_parser.add_argument('--custome', action='store_true', help='eval custome dataset')

    # 路径配置
    eval_parser.add_argument('--custome_path', type=str, default='./YOLO', help='Path to custom dataset')
    eval_parser.add_argument('--weights', type=str, default=None, help='Path to model weights file')
    eval_parser.add_argument('--unpaired_weights', type=str, default='E:\\LLIE\\HVI-FourNet\\weights\\LOLv2_syn_colorbest\\train\\epoch_980.pth',
                             help='Weights for unpaired task')
    eval_parser.add_argument('--output', type=str, default='./output/', help='Output directory')

    # HVI 参数
    eval_parser.add_argument('--alpha', type=float, default=1.0, help='HVI alpha parameter')
    eval_parser.add_argument('--gamma', type=float, default=1.0, help='Gamma correction parameter')

    ep = eval_parser.parse_args()

    cuda = True
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, or need to change CUDA_VISIBLE_DEVICES number")

    if not os.path.exists(ep.output):
        os.makedirs(ep.output, exist_ok=True)

    # 默认设置
    norm_size = True
    num_workers = 1
    alpha = ep.alpha
    weight_path = ep.weights

    # ==========================
    # 1. 配置数据集路径和默认权重
    # ==========================
    if ep.lol:
        eval_data = DataLoader(dataset=get_eval_set("./datasets/LOLdataset/eval15/low"), num_workers=num_workers,
                               batch_size=1, shuffle=False)
        output_folder = os.path.join(ep.output, 'LOLv1/')
        if weight_path is None: weight_path = 'E:\\LLIE\\HVI-FourNet\\weights\\loss_ablation\\train_Wo_wave\\best.pth'

    elif ep.lol_v2_real:
        eval_data = DataLoader(dataset=get_eval_set("./datasets/LOLv2/Real_captured/Test/Low"), num_workers=num_workers,
                               batch_size=1, shuffle=False)
        output_folder = os.path.join(ep.output, 'LOLv2_real/')
        if weight_path is None: weight_path = 'E:\\LLIE\\HVI-FourNet\\weights\\12_22LOLv2real\\train\\epoch_860.pth'

    elif ep.lol_v2_syn:
        eval_data = DataLoader(dataset=get_eval_set("./datasets/LOLv2/Synthetic/Test/Low"), num_workers=num_workers,
                               batch_size=1, shuffle=False)
        output_folder = os.path.join(ep.output, 'LOLv2_syn/')
        if weight_path is None: weight_path = 'E:\\LLIE\\HVI-FourNet\\weights\\1_21LOLv2syn\\train\\epoch_980.pth'

    elif ep.loli_street:
        eval_data = DataLoader(dataset=get_eval_set("./datasets/LoLI-Street/Val/Low"), num_workers=num_workers,
                               batch_size=1, shuffle=False)
        output_folder = os.path.join(ep.output, 'LoLI-Street/')
        if weight_path is None: weight_path = 'weights/train_LOLv2real_12_11/epoch_1000.pth'

    # LSRW
    elif ep.lsrw_huawei:
        eval_data = DataLoader(dataset=get_eval_set("./datasets/LSRW/Eval/Huawei/low"), num_workers=num_workers,
                               batch_size=1, shuffle=False)
        output_folder = os.path.join(ep.output, 'LSRW_Huawei/')
        if weight_path is None: weight_path = 'weights/train_LOLv2real_12_11/epoch_1000.pth'

    elif ep.lsrw_nikon:
        eval_data = DataLoader(dataset=get_eval_set("./datasets/LSRW/Eval/Nikon/low"), num_workers=num_workers,
                               batch_size=1, shuffle=False)
        output_folder = os.path.join(ep.output, 'LSRW_Nikon/')
        if weight_path is None: weight_path = 'weights/train_LOLv2real_12_11/epoch_1000.pth'

    elif ep.SICE_grad:
        eval_data = DataLoader(dataset=get_SICE_eval_set("./datasets/SICE/SICE_Grad"), num_workers=num_workers,
                               batch_size=1, shuffle=False)
        output_folder = os.path.join(ep.output, 'SICE_grad/')
        norm_size = False

    elif ep.SICE_mix:
        eval_data = DataLoader(dataset=get_SICE_eval_set("./datasets/SICE/SICE_Mix"), num_workers=num_workers,
                               batch_size=1, shuffle=False)
        output_folder = os.path.join(ep.output, 'SICE_mix/')
        norm_size = False

    elif ep.fivek:
        eval_data = DataLoader(dataset=get_SICE_eval_set("./datasets/FiveK/test/input"), num_workers=num_workers,
                               batch_size=1, shuffle=False)
        output_folder = os.path.join(ep.output, 'fivek/')
        norm_size = False

    elif ep.custome:
        dataset_path = ep.custome_path
        eval_data = DataLoader(dataset=get_SICE_eval_set(dataset_path), num_workers=num_workers, batch_size=1,
                               shuffle=False)
        output_folder = os.path.join(ep.output, 'custome/')
        norm_size = False
        if weight_path is None: weight_path = 'weights/train_LOLv2real_12_11/epoch_1000.pth'

    # Unpaired datasets (新增逻辑)
    elif ep.unpaired:
        if ep.DICM:
            eval_data = DataLoader(dataset=get_SICE_eval_set("./datasets/DICM"), num_workers=num_workers, batch_size=1,
                                   shuffle=False)
            output_folder = os.path.join(ep.output, 'DICM/')
        elif ep.LIME:
            eval_data = DataLoader(dataset=get_SICE_eval_set("./datasets/LIME"), num_workers=num_workers, batch_size=1,
                                   shuffle=False)
            output_folder = os.path.join(ep.output, 'LIME/')
        elif ep.MEF:
            eval_data = DataLoader(dataset=get_SICE_eval_set("./datasets/MEF"), num_workers=num_workers, batch_size=1,
                                   shuffle=False)
            output_folder = os.path.join(ep.output, 'MEF/')
        elif ep.NPE:
            eval_data = DataLoader(dataset=get_SICE_eval_set("./datasets/NPE"), num_workers=num_workers, batch_size=1,
                                   shuffle=False)
            output_folder = os.path.join(ep.output, 'NPE/')
        elif ep.VV:
            eval_data = DataLoader(dataset=get_SICE_eval_set("./datasets/VV"), num_workers=num_workers, batch_size=1,
                                   shuffle=False)
            output_folder = os.path.join(ep.output, 'VV/')
        elif ep.custome:
            eval_data = DataLoader(dataset=get_SICE_eval_set(ep.custome_path), num_workers=num_workers, batch_size=1,
                                   shuffle=False)
            output_folder = os.path.join(ep.output, 'custome/')

        # Unpaired 设置
        alpha = ep.alpha
        norm_size = False
        if weight_path is None:
            weight_path = ep.unpaired_weights

    else:
        print("Error: No dataset specified.")
        exit(1)

    print(f"Output Folder: {output_folder}")
    print(f"Weights Path: {weight_path}")
    print(f"Model Type: {ep.model_type}")

    # ==========================
    # 2. 构建模型
    # ==========================
    if ep.model_type == 'fgnet':
        if HVI_FGNet is None: raise ImportError("HVI_FGNet model code missing.")
        eval_net = HVI_FGNet().cuda()



    else:
        raise ValueError(f"Unknown model type: {ep.model_type}")

    # ==========================
    # 3. 执行评估
    # ==========================
    eval(eval_net, eval_data, weight_path, output_folder,
         norm_size=norm_size,
         LOL=ep.lol,
         v2=(ep.lol_v2_real or ep.lol_v2_syn),
         unpaired=(ep.unpaired or ep.custome),
         alpha=alpha,
         gamma=ep.gamma)