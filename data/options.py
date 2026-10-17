import argparse


def _str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def option():
    parser = argparse.ArgumentParser(description='HVI-FGNet Final Training')
    parser.add_argument('--batchSize', type=int, default=8, help='training batch size')
    parser.add_argument('--cropSize', type=int, default=256, help='image crop size (patch size)')
    parser.add_argument('--nEpochs', type=int, default=1000, help='number of epochs to end')
    parser.add_argument('--start_epoch', type=int, default=0, help='number of epochs to start')
    parser.add_argument('--snapshots', type=int, default=10, help='Snapshots for save checkpoints')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning Rate')
    parser.add_argument('--gpu_mode', type=_str2bool, default=True)
    parser.add_argument('--shuffle', type=_str2bool, default=True)
    parser.add_argument('--threads', type=int, default=4, help='number of threads for dataloader')

    # scheduler
    parser.add_argument('--cos_restart_cyclic', type=_str2bool, default=False)
    parser.add_argument('--cos_restart', type=_str2bool, default=True)
    parser.add_argument('--warmup_epochs', type=int, default=3, help='warmup_epochs')
    parser.add_argument('--start_warmup', type=_str2bool, default=True)

    # --- Datasets ---
    parser.add_argument('--data_train_lol_v1', type=str, default='./datasets/LOLdataset/our485')
    parser.add_argument('--data_train_lolv2_real', type=str, default='./datasets/LOLv2/Real_captured/Train')
    parser.add_argument('--data_train_lolv2_syn', type=str, default='./datasets/LOLv2/Synthetic/Train')

    parser.add_argument('--data_val_lol_v1', type=str, default='./datasets/LOLdataset/eval15/low')
    parser.add_argument('--data_val_lolv2_real', type=str, default='./datasets/LOLv2/Real_captured/Test/Low')
    parser.add_argument('--data_val_lolv2_syn', type=str, default='./datasets/LOLv2/Synthetic/Test/Low')

    parser.add_argument('--data_valgt_lol_v1', type=str, default='./datasets/LOLdataset/eval15/high/')
    parser.add_argument('--data_valgt_lolv2_real', type=str, default='./datasets/LOLv2/Real_captured/Test/Normal/')
    parser.add_argument('--data_valgt_lolv2_syn', type=str, default='./datasets/LOLv2/Synthetic/Test/Normal/')

    parser.add_argument('--val_folder', default='./results/', help='Location to save validation datasets')

    # --- Final Loss Weights (Removed Wavelet Loss) ---
    parser.add_argument('--rec_weight', type=float, default=1, help='Pixel reconstruction loss (Charbonnier)')
    parser.add_argument('--ssim_weight', type=float, default=0.2, help='SSIM loss')
    parser.add_argument('--hvi_i_weight', type=float, default=1.0, help='HVI I-channel loss')
    parser.add_argument('--hvi_hv_weight', type=float, default=1.0, help='HVI HV-channel loss')
    parser.add_argument('--freq_weight', type=float, default=0.1, help='Frequency domain loss')
    parser.add_argument('--perc_weight', type=float, default=0.04, help='Perceptual loss')

    # Training tricks
    parser.add_argument('--gamma', type=_str2bool, default=False)
    parser.add_argument('--start_gamma', type=int, default=60)
    parser.add_argument('--end_gamma', type=int, default=120)
    parser.add_argument('--grad_detect', type=_str2bool, default=False)
    parser.add_argument('--grad_clip', type=_str2bool, default=True)

    parser.add_argument('--dataset', type=str, default='lolv2_syn', choices=['lol_v1', 'lolv2_real', 'lolv2_syn'],
                        help='Select dataset')
    parser.add_argument('--exp_name', type=str, default='Trans_conv_syn', help='Name of the experiment')

    return parser