import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from torchvision.models import vgg19
from net.HVI_transform import RGB_HVI
import math

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
    def forward(self, x, y):
        return torch.mean(torch.sqrt((x - y)**2 + self.eps**2))

class FrequencyLoss(nn.Module):
    def __init__(self):
        super(FrequencyLoss, self).__init__()
        self.criterion = nn.L1Loss()
    def forward(self, pred, gt):
        pred_fft = torch.fft.rfft2(pred + 1e-8, norm='backward')
        gt_fft = torch.fft.rfft2(gt + 1e-8, norm='backward')
        loss_amp = self.criterion(torch.log(torch.abs(pred_fft) + 1), torch.log(torch.abs(gt_fft) + 1))
        loss_pha = self.criterion(torch.angle(pred_fft), torch.angle(gt_fft))
        return loss_amp, loss_pha

class HVILoss(nn.Module):
    def __init__(self):
        super(HVILoss, self).__init__()
        self.trans = RGB_HVI()
        self.criterion = CharbonnierLoss()
    def forward(self, pred_rgb, gt_rgb):
        pred_hvi = self.trans.HVIT(pred_rgb)
        gt_hvi = self.trans.HVIT(gt_rgb)
        return self.criterion(pred_hvi[:, 2:, ...], gt_hvi[:, 2:, ...]), self.criterion(pred_hvi[:, :2, ...], gt_hvi[:, :2, ...])

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, channel=3):
        super().__init__()
        self.window_size, self.channel = window_size, channel
        self.window = self.create_window(window_size, channel)
    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - window_size//2)**2 / float(2 * sigma**2)) for x in range(window_size)])
        return gauss / gauss.sum()
    def create_window(self, window_size, channel):
        _1D = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D = _1D.mm(_1D.t()).float().unsqueeze(0).unsqueeze(0)
        return _2D.expand(channel, 1, window_size, window_size).contiguous()
    def _ssim(self, img1, img2, window, window_size, channel):
        mu1, mu2 = F.conv2d(img1, window, padding=window_size//2, groups=channel), F.conv2d(img2, window, padding=window_size//2, groups=channel)
        mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
        sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2
        return (((2*mu1_mu2 + 1e-4) * (2*sigma12 + 9e-4)) / ((mu1_sq + mu2_sq + 1e-4) * (sigma1_sq + sigma2_sq + 9e-4))).mean()
    def forward(self, img1, img2):
        self.window = self.window.to(img1.device).type_as(img1)
        return 1 - self._ssim(img1, img2, self.window, self.window_size, self.channel)

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg19(pretrained=True)
        self.loss_network = nn.Sequential(*list(vgg.features)[:35]).eval()
        for param in self.loss_network.parameters(): param.requires_grad = False
    def forward(self, pred, gt):
        return F.mse_loss(self.loss_network(pred), self.loss_network(gt))

class FGNetLoss(nn.Module):
    def __init__(self, weights=None):
        super(FGNetLoss, self).__init__()
        self.weights = weights if weights else {
            'rec': 1.5, 'ssim': 0.2, 'hvi_i': 1.0, 'hvi_hv': 0.8,
            'freq': 0.1, 'perc': 0.04
        }
        self.loss_char = CharbonnierLoss()
        self.loss_ssim = SSIMLoss()
        self.loss_hvi = HVILoss()
        self.loss_freq = FrequencyLoss()
        self.loss_perc = PerceptualLoss()

    def forward(self, pred, gt):
        l_rec = self.loss_char(pred, gt)
        l_ssim = self.loss_ssim(pred, gt)
        l_i, l_hv = self.loss_hvi(pred, gt)
        l_amp, l_pha = self.loss_freq(pred, gt)
        l_perc = self.loss_perc(pred, gt)

        total_loss = (
                self.weights['rec'] * l_rec +
                self.weights['ssim'] * l_ssim +
                self.weights['hvi_i'] * l_i +
                self.weights['hvi_hv'] * l_hv +
                self.weights['freq'] * (l_amp + l_pha) +
                self.weights['perc'] * l_perc
        )

        loss_dict = {
            'Total': total_loss.item(), 'Rec': l_rec.item(), 'SSIM': l_ssim.item(),
            'HVI_I': l_i.item(), 'HVI_HV': l_hv.item(), 'Freq': (l_amp+l_pha).item(), 'Perc': l_perc.item()
        }
        return total_loss, loss_dict