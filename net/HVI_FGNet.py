import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from .HVI_transform import RGB_HVI


# ==============================================================================
# 1. 基础组件 (Basic Blocks)
# ==============================================================================

def dwt_init(x):
    """
    离散小波变换 (Haar Wavelet)
    """
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4
    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)


class DWTDownSample(nn.Module):
    """
    DWT 下采样模块
    """

    def __init__(self, in_channels, out_channels):
        super(DWTDownSample, self).__init__()
        self.conv = nn.Conv2d(in_channels * 4, out_channels, 1, 1, 0, bias=False)

    def forward(self, x):
        dwt_out = dwt_init(x)
        c = x.shape[1]
        # x_High 包含 HL, LH, HH，通道数是输入通道数的 3 倍
        x_High = dwt_out[:, c:, :, :]
        out = self.conv(dwt_out)
        return out, x_High


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# ==============================================================================
# [改进点 1] 交互融合模块 (Cross-Stream Interaction)
# ==============================================================================

class InteractiveFusion(nn.Module):


    def __init__(self, channels):
        super(InteractiveFusion, self).__init__()
        # 简单的通道注意力交互
        self.conv_i_to_hv = nn.Conv2d(channels, channels, 1)
        self.conv_hv_to_i = nn.Conv2d(channels, channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, f_i, f_hv):
        # I 流利用 HV 的信息来加权: 如果该区域色彩丰富/纹理清晰，可能需要保留 I 的结构
        att_from_hv = self.sigmoid(self.conv_hv_to_i(f_hv))
        f_i_out = f_i * att_from_hv + f_i

        # HV 流利用 I 的信息来加权: 如果该区域很亮，色彩信噪比高，增强 HV
        att_from_i = self.sigmoid(self.conv_i_to_hv(f_i))
        f_hv_out = f_hv * att_from_i + f_hv

        return f_i_out, f_hv_out


# ==============================================================================
# [改进点 3] 可学习高频门控 (Learnable High-Frequency Gating)
# ==============================================================================

class HighFreqGating(nn.Module):
    """
    用于 I-Stream 的解码阶段。
    不再直接丢弃所有高频信息 (zeros)，而是根据当前的解码特征 f_dec，
    预测一个 Mask，决定从 Encoder 跳跃过来的高频信息 (hf_skip) 哪些该保留，哪些该抑制。
    """

    def __init__(self, in_channels, skip_channels):
        super(HighFreqGating, self).__init__()
        # in_channels: 解码器当前特征通道数
        # skip_channels: 小波高频特征通道数 (通常是下一级 out_channels * 3)
        self.gate_net = nn.Sequential(
            nn.Conv2d(in_channels, skip_channels, 1),
            nn.BatchNorm2d(skip_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(skip_channels, skip_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, f_dec, hf_skip):
        # f_dec: [B, C_in, H, W]
        # hf_skip: [B, C_skip, H, W]

        # 生成门控权重 [0, 1]
        mask = self.gate_net(f_dec)

        # 软门控筛选高频信息
        return hf_skip * mask


# ==============================================================================
# 2. 光照引导模块 (Illumination Guidance Module)
# ==============================================================================

class IlluminationGuidance(nn.Module):
    def __init__(self):
        super(IlluminationGuidance, self).__init__()
        self.guide_net = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 1, 1, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, img_i):
        guidance = self.guide_net(img_i)
        return guidance


# ==============================================================================
# 3. 双域门控模块 (Dual-Domain Block)
# ==============================================================================

class FrequencyGatedAttention(nn.Module):
    def __init__(self, channels):
        super(FrequencyGatedAttention, self).__init__()
        self.conv1x1 = nn.Conv2d(channels, channels, 1)

    def forward(self, x, guidance):
        # FFT
        x_fft = torch.fft.rfft2(x, norm='backward')
        amp = torch.abs(x_fft)
        pha = torch.angle(x_fft)

        # Resize guidance
        g_resized = F.interpolate(guidance, size=amp.shape[-2:], mode='bilinear', align_corners=False)

        # Gate
        amp_mod = self.conv1x1(amp)
        amp_new = amp + amp_mod * g_resized

        # iFFT
        x_fft_new = amp_new * torch.exp(1j * pha)
        x_out = torch.fft.irfft2(x_fft_new, s=x.shape[-2:], norm='backward')

        return x_out


class SpatialGatedConv(nn.Module):
    def __init__(self, channels):
        super(SpatialGatedConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        )
        self.attn = nn.Sequential(
            nn.Conv2d(channels + 1, channels // 2, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 2, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x, guidance):
        feat = self.conv(x)
        g_resized = F.interpolate(guidance, size=x.shape[-2:], mode='bilinear', align_corners=False)
        attn_map = self.attn(torch.cat([x, g_resized], dim=1))
        return x + feat * attn_map


class DualDomainBlock(nn.Module):
    def __init__(self, channels):
        super(DualDomainBlock, self).__init__()
        self.freq_gate = FrequencyGatedAttention(channels)
        self.spatial_gate = SpatialGatedConv(channels)
        self.fusion = nn.Conv2d(channels * 2, channels, 1)

    def forward(self, x, guidance):
        f_freq = self.freq_gate(x, guidance)
        f_spatial = self.spatial_gate(x, guidance)
        return self.fusion(torch.cat([f_freq, f_spatial], dim=1)) + x


# ==============================================================================
# [改进点 2] Transformer 瓶颈层 (Global Context)
# ==============================================================================

class TransformerBlock(nn.Module):
    """
    使用简单的 Transformer 结构替换 Bottleneck。这里实现一个基于 Multi-Head Self Attention 的模块。
    为了适应不同分辨率，这里简化为全局 Attention (因为 Bottleneck 分辨率通常较小，如 32x32 或 16x16)。
    """
    def __init__(self, channels, num_heads=4, ffn_expansion=2.66):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, int(channels * ffn_expansion)),
            nn.GELU(),
            nn.Linear(int(channels * ffn_expansion), channels)
        )

    def forward(self, x, guidance=None):
        # guidance 在这里是可选的，或者可以通过 concat 注入，这里为了简洁主要关注 Global Context
        b, c, h, w = x.shape
        x_flat = x.flatten(2).transpose(1, 2)  # [B, H*W, C]

        # Attention
        x_norm = self.norm1(x_flat)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x_flat = x_flat + attn_out

        # FFN
        x_norm = self.norm2(x_flat)
        ffn_out = self.ffn(x_norm)
        x_flat = x_flat + ffn_out

        # Reshape back
        out = x_flat.transpose(1, 2).reshape(b, c, h, w)
        return out


# ==============================================================================
# 4. 上采样 (UpSample)
# ==============================================================================

class WaveletGuidedUpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(WaveletGuidedUpSample, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # 处理跳跃过来的高频特征 (输入通道是 out_channels * 3)
        self.high_freq_proc = nn.Sequential(
            nn.Conv2d(out_channels * 3, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )

    def forward(self, x, x_high_freq):
        # x_high_freq 可以是全零 (旧版 I流) 或 Gated后的特征 (新版 I流)
        x_up = self.conv(self.up(x))
        hf_feat = self.high_freq_proc(x_high_freq)
        return x_up + hf_feat


# ==============================================================================
# 5. 改进后的整体架构 (HVI_FGNet_Improved)
# ==============================================================================

class HVI_FGNet(nn.Module):
    def __init__(self, channels=[32, 64, 128, 256]):
        super(HVI_FGNet, self).__init__()
        self.trans = RGB_HVI()
        [c1, c2, c3, c4] = channels

        # --- 光照先验生成器 ---
        self.igm = IlluminationGuidance()

        # --- 初始卷积 ---
        self.conv_i_0 = nn.Conv2d(1, c1, 3, 1, 1)
        self.conv_hv_0 = nn.Conv2d(2, c1, 3, 1, 1)

        # --- Encoder Path ---
        # Level 1
        self.i_enc1 = DualDomainBlock(c1)
        self.hv_enc1 = DualDomainBlock(c1)
        self.bridge1 = InteractiveFusion(c1)  # [New]
        self.down1_i = DWTDownSample(c1, c2)
        self.down1_hv = DWTDownSample(c1, c2)

        # Level 2
        self.i_enc2 = DualDomainBlock(c2)
        self.hv_enc2 = DualDomainBlock(c2)
        self.bridge2 = InteractiveFusion(c2)  # [New]
        self.down2_i = DWTDownSample(c2, c3)
        self.down2_hv = DWTDownSample(c2, c3)

        # Level 3
        self.i_enc3 = DualDomainBlock(c3)
        self.hv_enc3 = DualDomainBlock(c3)
        self.bridge3 = InteractiveFusion(c3)  # [New]
        self.down3_i = DWTDownSample(c3, c4)
        self.down3_hv = DWTDownSample(c3, c4)

        # --- Bottleneck (Transformer) ---
        self.i_bot = TransformerBlock(c4)
        self.hv_bot = TransformerBlock(c4)

        # --- Decoder Path ---
        # [New] 高频门控模块初始化
        # Gate 的输入是解码器当前层 c_in，要门控的目标是 skip 过来的高频 (c_out * 3)
        self.hf_gate3 = HighFreqGating(c4, c3 * 3)
        self.hf_gate2 = HighFreqGating(c3, c2 * 3)
        self.hf_gate1 = HighFreqGating(c2, c1 * 3)

        # Level 3
        self.up3_i = WaveletGuidedUpSample(c4, c3)
        self.up3_hv = WaveletGuidedUpSample(c4, c3)
        self.i_dec3 = DualDomainBlock(c3)
        self.hv_dec3 = DualDomainBlock(c3)

        # Level 2
        self.up2_i = WaveletGuidedUpSample(c3, c2)
        self.up2_hv = WaveletGuidedUpSample(c3, c2)
        self.i_dec2 = DualDomainBlock(c2)
        self.hv_dec2 = DualDomainBlock(c2)

        # Level 1
        self.up1_i = WaveletGuidedUpSample(c2, c1)
        self.up1_hv = WaveletGuidedUpSample(c2, c1)
        self.i_dec1 = DualDomainBlock(c1)
        self.hv_dec1 = DualDomainBlock(c1)

        # --- Fusion ---
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(c1 * 2, c1, 3, 1, 1),
            SEBlock(c1),
            nn.Conv2d(c1, 3, 3, 1, 1)
        )

    def forward(self, x):
        # 1. HVI 变换
        hvi = self.trans.HVIT(x)
        h = hvi[:, 0:1, :, :]
        v = hvi[:, 1:2, :, :]
        i = hvi[:, 2:3, :, :]
        hv = torch.cat([h, v], dim=1)

        # 2. Guidance
        guidance = self.igm(i)

        # 3. Init
        f_i = self.conv_i_0(i)
        f_hv = self.conv_hv_0(hv)

        # 4. Encoder Path
        # L1
        f_i = self.i_enc1(f_i, guidance)
        f_hv = self.hv_enc1(f_hv, guidance)
        f_i, f_hv = self.bridge1(f_i, f_hv)  # [New] 交互

        f_i_skip1 = f_i
        f_hv_skip1 = f_hv

        # Down 1
        f_i, hf_i1 = self.down1_i(f_i)  # I 流的高频也保留下来，用于门控
        f_hv, hf_hv1 = self.down1_hv(f_hv)

        # L2
        f_i = self.i_enc2(f_i, guidance)
        f_hv = self.hv_enc2(f_hv, guidance)
        f_i, f_hv = self.bridge2(f_i, f_hv)  # [New] 交互

        f_i_skip2 = f_i
        f_hv_skip2 = f_hv

        # Down 2
        f_i, hf_i2 = self.down2_i(f_i)
        f_hv, hf_hv2 = self.down2_hv(f_hv)

        # L3
        f_i = self.i_enc3(f_i, guidance)
        f_hv = self.hv_enc3(f_hv, guidance)
        f_i, f_hv = self.bridge3(f_i, f_hv)  # [New] 交互

        f_i_skip3 = f_i
        f_hv_skip3 = f_hv

        # Down 3
        f_i, hf_i3 = self.down3_i(f_i)
        f_hv, hf_hv3 = self.down3_hv(f_hv)

        # 5. Bottleneck (Transformer)
        # [New] 使用 Transformer 获取全局特征
        f_i = self.i_bot(f_i)
        f_hv = self.hv_bot(f_hv)

        # 6. Decoder Path
        # L3
        # [New] I流: 使用 Gate 筛选 hf_i3 (之前是 hf_i3 直接丢弃)
        # f_i (C4) -> Gate -> Mask for hf_i3 (C3*3)
        gated_hf_i3 = self.hf_gate3(f_i, hf_i3)
        f_i = self.up3_i(f_i, gated_hf_i3) + f_i_skip3
        # HV流: 仍然直接使用 hf_hv3 (色彩流通常需要所有高频)
        f_hv = self.up3_hv(f_hv, hf_hv3) + f_hv_skip3

        f_i = self.i_dec3(f_i, guidance)
        f_hv = self.hv_dec3(f_hv, guidance)

        # L2
        gated_hf_i2 = self.hf_gate2(f_i, hf_i2)
        f_i = self.up2_i(f_i, gated_hf_i2) + f_i_skip2

        f_hv = self.up2_hv(f_hv, hf_hv2) + f_hv_skip2

        f_i = self.i_dec2(f_i, guidance)
        f_hv = self.hv_dec2(f_hv, guidance)

        # L1
        gated_hf_i1 = self.hf_gate1(f_i, hf_i1)
        f_i = self.up1_i(f_i, gated_hf_i1) + f_i_skip1

        f_hv = self.up1_hv(f_hv, hf_hv1) + f_hv_skip1

        f_i = self.i_dec1(f_i, guidance)
        f_hv = self.hv_dec1(f_hv, guidance)

        # 7. Fusion
        f_cat = torch.cat([f_i, f_hv], dim=1)
        residual = self.fusion_conv(f_cat)

        out_hvi = hvi + residual
        out_rgb = self.trans.PHVIT(out_hvi)

        return out_rgb