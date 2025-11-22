# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------
# Basic blocks
# ----------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, norm=True):
        super().__init__()
        pad = kernel_size // 2
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size, stride, pad, bias=not norm)]
        if norm:
            layers.append(nn.GroupNorm(8, out_ch))
        layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = ConvBlock(ch, ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, 1, 1, bias=False)
        self.gn = nn.GroupNorm(8, ch)
    def forward(self, x):
        r = self.conv1(x)
        r = self.conv2(r)
        r = self.gn(r)
        return F.relu(x + r)

# ----------------------------------------
# DeltaZ UNet
# ----------------------------------------
class DeltaZUNet(nn.Module):
    def __init__(self, in_ch=11, base_ch=32, depth=4, out_confidence=False):
        super().__init__()
        # encoder
        self.enc_convs = nn.ModuleList()
        ch = in_ch
        for i in range(depth):
            out = base_ch * (2**i)
            self.enc_convs.append(nn.Sequential(
                ConvBlock(ch, out, ks=3),
                ResidualBlock(out)
            ))
            ch = out

        # bottleneck
        self.bottleneck = nn.Sequential(ConvBlock(ch, ch*2, ks=3), ResidualBlock(ch*2))

        # decoder
        self.dec_convs = nn.ModuleList()
        for i in reversed(range(depth)):
            out = base_ch * (2**i)
            self.dec_convs.append(nn.Sequential(
                ConvBlock(ch*2, out, ks=3),
                ResidualBlock(out)
            ))
            ch = out

        # outputs
        self.delta_head = nn.Conv2d(ch, 1, 3, 1, 1)
        self.out_confidence = out_confidence
        if out_confidence:
            self.conf_head = nn.Conv2d(ch, 1, 3, 1, 1)

        self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        enc_feats = []
        h = x
        # encoder (save skip)
        for enc in self.enc_convs:
            h = enc(h)
            enc_feats.append(h)
            h = self.pool(h)

        # bottleneck
        h = self.bottleneck(h)
        
        # decoder (upsample + skip)
        for i, dec in enumerate(self.dec_convs):
            h = F.interpolate(h, scale_factor=2.0, mode='bilinear', align_corners=False)
            skip = enc_feats[-1 - i]
            # if shapes mismatched, center-crop skip (safe guard)
            if skip.shape[-2:] != h.shape[-2:]:
                skip = center_crop_to(skip, h.shape[-2:])
            h = torch.cat([h, skip], dim=1)
            h = dec(h)
        delta = self.delta_head(h)           # raw Δz
        if self.out_confidence:
            conf = torch.sigmoid(self.conf_head(h))
            return delta, conf
        return delta

def center_crop_to(x, target_hw):
    _,_,H,W = x.shape
    h,w = target_hw
    top = (H - h)//2
    left = (W - w)//2
    return x[..., top:top+h, left:left+w]

# ----------------------------------------
# Multi-view refiner (warp neighbors features + small conv)
# ----------------------------------------
class MultiViewRefiner(nn.Module):
    def __init__(self, feat_ch=64, num_neighbors=4):
        super().__init__()
        # Takes reference feature and stacked warped neighbor features
        # Input channels = feat_ch * (1 + num_neighbors)
        self.num_neighbors = num_neighbors
        self.conv = nn.Sequential(
            ConvBlock(feat_ch*(1+num_neighbors), feat_ch),
            ResidualBlock(feat_ch),
            ConvBlock(feat_ch, feat_ch//2),
        )
        self.residual_out = nn.Conv2d(feat_ch//2, 1, 3, 1, 1)  # Δz residual

    def forward(self, ref_feat, warped_nei_feats):
        """
        ref_feat: [B, C, H, W]
        warped_nei_feats: list of [B, C, H, W] length num_neighbors
        """
        x = torch.cat([ref_feat] + warped_nei_feats, dim=1)
        x = self.conv(x)
        res = self.residual_out(x)
        return res
