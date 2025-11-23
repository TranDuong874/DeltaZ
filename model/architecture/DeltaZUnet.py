import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, norm=True):
        super().__init__()
        pad = kernel_size // 2
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size, stride, pad, bias=not norm)]
        if norm:
            layers.append(nn.GroupNorm(8, out_ch))
        layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


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

def center_crop_to(x, target_hw):
    """Center crop tensor x to target height and width."""
    _, _, H, W = x.shape
    h, w = target_hw
    top = (H - h) // 2
    left = (W - w) // 2
    return x[..., top:top+h, left:left+w]
    
class DeltaZUnet(nn.Module):
    def __init__(self, in_channels=11, base_channel=32, depth=4, out_confidence=False):
        super().__init__()  # Fixed: was super.__init__()
        
        # Encoder
        self.encoder_convs = nn.ModuleList()
        ch = in_channels
        for i in range(depth):
            out_dim = base_channel * (2 ** i)
            self.encoder_convs.append(
                nn.Sequential(
                    ConvBlock(ch, out_dim, kernel_size=3),
                    ResidualBlock(out_dim)  # Fixed: was ResiduealBlock (typo)
                )
            )
            ch = out_dim
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            ConvBlock(ch, ch * 2, kernel_size=3),
            ResidualBlock(ch * 2)
        )
        
        # Decoder
        self.decoder_convs = nn.ModuleList()
        ch = ch * 2  # Start from bottleneck output channels
        for i in reversed(range(depth)):
            skip_ch = base_channel * (2 ** i)  # Skip connection channels
            out_dim = base_channel * (2 ** i)
            self.decoder_convs.append(
                nn.Sequential(
                    ConvBlock(ch + skip_ch, out_dim, kernel_size=3),  # ch + skip_ch for concatenation
                    ResidualBlock(out_dim)
                )
            )
            ch = out_dim
        
        # Output heads
        self.delta_head = nn.Conv2d(ch, 1, 3, 1, 1)
        self.out_confidence = out_confidence
        if out_confidence:
            self.conf_head = nn.Conv2d(ch, 1, 3, 1, 1)
        
        self.pool = nn.AvgPool2d(2)
    
    def forward(self, x):
        enc_feats = []
        h = x
        
        # Encoder
        for encoder in self.encoder_convs:
            h = encoder(h)
            enc_feats.append(h)  # Fixed: was encoded_feats (typo)
            h = self.pool(h)
        
        # Bottleneck
        h = self.bottleneck(h)
        
        # Decoder
        for i, decoder in enumerate(self.decoder_convs):
            # Upsample
            h = F.interpolate(h, scale_factor=2.0, mode='bilinear', align_corners=False)
            
            # Skip connection
            skip = enc_feats[-1 - i]
            if skip.shape[-2:] != h.shape[-2:]:
                skip = center_crop_to(skip, h.shape[-2:])
            
            h = torch.cat([h, skip], dim=1)
            h = decoder(h)
        
        # Output
        delta_z = self.delta_head(h)
        
        if self.out_confidence:
            confidence = torch.sigmoid(self.conf_head(h))
            return delta_z, confidence
        
        return delta_z



