import torch
import torch.nn as nn
import torch.nn.functional as F


class NSDA(nn.Module):
    def __init__(self, in_channels, out_channels, window_size=(31, 31), shortcut=True, unbiased=True):
        super(NSDA, self).__init__()

        self.kernel_size = window_size
        self.padding = int(window_size[0] // 2)
        self.unbiased = unbiased
        self.use_add = shortcut and in_channels == out_channels

        
    def forward(self, x):
        with torch.no_grad():
            mean_values      = F.avg_pool2d(x, self.kernel_size, stride=1, padding=self.padding)

            squared_diff = (x - mean_values).pow(2)

            variance_values = F.avg_pool2d(squared_diff, self.kernel_size, stride=1, padding=self.padding)

            if self.unbiased:
                variance_values = variance_values * (self.kernel_size[0] * self.kernel_size[1]) / (self.kernel_size[0] * self.kernel_size[1] - 1)
   
            attention_map = 1 - torch.exp(-squared_diff / (2 * variance_values + 1e-6))

        out = x * attention_map

        if self.use_add:
            out += x

        return out


# 定义U-Net网络的编码器部分
class UNetEncoder(nn.Module):
    # 标准卷积+标准化+激活函数
    """Standard convolution with args(ch_in, ch_out, kernel, stride, paddin g, groups, dilation, activation)."""
    default_act = nn.ReLU() # default_act = nn.SiLU(), nn.GELU(), ReLU()  # default activation
    def __init__(self, in_channels, out_channels, act=True):
        super(UNetEncoder, self).__init__()
        self.conv1      = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2      = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn         = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.act        = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        

        self.attention  = NSDA(out_channels, out_channels ) #! attention
        
        
    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        # x = self.attention(x)
        # (Dynamic Neighborhood Scaling)DyNS-equipped NSDA 
        _,c,h,w = x.shape
        if (h//8)%2==0:
            x = NSDA(c, c, (h//8+1, w//8+1))(x)
        else:
            x = NSDA(c, c, (h//8+2, w//8+2))(x)

        
        skip_connection = x
        x = self.downsample(x)
        return x, skip_connection

# 定义U-Net网络的解码器部分
class UNetDecoder(nn.Module):
    # 标准卷积+标准化+激活函数
    """Standard convolution with args(ch_in, ch_out, kernel, stride, paddin g, groups, dilation, activation)."""
    default_act = nn.ReLU() # default_act = nn.SiLU(), nn.GELU(), ReLU()  # default activation
    def __init__(self, in_channels, out_channels, act=True):
        super(UNetDecoder, self).__init__()
        self.conv1     = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2     = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.upsample  = F.interpolate # nn.UpsamplingBilinear2d(scale_factor = 2) ; nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.bn        = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.act       = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        
        self.attention  = NSDA(out_channels, out_channels )
        
    def forward(self, x, skip_connection):
        x = self.upsample(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = torch.cat([x, skip_connection], dim=1)

        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        
        # x = self.attention(x)
        # (Dynamic Neighborhood Scaling)DyNS-equipped NSDA 
        _,c,h,w = x.shape

        if (h//8)%2==0:
            x = NSDA(c, c, (h//8+1, w//8+1))(x)
        else:
            x = NSDA(c, c, (h//8+2, w//8+2))(x)
        
        return x


class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super(UNet, self).__init__()
        in_filters  = [192, 384, 768, 1024]
        out_filters = [64, 128, 256, 512]
        
        self.encoder1 = UNetEncoder(in_channels, out_filters[0])
        self.encoder2 = UNetEncoder(out_filters[0], out_filters[1])
        self.encoder3 = UNetEncoder(out_filters[1], out_filters[2])
        self.encoder4 = UNetEncoder(out_filters[2], out_filters[3])
        self.center   = UNetEncoder(out_filters[3], out_filters[3])
        self.decoder4 = UNetDecoder(in_filters[3], out_filters[3])
        self.decoder3 = UNetDecoder(in_filters[2], out_filters[2])
        self.decoder2 = UNetDecoder(in_filters[1], out_filters[1])
        self.decoder1 = UNetDecoder(in_filters[0], out_filters[0])
        self.final_conv = nn.Conv2d(out_filters[0], num_classes, kernel_size=1)
        
    def forward(self, x):
        x, skip1 = self.encoder1(x)
        x, skip2 = self.encoder2(x)
        x, skip3 = self.encoder3(x)
        x, skip4 = self.encoder4(x)
        _, x = self.center(x)
        # print(f'self.center(x) shape: {x.shape}')
        x = self.decoder4(x, skip4)
        x = self.decoder3(x, skip3)
        x = self.decoder2(x, skip2)
        x = self.decoder1(x, skip1)
        x = self.final_conv(x)
        return x
    
    
if __name__ == '__main__':

    unet = UNet(num_classes=2)

    from thop import clever_format, profile
    from torchsummary import summary

    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    unet = unet.to(device)

    # summary(segformer, (3, 256, 256))
        
    dummy_input     = torch.randn(1, 3, 256, 256).to(device)
    flops, params   = profile(unet, (dummy_input, ), verbose=False)

    flops           = flops * 2
    flops, params   = clever_format([flops, params], "%.4f")
    print(f'Total GFLOPS/MACs: {flops}')
    print(f'Total params: {params}')