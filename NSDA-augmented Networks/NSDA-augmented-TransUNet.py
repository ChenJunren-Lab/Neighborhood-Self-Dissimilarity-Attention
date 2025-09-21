import torch
import torch.nn as nn
import functools
import torch.nn.functional as F

from vit_seg_modeling import VisionTransformer as ViT_seg
from vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

from torchsummary import summary
from thop import clever_format, profile


def get_transUNet(num_classes):
    img_size = 256 # 512
    vit_patches_size = 16
    vit_name = 'R50-ViT-B_16'

    config_vit = CONFIGS_ViT_seg[vit_name]
    config_vit.num_classes = num_classes
    config_vit.n_skip = 3
    if vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(img_size / vit_patches_size), int(img_size / vit_patches_size))
    net = ViT_seg(config_vit, img_size=img_size, num_classes=num_classes)
    return net



if __name__ == '__main__':
    
    img = torch.randn((2, 3, 256, 256))
    net = get_transUNet(9)
    segments = net(img)
    print(segments.size())
    # for edge in edges:
    #     print(edge.size())
    
    
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = get_transUNet(num_classes=9)

    # # summary(net.to(device), input_size=(3, 512, 512))
    dummy_input     = torch.randn(1, 3, 256, 256).to(device)
    flops, params   = profile(net.to(device), (dummy_input, ), verbose=False)

    flops           = flops * 2
    flops, params   = clever_format([flops, params], "%.3f")
    print('Total GFLOPS: %s' % (flops))
    print('Total params: %s' % (params))