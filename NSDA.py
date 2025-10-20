import torch
import torch.nn as nn
import torch.nn.functional as F


class NSDA(nn.Module):
    # NSDA without Dynamic Neighborhood Scaling (DyNS)
    def __init__(self, in_channels, out_channels, window_size=(31, 31), shortcut=True, unbiased=False):
        super(NSDA, self).__init__()

        self.kernel_size = window_size
        self.padding = int(window_size[0] // 2)
        self.unbiased = unbiased
        self.use_add = shortcut and in_channels == out_channels

        
    def forward(self, x):
        with torch.no_grad():
            mean_values     = F.avg_pool2d(x, self.kernel_size, stride=1, padding=self.padding)

            squared_diff    = (x - mean_values).pow(2)

            variance_values = F.avg_pool2d(squared_diff, self.kernel_size, stride=1, padding=self.padding)

            if self.unbiased: # Bessel's correction
                variance_values = variance_values * (self.kernel_size[0] * self.kernel_size[1]) / (self.kernel_size[0] * self.kernel_size[1] - 1)
   
            attention_map = 1 - torch.exp(-squared_diff / (2 * variance_values + 1e-6)) # Gaussian-kernel-based dissimilarity measure
            '''
            # attention_map = torch.exp(-squared_diff / (2 * variance_values + 1e-6))   # Gaussian kernel for the similarity measure
            # attention_map = torch.sigmoid(torch.abs(x - mean_values))                 # dissimilarity measure based on sigmoid-activated Euclidean distance
            '''

        out = x * attention_map

        if self.use_add:
            out += x

        return out
    

if __name__ == '__main__':

    input = torch.randn(2, 3, 8, 8)
    b, c, h, w = input.shape

    # NSDA without Dynamic Neighborhood Scaling (DyNS)
    attention = NSDA(in_channels=c, out_channels=c)
    output1 = attention(input)

    # DyNS-equipped NSDA 
    if (h//8)%2 == 0:
        output2 = NSDA(in_channels=c, out_channels=c,  window_size=(h//8+1, w//8+1))(input)
    else:
        output2 = NSDA(in_channels=c, out_channels=c, window_size=(h//8+2, w//8+2))(input)

    print(f'Shape of the input: {input.shape}\nShape of the output1: {output1.shape}\nShape of the output2: {output2.shape}')
