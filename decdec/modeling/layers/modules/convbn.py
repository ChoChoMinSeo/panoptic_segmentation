import torch.nn as nn
import torch.nn.functional as F
import torch

def get_activation_fn(activation):
    if activation is None or activation.lower() == 'none':
        return nn.Identity()
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "swish":
        return F.silu
    else:
        raise NotImplementedError
    
def get_norm(name, channels):
    if name is None or name.lower() == 'none':
        return nn.Identity()

    if name.lower() == 'syncbn':
        return nn.SyncBatchNorm(channels, eps=1e-3, momentum=0.01)
    
class ConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, norm='None', act='relu',
                 conv_type='2d', norm_init=1.0):
        super().__init__()
        
        if conv_type == '2d':
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        elif conv_type == '1d':
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

        self.norm = get_norm(norm, out_channels)
        self.act = get_activation_fn(act)

        
        nn.init.xavier_uniform_(self.conv.weight)
        if bias:
            nn.init.zeros_(self.conv.bias)
        if norm is not None:
            nn.init.constant_(self.norm.weight, norm_init)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))
    

class ASPP(nn.Module):
    def __init__(self, in_channels, output_channels, atrous_rates):
        super().__init__()

        self._aspp_conv0 = ConvBN(in_channels, output_channels, kernel_size=1, bias=False,
                                  norm='syncbn', act='gelu')

        rate1, rate2, rate3 = atrous_rates
        self._aspp_conv1 = ConvBN(in_channels, output_channels, kernel_size=3, dilation=rate1, padding=rate1, bias=False,
                                  norm='syncbn', act='gelu')

        self._aspp_conv2 = ConvBN(in_channels, output_channels, kernel_size=3, dilation=rate2, padding=rate2, bias=False,
                                  norm='syncbn', act='gelu')

        self._aspp_conv3 = ConvBN(in_channels, output_channels, kernel_size=3, dilation=rate3, padding=rate3, bias=False,
                                  norm='syncbn', act='gelu')

        self._avg_pool = nn.AdaptiveAvgPool2d(1)
        self._aspp_pool = ConvBN(in_channels, output_channels, kernel_size=1, bias=False,
                                 norm='syncbn', act='gelu')

        self._proj_conv_bn_act = ConvBN(output_channels * 5, output_channels, kernel_size=1, bias=False,
                                 norm='syncbn', act='gelu')
        # https://github.com/google-research/deeplab2/blob/main/model/decoder/aspp.py#L249
        self._proj_drop = nn.Dropout(p=0.1)

    def forward(self, x):
        results = []
        results.append(self._aspp_conv0(x))
        results.append(self._aspp_conv1(x))
        results.append(self._aspp_conv2(x))
        results.append(self._aspp_conv3(x))
        align_corners = (x.shape[-1] % 2 == 1)
        results.append(F.interpolate(self._aspp_pool(self._avg_pool(x)), size=x.shape[-2:], mode='bilinear', align_corners=align_corners))

        x = torch.cat(results, dim=1)
        x = self._proj_conv_bn_act(x)
        x = self._proj_drop(x)

        return x
    
