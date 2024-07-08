import torch.nn as nn
import torch.nn.functional as F
import torch
from .modules import ASPP, ConvBN, get_norm
import math

class SemanticPredictor(nn.Module):
    def __init__(self, in_channels, os8_channels, os4_channels, num_classes):
        super().__init__()

        # Below is PanopticDeepLabSingleDecoder
        self._aspp = ASPP(
            in_channels=in_channels,
            # https://github.com/google-research/deeplab2/blob/main/configs/coco/kmax_deeplab/kmax_meta_r50_os32.textproto#L35
            output_channels=256,
            # https://github.com/google-research/deeplab2/blob/main/configs/coco/kmax_deeplab/kmax_meta_r50_os32.textproto#L36
            atrous_rates=[6,12,18])
        
        self._low_level_projection_os8 = ConvBN(os8_channels, 64, kernel_size=1, bias=False,
                                                norm='syncbn', act='gelu')

        self._low_level_fusion_os8_conv0_bn_act = ConvBN(256 + 64, 256 + 64, groups=256 + 64, kernel_size=5, padding=2, bias=False,
                                                         norm='syncbn', act='gelu')
        self._low_level_fusion_os8_conv1_bn_act = ConvBN(256 + 64, 256, kernel_size=1,bias=False,
                                                         norm='syncbn', act='gelu')

        self._low_level_projection_os4 = ConvBN(os4_channels, 32, kernel_size=1, bias=False,
                                                norm='syncbn', act='gelu')

        self._low_level_fusion_os4_conv0_bn_act = ConvBN(256 + 32, 256 + 32, groups=256 + 32, kernel_size=5, padding=2, bias=False,
                                                         norm='syncbn', act='gelu')
        self._low_level_fusion_os4_conv1_bn_act = ConvBN(256 + 32, 256, kernel_size=1,bias=False,
                                                         norm='syncbn', act='gelu')

        # Below is PanopticDeepLabSingleHead
        self.conv_block_0 = ConvBN(256, 256, groups=256, kernel_size=5, padding=2, bias=False,
                                   norm='syncbn', act='gelu')
        self.conv_block_1 = ConvBN(256, 256, kernel_size=1,bias=False,
                                   norm='syncbn', act='gelu')
        self.final_conv = ConvBN(256, num_classes, kernel_size=1, norm=None, act=None)

    def forward(self, x, low_features_os8, low_features_os4):
        x = self._aspp(x)
        align_corners = (x.shape[-1] % 2 == 1)
        low_features_os8 = self._low_level_projection_os8(low_features_os8)
        x = F.interpolate(x, size=low_features_os8.shape[-2:], mode='bilinear', align_corners=align_corners)
        x = torch.concat([x, low_features_os8], dim=1)
        x = self._low_level_fusion_os8_conv0_bn_act(x)
        x = self._low_level_fusion_os8_conv1_bn_act(x)

        low_features_os4 = self._low_level_projection_os4(low_features_os4)
        x = F.interpolate(x, size=low_features_os4.shape[-2:], mode='bilinear', align_corners=align_corners)
        x = torch.concat([x, low_features_os4], dim=1)
        x = self._low_level_fusion_os4_conv0_bn_act(x)
        x = self._low_level_fusion_os4_conv1_bn_act(x)

        x = self.conv_block_0(x)
        x = self.conv_block_1(x)
        x = self.final_conv(x)
        return x
    
# https://github.com/google-research/deeplab2/blob/7a01a7165e97b3325ad7ea9b6bcc02d67fecd07a/model/layers/resized_fuse.py#L31
class ResizedFuse(nn.Module):
    def __init__(self, low_in_channels, high_in_channels, out_channels):
        super().__init__()
        self.low_in_channels = low_in_channels
        self.high_in_channels = high_in_channels
        self.out_channels = out_channels
        if low_in_channels != out_channels:
            self._conv_bn_low = ConvBN(low_in_channels, out_channels, kernel_size=1, bias=False, norm='syncbn', act=None)
        if high_in_channels != out_channels:
            self._conv_bn_high = ConvBN(high_in_channels, out_channels, kernel_size=1, bias=False, norm='syncbn', act=None)

    def forward(self, lowres_x, highres_x):

        align_corners = (lowres_x.shape[-1] % 2 == 1)
        if self.low_in_channels != self.out_channels:
            lowres_x = F.gelu(lowres_x)
            lowres_x = self._conv_bn_low(lowres_x)
            lowres_x = F.interpolate(lowres_x, size=highres_x.shape[2:], mode='bilinear', align_corners=align_corners)
        else:
            lowres_x = F.interpolate(lowres_x, size=highres_x.shape[2:], mode='bilinear', align_corners=align_corners)

        if self.high_in_channels != self.out_channels:
            highres_x = F.gelu(highres_x)
            highres_x = self._conv_bn_high(highres_x)

        return lowres_x + highres_x
    
def add_bias_towards_void(query_class_logits, void_prior_prob=0.9):
    class_logits_shape = query_class_logits.shape
    init_bias = [0.0] * class_logits_shape[-1]
    init_bias[-1] = math.log(
      (class_logits_shape[-1] - 1) * void_prior_prob / (1 - void_prior_prob))
    return query_class_logits + torch.tensor(init_bias, dtype=query_class_logits.dtype).to(query_class_logits)

class Predictor(nn.Module):
    def __init__(self, in_channel_pixel, in_channel_query, num_classes=133+1):
        super().__init__()
        self._pixel_space_head_conv0bnact = ConvBN(in_channel_pixel, in_channel_pixel, kernel_size=5, groups=in_channel_pixel, padding=2, bias=False,
                                                   norm='syncbn', act='gelu')
        self._pixel_space_head_conv1bnact = ConvBN(in_channel_pixel, 256, kernel_size=1, bias=False, norm='syncbn', act='gelu')
        self._pixel_space_head_last_convbn = ConvBN(256, 128, kernel_size=1, bias=True, norm='syncbn', act=None)

        self._transformer_mask_head = ConvBN(256, 128, kernel_size=1, bias=False, norm='syncbn', act=None, conv_type='1d')
        self._transformer_class_head = ConvBN(256, num_classes, kernel_size=1, norm=None, act=None, conv_type='1d')

        self._pixel_space_mask_batch_norm = get_norm('syncbn', channels=1)
        nn.init.constant_(self._pixel_space_mask_batch_norm.weight, 0.1)


    def forward(self, mask_embeddings, class_embeddings, pixel_feature):
        # mask_embeddings/class_embeddings: B x C x N
        # pixel feature: B x C x H x W
        pixel_space_feature = self._pixel_space_head_conv0bnact(pixel_feature)
        pixel_space_feature = self._pixel_space_head_conv1bnact(pixel_space_feature)
        pixel_space_feature = self._pixel_space_head_last_convbn(pixel_space_feature)
        pixel_space_normalized_feature = F.normalize(pixel_space_feature, p=2, dim=1)

        cluster_class_logits = self._transformer_class_head(class_embeddings).permute(0, 2, 1).contiguous()
        cluster_class_logits = add_bias_towards_void(cluster_class_logits)
        cluster_mask_kernel = self._transformer_mask_head(mask_embeddings)
        mask_logits = torch.einsum('bchw,bcn->bnhw',
          pixel_space_normalized_feature, cluster_mask_kernel)
        
        mask_logits = self._pixel_space_mask_batch_norm(mask_logits.unsqueeze(dim=1)).squeeze(dim=1)


        return {
            'class_logits': cluster_class_logits,
            'mask_logits': mask_logits,
            'pixel_feature': pixel_space_normalized_feature}