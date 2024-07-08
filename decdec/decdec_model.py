import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling import (
    Backbone, 
    PixelDecoderLayerNoCA,
    PixelDecoderLayer,
    TransformerDecoderLayer,
    Predictor,
    ConvBN,
    
)
class DecDec(nn.Module):
    def __init__(
        self, 
        # transformer_decoder:nn.ModuleList,
        # sem_seg_head:nn.Module,
        # class_head:nn.Module,

        backbone_name = 'resnet50',
        freeze_backbone = False,
        num_queries = 100,
        num_classes = 133,
        feature_dim=[2048,1024,512,256,256],
        pixel_decoder_layers=[1,5,1,1],
        transformer_decoder_layers=[2,2,2,0],
        d_model = 256,
        ffn_dim = 2048,
        n_head = 8,
        aux_loss = True
    ):
        super().__init__()
        self.pixel_decoder_layers = pixel_decoder_layers
        self.transformer_decoder_layers = transformer_decoder_layers
        self.object_query = nn.Parameter(torch.zeros(1,num_queries,d_model,requires_grad=True))
        self.backbone = Backbone(backbone_name,freeze_backbone)

        self.pixel_decoder = nn.ModuleList()
        self.transformer_decoder = nn.ModuleList()
        for i in range(3):
            if i==0:
                layer = nn.ModuleList([PixelDecoderLayerNoCA(feature_dim[i],d_model,n_head,ffn_dim,dropout=0.1, activation_fn = 'gelu',interpolate = True)])
                for _ in range(pixel_decoder_layers[i]-1):
                    layer.append(PixelDecoderLayerNoCA(feature_dim[i+1],d_model,n_head,ffn_dim,dropout=0.1, activation_fn = 'gelu',interpolate = False))
                self.pixel_decoder.append(layer)
            else:
                layer = nn.ModuleList([PixelDecoderLayer(feature_dim[i],d_model,n_head,ffn_dim,dropout=0.1, activation_fn = 'gelu',interpolate = True)])
                for _ in range(pixel_decoder_layers[i]-1):
                    layer.append(PixelDecoderLayer(feature_dim[i+1],d_model,n_head,ffn_dim,dropout=0.1, activation_fn = 'gelu',interpolate = False))
                self.pixel_decoder.append(layer)
            self.transformer_decoder.append(nn.ModuleList([TransformerDecoderLayer(feature_dim[i+1],d_model,n_head,ffn_dim,dropout=0.1,activation_fn='gelu')]*transformer_decoder_layers[i]))

        layer = nn.ModuleList([PixelDecoderLayer(feature_dim[-1],d_model,n_head,ffn_dim,dropout=0.1, activation_fn = 'gelu',interpolate = False)]*pixel_decoder_layers[-1])
        self.pixel_decoder.append(layer)

        self.final_feature_conv = ConvBN(feature_dim[-1],feature_dim[-1],kernel_size=1, bias=False, norm='syncbn', act=None)

        self.seg_head = Predictor(in_channel_pixel=feature_dim[-1],in_channel_query=d_model,num_classes=num_classes+1)
        # self.transformer_decoder = transformer_decoder()
        # self.seg_head = sem_seg_head
        # self.class_head = class_head
        self.class_proj = ConvBN(256, 256, kernel_size=1, bias=False, norm='syncbn', act='gelu',conv_type='1d')
        self.mask_proj = ConvBN(256, 256, kernel_size=1, bias=False, norm='syncbn', act='gelu',conv_type='1d')

        self.aux_loss = aux_loss
    def forward(self,x):
        bsz, c,h,w = x.shape
        feature = self.backbone(x)
        object_query = self.object_query.repeat(bsz,1,1)
        attn_maps = []
        # feature_maps = []
        for idx,pix_layers in enumerate(self.pixel_decoder_layers):
            for i in range(pix_layers):
                feature,attn_map = self.pixel_decoder[idx][i](feature,object_query)
            attn_maps.append(attn_map)
            # feature_maps.append(feature)
            for i in range(self.transformer_decoder_layers[idx]):
                object_query = self.transformer_decoder[idx][i](feature,object_query)

        feature = F.interpolate(feature,scale_factor=(4,4))
        feature = self.final_feature_conv(feature)

        object_query = object_query.transpose(1,2)
        class_emb = self.class_proj(object_query)
        mask_emb = self.mask_proj(object_query)
        
        # predict
        prediction_result = self.seg_head(class_emb,mask_emb,feature)

        predictions_class = []
        predictions_mask = []
        predictions_pixel_feature = []

        predictions_class.append(prediction_result['class_logits'])
        predictions_mask.append(prediction_result['mask_logits'])
        predictions_pixel_feature.append(prediction_result['pixel_feature'])
        # pred_class = self.class_head(object_query)
        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'pixel_feature': predictions_pixel_feature[-1],
            'aux_results': attn_maps   
        }
        return out
    

    
x = torch.randn(1,3,256,256)
model = DecDec()
out = model(x)
print(out['pred_logits'].shape,out['pred_masks'].shape,out['pixel_feature'].shape)