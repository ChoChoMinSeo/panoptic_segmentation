import torch.nn as nn
import torch.nn.functional as F
import torch
from .modules import MultiheadSelfAttention,MultiheadCrossAttention,FeedForwardNetwork, ConvBN, PositionalEmbedding1D, get_sinusoid_encoding

class PixelDecoderLayer(nn.Module):
    def __init__(
        self, 
        resolution,
        query_dim,
        d_model,
        num_heads,
        ffn_dim,
        dropout=0.1,
        activation_fn = 'relu',
        interpolate = True
    ):
        super().__init__()
        self.interpolate = interpolate
        self.pos_emb = PositionalEmbedding1D(seq_len = resolution**2, emb_dim = d_model, relative = True)
        self.cross_attention = MultiheadCrossAttention(d_model, num_heads, dropout, softmax_dim=1)
        self.self_attention = MultiheadSelfAttention(d_model, num_heads, dropout)
        self.ffn = FeedForwardNetwork(d_model, ffn_dim, activation_fn,dropout)
        self.ln_ca = nn.LayerNorm(d_model)
        self.ln_sa = nn.LayerNorm(d_model)
        self.ln_ffn = nn.LayerNorm(d_model)
        self.feature_conv = ConvBN(query_dim, d_model, kernel_size= 1, stride=1, padding=0, norm='syncbn', conv_type = '1d')
        if self.interpolate:
            self.final_conv = ConvBN(d_model, query_dim//2, kernel_size= 1, stride=1, padding=0, norm='syncbn', conv_type = '1d')
        else:
            self.final_conv = ConvBN(d_model, query_dim, kernel_size= 1, stride=1, padding=0, norm='syncbn', conv_type = '1d')


    def forward(self,feature, object_query):
        b,c,h,w = feature.shape
        feature = feature.flatten(start_dim = 2)
        feature = self.feature_conv(feature).transpose(1,2)
        feature = self.pos_emb(feature)


        residual = feature
        feature = self.cross_attention(query = feature, key = object_query, value = object_query)
        feature = self.ln_ca(feature+residual)

        residual = feature
        feature = self.self_attention(query = feature, key = feature, value = feature)
        feature = self.ln_sa(feature+residual)

        residual = feature
        feature = self.ffn(feature)
        feature = self.ln_ffn(feature+residual).transpose(1,2)
        
        feature = self.final_conv(feature)

        if self.interpolate:
            feature = F.interpolate(feature.reshape(b,c//2,h,w), scale_factor=(2,2),mode='bilinear')
        else:
            feature = feature.reshape(b,c,h,w)
        return feature
        
class PixelDecoderLayerNoCA(nn.Module):
    def __init__(
        self, 
        resolution,
        query_dim,
        d_model,
        num_heads,
        ffn_dim,
        dropout=0.1,
        activation_fn = 'gelu',
        interpolate = False
    ):
        super().__init__()
        self.interpolate = interpolate
        self.pos_emb = PositionalEmbedding1D(seq_len = resolution**2, emb_dim = d_model, relative = True)
        self.self_attention = MultiheadSelfAttention(d_model, num_heads, dropout)
        self.ffn = FeedForwardNetwork(d_model, ffn_dim, activation_fn,dropout)
        self.ln_sa = nn.LayerNorm(d_model)
        self.ln_ffn = nn.LayerNorm(d_model)
        self.feature_conv = ConvBN(query_dim, d_model, kernel_size= 1, stride=1, padding=0, norm='syncbn', conv_type = '1d')
        if self.interpolate:
            self.final_conv = ConvBN(d_model, query_dim//2, kernel_size= 1, stride=1, padding=0, norm='syncbn', conv_type = '1d')
        else:
            self.final_conv = ConvBN(d_model, query_dim, kernel_size= 1, stride=1, padding=0, norm='syncbn', conv_type = '1d')

    def forward(self,feature,object_query):
        b,c,h,w = feature.shape
        feature = feature.flatten(start_dim = 2)
        feature = self.feature_conv(feature).transpose(1,2)
        feature = self.pos_emb(feature)

        residual = feature
        feature = self.self_attention(query = feature, key = feature, value = feature)
        feature = self.ln_sa(feature+residual)

        residual = feature
        feature = self.ffn(feature)
        feature = self.ln_ffn(feature+residual).transpose(1,2)

        feature = self.final_conv(feature)

        if self.interpolate:
            feature = F.interpolate(feature.reshape(b,c//2,h,w), scale_factor=(2,2),mode='bilinear')
        else:
            feature = feature.reshape(b,c,h,w)

        return feature
        
# model = PixelDecoderLayer(query_dim=2048,d_model=256,num_heads=8,ffn_dim=2048,interpolate=False)
# # model = PixelDecoderLayerNoCA(query_dim=2048,d_model=256,num_heads=8,ffn_dim=2048)

# f = torch.randn(1,2048,8,8)
# c = torch.randn(1,100,256)
# f,attn = model(f,c)
# # f = model(f)

# print(f.shape)
# print(attn.shape)