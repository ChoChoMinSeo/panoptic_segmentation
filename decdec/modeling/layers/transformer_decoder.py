import torch.nn as nn
import torch
from .modules import MultiheadSelfAttention,MultiheadCrossAttention,FeedForwardNetwork, ConvBN

class TransformerDecoderLayer(nn.Module):
    def __init__(
        self, 
        query_dim,
        d_model,
        num_heads,
        ffn_dim,
        dropout=0.1,
        activation_fn = 'relu',
    ):
        super().__init__()
        self.cross_attention = MultiheadCrossAttention(d_model, num_heads, dropout, softmax_dim=1)
        self.self_attention = MultiheadSelfAttention(d_model, num_heads, dropout)
        self.ffn = FeedForwardNetwork(d_model, ffn_dim, activation_fn,dropout)
        self.ln_ca = nn.LayerNorm(d_model)
        self.ln_sa = nn.LayerNorm(d_model)
        self.ln_ffn = nn.LayerNorm(d_model)
        self.feature_conv = ConvBN(query_dim, d_model, kernel_size= 1, stride=1, padding=0, norm='syncbn', conv_type = '1d')


    def forward(self,feature, object_query):
        feature = feature.flatten(start_dim = 2)
        feature = self.feature_conv(feature).transpose(1,2)

        residual = object_query
        object_query,_ = self.cross_attention(query = object_query, key = feature, value = feature)
        object_query = self.ln_ca(object_query+residual)

        residual = object_query
        object_query = self.self_attention(query = object_query, key = object_query, value = object_query)
        object_query = self.ln_sa(object_query+residual)

        residual = object_query
        object_query = self.ffn(object_query)
        object_query = self.ln_ffn(object_query+residual)
        return object_query
        
        
# model = TransformerDecoderLayer(query_dim=1024,d_model=256,num_heads=8,ffn_dim=2048)

# f = torch.randn(1,1024,16,16)
# c = torch.randn(1,100,256)
# f = model(f,c)
# print(f.shape)