import math 
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiheadSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5
        
        self.k_proj = nn.Linear(
            embed_dim, embed_dim, bias=True
        )
        self.q_proj = nn.Linear(
            embed_dim, embed_dim, bias=True
        )
        self.v_proj = nn.Linear(
            embed_dim, embed_dim, bias=True
        )
        self.out_proj = nn.Linear(
            embed_dim, embed_dim, bias=True
        )
        self.dropout_module = nn.Dropout(dropout)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)
    
    def forward(
        self,
        query,
        key,
        value,
    ):
        bsz,tgt_len,_ = query.size()
        
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        q *= self.scaling
        
        q = q.view(bsz,tgt_len, self.num_heads, self.head_dim).transpose(1,2)
        k = k.view(bsz,tgt_len, self.num_heads, self.head_dim).transpose(1,2)
        v = v.view(bsz,tgt_len, self.num_heads, self.head_dim).transpose(1,2)
        q = q.reshape(bsz * self.num_heads, tgt_len, self.head_dim)
        k = k.reshape(bsz * self.num_heads, tgt_len, self.head_dim)
        v = v.reshape(bsz * self.num_heads, tgt_len, self.head_dim)
        attn_weights = torch.bmm(q,k.transpose(1,2))
        
        attn_weights = F.softmax(attn_weights, dim = -1, dtype = torch.float32).type_as(attn_weights)
        attn_probs = self.dropout_module(attn_weights)
        attn = torch.bmm(attn_probs,v)
        attn = attn.transpose(0,1).reshape(tgt_len, bsz, self.embed_dim).transpose(0,1)
            
        attn = self.out_proj(attn)
        return attn
    
class MultiheadCrossAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        softmax_dim = -1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5
        self.softmax_dim = softmax_dim
        self.k_proj = nn.Linear(
            embed_dim, embed_dim, bias=True
        )
        self.q_proj = nn.Linear(
            embed_dim, embed_dim, bias=True
        )
        self.v_proj = nn.Linear(
            embed_dim, embed_dim, bias=True
        )
        self.out_proj = nn.Linear(
            embed_dim, embed_dim, bias=True
        )
        self.dropout_module = nn.Dropout(dropout)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)
    
    def forward(
        self,
        query,
        key,
        value,
    ):
        bsz,tgt_len,_ = query.size()
        _,src_len,_ = value.size()
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        q *= self.scaling
        
        q = q.view(bsz,tgt_len, self.num_heads, self.head_dim).transpose(1,2)
        k = k.view(bsz,src_len, self.num_heads, self.head_dim).transpose(1,2)
        v = v.view(bsz,src_len, self.num_heads, self.head_dim).transpose(1,2)
        q = q.reshape(bsz * self.num_heads, tgt_len, self.head_dim)
        k = k.reshape(bsz * self.num_heads, src_len, self.head_dim)
        v = v.reshape(bsz * self.num_heads, src_len, self.head_dim)
        attn_weights = torch.bmm(q,k.transpose(1,2))
        
        attn_weights = F.softmax(attn_weights, dim = self.softmax_dim, dtype = torch.float32).type_as(attn_weights)
        attn_probs = self.dropout_module(attn_weights)
        attn = torch.bmm(attn_probs,v)
        attn = attn.transpose(0,1).reshape(tgt_len, bsz, self.embed_dim).transpose(0,1)
            
        attn = self.out_proj(attn)
        return attn, torch.mean(attn_weights.reshape(bsz,self.num_heads,tgt_len,src_len),dim=1).squeeze(1)