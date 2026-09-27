import torch
import torch.nn as nn
import torch.nn.functional as F

#Simple Self_Attention
class SelfAttention(nn.Module):
    def __init__(self, embed_size):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        
        # weights for q, k, v
        self.Wq = nn.Linear(embed_size, embed_size, bias=False)
        self.Wk = nn.Linear(embed_size, embed_size, bias=False)
        self.Wv = nn.Linear(embed_size, embed_size, bias=False)

    def forward(self, x):
        
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)

        # multiply q and k
        scores = torch.matmul(q, k.transpose(1, 2))
        
        # scale
        scores = scores / (self.embed_size ** 0.5)
        
        # softmax
        weights = F.softmax(scores, dim=-1)
        
        # multiply by v
        out = torch.matmul(weights, v)
        return out


# Simple Encoder Layer
class EncoderBlock(nn.Module):
    def __init__(self, embed_size):
        super(EncoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size)
        
        # feed forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_size, embed_size * 4),
            nn.ReLU(),
            nn.Linear(embed_size * 4, embed_size)
        )
        
        # norm layers
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

    def forward(self, x):
        # self attention + residual
        att_out = self.attention(x)
        x = self.norm1(x + att_out)
        
        # ffn + residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x


# Test 
x = torch.randn(2, 5, 8) # batch=2  seq_len=5  embed_size=8
encoder = EncoderBlock(embed_size=8)
output = encoder(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)
