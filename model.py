import math
from turtle import forward
import torch
import torch.nn as nn
import torch.functional as F

class LayerNormalization(nn.Module):
    
    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # start with 1
        self.beta = nn.Parameter(torch.zeros(1)) # start with 0
        
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        mean = x.mean(dim = -1, keepdim=True)
        std = x.std(dim = -1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.beta
    
class FeedForwardLayer(nn.Module):
    
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        x = self.dropout(torch.relu(self.linear1(x)))
        x = self.linear2(x)
        return x
    
class InputEmbedding(nn.Module):
    
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, dtype=torch.float32)
        
    def forward(self, x: torch.LongTensor) -> torch.FloatTensor:
        # x: (batch_size, seq_len) --> (batch_size, seq_len, d_model)
        return self.embedding(x) * math.sqrt(self.d_model)
    

class PositionEmbedding(nn.Module):
    
    def __init__(self, d_model: int, max_seq_len: int, dropout: float) -> None:
        super().__init__()
        assert d_model % 2 == 0, 'd_model must be even'
        
        self.dropout = nn.Dropout(dropout)
        
        # pe: (max_seq_len, d_model)
        pe = torch.zeros(max_seq_len, d_model)
        # position: (max_seq_len, 1)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        # div_term: (d_model // 2, )
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # pe: (1, max_seq_len, d_model)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = x + (self.pe[:, :x.size(1), :]).requires_grad_(False)
        return self.dropout(x)

class ResidualConnection(nn.Module):
    
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()
        
    def forward(self, x: torch.FloatTensor, sublayer: nn.Module) -> torch.FloatTensor:
        return x + self.dropout(sublayer(self.norm(x)))
    
class MultiHeadAttentionBlock(nn.Module):
    
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        assert d_model % n_heads == 0, 'd_model must be divisible by n_heads'
        self.dropout = nn.Dropout(dropout)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        
    @staticmethod
    def attention(q: torch.FloatTensor, k: torch.FloatTensor, v: torch.FloatTensor, mask, dropout: nn.Dropout):
        # (batch_size, n_heads, seq_len, d_k) --> (batch_size, n_heads, seq_len, seq_len)
        score = (q @ k.transpose(-2, -1)) / math.sqrt(v.size(-1))
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)
        score = score.softmax(dim = -1)
        if dropout is not None:
            score = dropout(score)
        return score @ v, score
        
    def forward(self, q, k, v, mask) -> torch.FloatTensor:
        # (batch_size, seq_len, d_model) --> (batch_size, n_heads, seq_len, d_k)
        query = self.w_q(q).view(-1, q.size(1), self.n_heads, self.d_k).transpose(1, 2)
        key = self.w_k(k).view(-1, k.size(1), self.n_heads, self.d_k).transpose(1, 2)
        value = self.w_v(v).view(-1, v.size(1), self.n_heads, self.d_k).transpose(1, 2)
        
        # (batch_size, n_heads, seq_len, d_k) --> (batch_size, n_heads, seq_len, d_k)
        x, score = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        # (batch_size, n_heads, seq_len, d_k) --> (batch_size, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(-1, x.size(1), self.n_heads * self.d_k)
        return self.w_o(x)
    
    
class EncoderBlock(nn.Module):
    
    def __init__(self, self_attention: MultiHeadAttentionBlock, feed_forward: FeedForwardLayer, dropout: float) -> None:
        super().__init__()
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        self.residual1 = ResidualConnection(dropout)
        self.residual2 = ResidualConnection(dropout)
        
    def forward(self, x: torch.FloatTensor, src_mask) -> torch.FloatTensor:
        x = self.residual1(x, lambda x: self.self_attention(x, x, x, src_mask))
        x = self.residual2(x, self.feed_forward)
        return x
    
   
class Encoder(nn.Module):
    """Some Information about Encoder"""
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)
    
class DecoderBlock(nn.Module):
    
    def __init__(self, self_attention: MultiHeadAttentionBlock, cross_attention: MultiHeadAttentionBlock, feed_forward: FeedForwardLayer, dropout: float) -> None:
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.feed_forward = feed_forward
        self.residual1 = ResidualConnection(dropout)
        self.residual2 = ResidualConnection(dropout)
        self.residual3 = ResidualConnection(dropout)
        
    def forward(self, x: torch.FloatTensor, encoder_output: torch.FloatTensor, src_mask, tgt_mask) -> torch.FloatTensor:
        x = self.residual1(x, lambda x: self.self_attention(x, x, x, tgt_mask))
        x = self.residual2(x, lambda x: self.cross_attention(x, encoder_output, encoder_output, src_mask))
        x = self.residual3(x, self.feed_forward)
        return x
    
class Decoder(nn.Module):
    
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
        
    def forward(self, x: torch.FloatTensor, encoder_output: torch.FloatTensor, src_mask, tgt_mask) -> torch.FloatTensor:
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
    
class ProjectionLayer(nn.Module):
    
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)
        
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return torch.log_softmax(self.linear(x), dim=-1) 
    
class Transformer(nn.Module):
    
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: nn.Module, tgt_embed: nn.Module, src_pos: PositionEmbedding, tgt_pos: PositionEmbedding, projection: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.src_pos = src_pos
        self.tgt_embed = tgt_embed
        self.tgt_pos = tgt_pos
        self.projection = projection
        
    def encode(self, src: torch.LongTensor, src_mask) -> torch.FloatTensor:
        return self.encoder(self.src_pos(self.src_embed(src)), src_mask)
    
    def decode(self, encoder_output: torch.FloatTensor, tgt: torch.LongTensor, src_mask, tgt_mask) -> torch.FloatTensor:
        return self.decoder(self.tgt_pos(self.tgt_embed(tgt)), encoder_output, src_mask, tgt_mask)
    
    def project(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.projection(x)