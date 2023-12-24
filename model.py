import math
from turtle import forward
import torch
import torch.nn as nn

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