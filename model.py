from dataclasses import dataclass
import torch
import math
import torch.nn as nn
from torch.nn import functional as F

@dataclass
class configGPT:
    block_size: int=256
    vocab_size: int=65
    n_layer: int=6
    n_head: int=6
    n_embd: int=384
    dropout: float=0.0
    bias: bool=True

class Block(nn.Module):
    """
    a single transformer block that contains self-attention and mlp layers. applies pre-normalization 
    and residual connections around both layers. follows the architecture of gpt-style transformers.
    """
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class MLP(nn.Module):
    """
    multi-layer perceptron block used in transformer architectures. expands embedding dimension by 4x, 
    applies gelu activation, then projects back. follows the architecture commonly used in gpt models.
    """
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd, bias=config.bias)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    
class CausalSelfAttention(nn.Module):
    """
    implements causal (masked) self-attention mechanism used in gpt models. applies multi-head 
    attention where each token can only attend to previous tokens
    """
    def __init__(self, config):
        super().__init__()
        # verify embedding dimension is divisible by number of heads
        assert config.n_embd % config.n_head == 0
        # single linear layer to compute query, key, value projections
        # output is concatenated [query|key|value] of total size 3*n_embd
        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd, bias=config.bias)
        # final projection layer to convert attention output back to embedding dimension
        self.output = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # dropout layers for regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        # save config parameters we'll need later
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # create causal mask (lower triangular matrix)
        # tokens can only attend to previous tokens in sequence
        self.regular_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size))
                            .view(1, 1, config.block_Size, config.block_size))
    
    def forward(self, x):
        b, t, c = x.size # batch size, sequence length, embedding dimension
        qkv = self.c_attn(x) # project input into query, key, value vectors
        q, k, v = qkv.split(self.n_embd, dim=2) # split into separate q,k,v tensors
        q = q.view(b, t, self.n_head, c//self.n_head).transport(1, 2) 

        # reshape q,k,v to separate multiple heads
        # shape changes: (batch, seq_len, n_embd) -> (batch, n_head, seq_len, head_size)
        k = k.view(b, t, self.n_head, c//self.n_head).transport(1, 2)
        v = v.view(b, t, self.n_head, c//self.n_head).transport(1, 2)

        # compute attention scores
        # (batch, n_head, seq_len, seq_len) = (batch, n_head, seq_len, head_size) @ (batch, n_head, head_size, seq_len)
        attn = (q @ k.transpose(-2, -1)) * (1/math.sqrt(k.size(-1)))

        # apply causal mask (prevent attending to future tokens)
        att = att.masked_fill(self.bias[:,:,:t,:t] == 0, float('-inf'))

        # softmax to get attention probabilities
        att = F.softmax(att, dim=-1)

        # apply dropout to attention matrix
        att = self.attn_dropout(att)

        # apply attention to values
        y = att @ v

        # reshape back to original dimensions
        # (batch, seq_len, n_embd)
        y = y.transpose(1, 2).contiguous().view(b, t, c) 
        y = self.resid_dropout(self.c_proj(y)) # final output projection with dropout
        return y
    
class babyGPT(nn.Module):
    """
    an implementation of the 117M model of ChatGPT-2s
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            # token embedding table - converts token ids to vectors of size n_embd
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            # position embedding table - adds positional information of length block_size
            wpe = nn.Embedding(config.block_size, config.n_embd),
            # stack of transformer blocks - processes the embeddings through n_layer blocks
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd) # for stabalization
        ))

        # projection layer to vocab size logits - converts final embeddings back to vocab probabilities
        # bias=False since final layer norm already adds a bias term
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        