from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

@dataclass
class configGPT:
    block_size: int=256
    vocab_size: int=65
    n_layer: int=6
    n_head: int=6
    n_embd: int=384

class Block(nn.Module):
    """
    a single transformer block that contains self-attention and mlp layers. applies pre-normalization 
    and residual connections around both layers. follows the architecture of gpt-style transformers.
    """
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class MLP(nn.Module):
    """
    """
    def __init__(self, config):
        pass

class CausalSelfAttention(nn.Module):
    """
    """
    def __init__(self, config):
        pass

class babyGPT(nn.Module):
    """
    
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

        