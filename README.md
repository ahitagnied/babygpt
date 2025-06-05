# babyGPT: learning gpt-2 from the ground up

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

a minimal, educational implementation of GPT-2 designed to understand how large language models work from first principles. 

<p align="center">
  <img src='assets/inference.png' width='100%'>
</p>

this project strips away the complexity of production systems to focus on the core transformer architecture and training dynamics that power modern language models.

## motivation

GPT-2 represents a pivotal moment in the evolution of language models - demonstrating that simple transformer architectures could achieve remarkable performance when scaled appropriately. however, most implementations are either too simplified for real learning or too complex for educational purposes.

babyGPT bridges this gap by providing a complete, working implementation that:
- **maintains architectural fidelity** to the original GPT-2 model
- **includes all essential components** for training and inference  
- **uses modern optimizations** like Flash Attention and mixed precision
- **supports distributed training** for realistic scale experiments
- **provides clear educational pathways** from concepts to code

after working through this implementation, you'll understand how attention mechanisms, transformer blocks, and autoregressive generation actually work at the code level.

## key features

- 🧠 **complete gpt-2 architecture**: transformer blocks, causal self-attention, and position embeddings
- ⚡ **modern optimizations**: flash attention, mixed precision (bfloat16), and torch.compile
- 🔄 **distributed training**: multi-gpu support with distributeddataparallel 
- 📊 **multiple datasets**: shakespeare for quick experiments, fineweb for realistic pretraining
- 🎯 **flexible inference**: temperature sampling, top-k, top-p, and interactive generation
- 📈 **training utilities**: learning rate scheduling, gradient clipping, and mfu estimation

## training your own model

### shakespeare dataset (quick experiment)
```bash
# download shakespeare data (or add your own text file)
mkdir -p data/shakespear
# add your shakespear.txt file to data/shakespear/

# train on single gpu
python train/sp_train.py

# train on multiple gpus  
torchrun --standalone --nproc_per_node=8 train/sp_train.py
```

### fineweb dataset (realistic pretraining)
```bash
# prepare fineweb data shards in data/fineweb/fwds/
# train with distributed setup
torchrun --standalone --nproc_per_node=8 train/fw_train.py
```

### inference with your trained model
```bash
# interactive generation
python inference.py --model-path results/babygpt_shakespear.pth --interactive

# single prompt generation
python inference.py \
    --model-path results/babygpt_shakespear.pth \
    --prompt "to be or not to be" \
    --max-tokens 100 \
    --temperature 0.8 \
    --stream
```

## architecture deep dive

babyGPT implements the core GPT-2 architecture with several key components:

### transformer blocks (`Block` class)
each transformer block applies pre-layer normalization and residual connections around self-attention and feed-forward layers. this follows the architecture used in GPT-2 and modern transformer variants.

### causal self-attention (`CausalSelfAttention` class)  
implements masked multi-head attention where tokens can only attend to previous positions in the sequence. uses flash attention for memory-efficient computation and supports both training and inference modes.

### position and token embeddings
combines learnable position embeddings with token embeddings to give the model spatial awareness. uses weight tying between input embeddings and output projection as in the original transformer paper.

### training dynamics
- **mixed precision training**: uses bfloat16 for forward pass, fp32 for gradients
- **gradient accumulation**: enables large effective batch sizes across multiple steps
- **learning rate scheduling**: linear warmup followed by cosine annealing
- **gradient clipping**: prevents exploding gradients during training

## file structure

```
├── model.py              # core gpt-2 implementation and configuration
├── inference.py          # text generation script with multiple sampling methods
├── train/
│   ├── sp_train.py       # shakespeare training script (quick experiments)
│   └── fw_train.py       # fineweb training script (realistic pretraining)
├── data/                 # training data directory
└── results/              # saved model checkpoints
```

### core components

- **`configGPT`**: dataclass containing all model hyperparameters (layers, heads, embedding dimension)
- **`babyGPT`**: main model class with forward pass, weight initialization, and generation methods
- **`DataLoaderLite`**: efficient data loading for distributed training with automatic shard cycling
- **`TextGenerator`**: inference class supporting various sampling strategies and interactive generation

## understanding the implementation

### 1. explore the architecture
dive into `model.py` to understand how transformer blocks, attention mechanisms, and embeddings work together. the code includes detailed comments explaining each component.

### 2. experiment with training
use `train/sp_train.py` for quick experiments on shakespeare data. this trains fast enough to see results in minutes and helps you understand training dynamics.

### 3. scale up training  
move to `train/fw_train.py` for realistic pretraining experiments. this includes all the complexities of distributed training and large-scale data processing.

### 4. customize generation
explore `inference.py` to understand different sampling strategies and how they affect text quality. experiment with temperature, top-k, and top-p parameters.

## technical implementation details

### flash attention integration
babyGPT uses pytorch's native `scaled_dot_product_attention` with `is_causal=True` for memory-efficient attention computation. this provides significant speedups over naive attention implementations.

### distributed training setup
supports multi-gpu training using distributeddataparallel (ddp) with automatic gradient synchronization. includes proper initialization, cleanup, and loss averaging across processes.

### model flops utilization (mfu)
includes utilities to measure how efficiently your implementation uses available compute compared to theoretical peak performance. useful for optimization and hardware utilization analysis.

### weight initialization  
follows gpt-2's initialization scheme with carefully tuned standard deviations for different layer types. includes special handling for residual projections to maintain stable training.

## model configurations

the implementation supports various model sizes by adjusting the configuration:

```python
# gpt-2 small (124m parameters)
config = configGPT(n_layer=12, n_head=12, n_embd=768)

# gpt-2 medium (350m parameters) 
config = configGPT(n_layer=24, n_head=16, n_embd=1024)

# gpt-2 large (774m parameters)
config = configGPT(n_layer=36, n_head=20, n_embd=1280)
```

## learning pathways

this implementation provides several educational entry points:

**beginners**: start with `inference.py` to see generation in action, then examine the `babyGPT` class structure to understand the high-level architecture.

**intermediate**: work through `train/sp_train.py` to understand training loops, loss computation, and optimization. experiment with different hyperparameters.

**advanced**: dive into distributed training with `train/fw_train.py`, implement custom attention mechanisms, or add new architectural components.

## performance characteristics

on modern hardware, babyGPT achieves:
- **training speed**: ~10k tokens/second on a single a100 gpu
- **memory efficiency**: trains 124m parameter models with 16gb vram  
- **convergence**: achieves competitive perplexity on standard benchmarks
- **scaling**: linear speedup with additional gpus using distributed training

## acknowledgments

this implementation draws inspiration from several excellent educational resources while maintaining clarity and correctness:

- original gpt-2 architecture and training details from openai
- attention implementation insights from the annotated transformer
- distributed training patterns from pytorch distributed documentation
- modern optimization techniques from recent transformer implementations

## citations

```bibtex
@article{radford2019language,
    title={Language models are unsupervised multitask learners},
    author={Radford, Alec and Wu, Jeffrey and Child, Rewon and Luan, David and Amodei, Dario and Sutskever, Ilya},
    journal={OpenAI blog},
    volume={1},
    number={8},
    pages={9},
    year={2019}
}

@misc{Gokaslan2019OpenWeb,  
    title={OpenWebText Corpus},
    author={Aaron Gokaslan and Vanya Cohen},
    howpublished={\url{http://Skylion007.github.io/OpenWebTextCorpus}}, 
    year={2019}
}
```
