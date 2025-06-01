import tiktoken
from model import *
import time, math, os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# read shakespeare text from a file
with open('./data/shakespear/shakespear.txt', 'r') as f:
    text = f.read()

# initialize the gpt2 tokenizer
enc = tiktoken.get_encoding('gpt2')

# take only the first 1000 characters of text for this example
toks = torch.tensor(enc.encode(text), dtype=torch.long) # tokenize the text using the gpt2 tokenizer

#-----------------------------------------------------------------------------------------
class DataLoaderLite:
    def __init__(self, b, t, rank, nump):
        self.b = b
        self.t = t
        self.rank = rank
        self.nump = nump
        self.toks = toks.to(device)
        self.device = device
        self.current_position = self.rank * self.b * self.t # start at the beginning of the data for this rank
        print(f"[rank {rank}] loaded {self.toks.size(0)} tokens on {device}")
    
    def next_batch(self):
        b, t = self.b, self.t
        buffer = self.toks[self.current_position: self.current_position + b*t + 1]
        x, y = (buffer[:-1].view(b, t)), (buffer[1:].view(b, t)) # inputs and targets
        # advance by world size
        self.current_position += b * t * self.nump
        # if next batch is out of bounds, then reset:
        if self.current_position + b * t * self.nump + 1 > self.toks.size(0):
            self.current_position = self.rank * self.b * self.t
        return x, y

#-----------------------------------------------------------------------------------------

from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
# need to run using torch run -> torchrun --standalone --nproc_per_node=4 train.py

# set up distributed data parallel (ddp) for multi-gpu training
ddp = int(os.environ.get('RANK', -1)) != -1 # check if we're in distributed training mode

if ddp:  # if we're in distributed training mode
    init_process_group(backend='nccl')  # initialize the process group using nccl backend (optimized for nvidia gpus)
    
    ddp_rank = int(os.environ.get('RANK'))  # global rank of this process across all nodes
    ddp_local_rank = int(os.environ.get('LOCAL_RANK'))  # local rank (which gpu to use)
    ddp_world_size = int(os.environ.get('WORLD_SIZE'))  # total number of processes/gpus in the training
    
    device = torch.device(f'cuda:{ddp_local_rank}') # assign this process to the appropriate gpu
    torch.cuda.set_device(device)  # set the cuda device for this process
    
    master_process = ddp_rank == 0  # only rank 0 is the master process
else:
    # vanilla training (single gpu or cpu)
    ddp_rank = 0  # no rank in non-distributed mode
    ddp_local_rank = 0  # no local rank in non-distributed mode
    ddp_world_size = 1  # only one process in non-distributed mode
    master_process = True  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

total_batch_size = 524288 # 2^19
# set batch size and sequence length
b, t = 16, 1024 # b=batch size, t=timesteps (sequence length)
# create a tensor from the tokens, adding one extra token for the target shift
assert total_batch_size % (b*t*ddp_world_size) == 0, "total_batch_size must be divisible by b*t*ddp_world_size"
num_grad_accum = total_batch_size // (b*t*ddp_world_size) 
if master_process: # only print once, for the master process 
    print(f"total batch size: {total_batch_size}")
    print(f"gradient accumulation steps: {num_grad_accum} \n")

#-----------------------------------------------------------------------------------------

train_loader = DataLoaderLite(b=8, t=512, rank=ddp_rank, nump=ddp_world_size) 

torch.set_float32_matmul_precision('high')

# initialise model 
model = babyGPT(configGPT(vocab_size=50304))
model.to(device)

# learning rate parameters
max_lr = 6e-4
min_lr = max_lr * 0.1
n_warmup = 10
n_steps = 50

# optimize:
optimizer = model.config_optimizer(weight_decay=0.1, lr=3e-4, betas=(0.9, 0.95), device=device)
model = torch.compile(model, backend="eager") # super ultra fast 

if ddp:
    from torch.nn.parallel import DistributedDataParallel as DDP
    # wrap the model with DDP
    model = DDP(model, device_ids=[ddp_local_rank])
    
# initialise the scheduler 
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=(n_steps - n_warmup),
    eta_min=min_lr
)

for step in range(n_steps):
    t0 = time.time()
    optimizer.zero_grad()
    lossf = 0.0
    for babystep in range(num_grad_accum):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits, loss = model(x, y)
            loss /= num_grad_accum # scale the loss by the number of gradient accumulation steps
            lossf += loss.detach()

        if ddp: 
            model.require_backward_grad_sync = (babystep == num_grad_accum - 1)
        loss.backward()

    if ddp: 
        dist.all_reduce(lossf, op=dist.ReduceOp.AVG) # sum the loss across all processes
    norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # get next lr using schedule 
    # handle learning rate
    if step < n_warmup:
        # linear warmup
        lr = max_lr * (step + 1) / n_warmup
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        # use cosine scheduler for steps after warmup
        scheduler.step()
        # get the current learning rate from the optimizer
        lr = optimizer.param_groups[0]['lr']
    optimizer.step()
    torch.cuda.synchronize() # wait till gpu is done
    t1 = time.time()
    s = (t1-t0) # in s
    toks = train_loader.b * train_loader.t * num_grad_accum * ddp_world_size
    toksps = toks / s
    if master_process: # only print once, for the master process
        print(f"step: {step:02d} | lr: {lr:.10f} | loss: {lossf:.10f} | norm: {norm:.4f} | time: {s*1000:.2f} ms | toks/s: {toksps:.2f}")

if ddp: 
    destroy_process_group()  # clean up the process group
    print("destroyed process group")

# save the model
if master_process: # only save once, for the master process
    os.makedirs('results', exist_ok=True)  # create results directory if it doesn't exist
    torch.save(model.state_dict(), 'results/babygpt_shakespear.pth')
    print("saved model")

import sys; sys.exit(0) # exit the script
#-----------------------------------------------------------------------------------------