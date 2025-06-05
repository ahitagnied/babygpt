import tiktoken
from model import *
import time, math, os
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_toks(f):
    npt = np.load(f)
    ptt = torch.tensor(npt, dtype=torch.long)  # load the tokens from the file
    return ptt
#-----------------------------------------------------------------------------------------
class DataLoaderLite:
    def __init__(self, b, t, rank, nump, split):
        self.b = b
        self.t = t
        self.rank = rank
        self.nump = nump
        assert split in ['train', 'val'], "split must be either 'train' or 'val'"

        # get the shard file names
        data_dir = './data/fineweb/fwds'
        shards = os.listdir(data_dir)
        shards = [s for s in shards if split in s]  # filter by split
        shards = sorted(shards)  # sort the shards by name
        shards = [os.path.join(data_dir, s) for s in shards]  # get the full path to the shards
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split '{split}' in {data_dir}"
        if master_process:
            print(f"[rank {rank}] found {len(shards)} shards for split '{split}' in {data_dir}")
        
        self.current_shard = 0  # start with the first shard
        self.toks = load_toks(self.shards[self.current_shard])  # load the tokens from the first shard
        self.current_position = self.rank * self.b * self.t  # start at the beginning of the data for this rank

    def next_batch(self):
        b, t = self.b, self.t
        buffer = self.toks[self.current_position: self.current_position + b*t + 1]
        x, y = (buffer[:-1].view(b, t)), (buffer[1:].view(b, t)) # inputs and targets
        # advance by world size
        self.current_position += b * t * self.nump
        # if next batch is out of bounds, then go to the next shard:
        if self.current_position + (b * t * self.nump + 1) > len(self.toks):
            self.current_shard = (self.current_shard + 1) % len(self.shards)  # go to the next shard
            self.toks = load_toks(self.shards[self.current_shard])  # load the tokens from the next shard
            self.current_position = self.rank * self.b * self.t
        return x, y
    
    def reset(self):
        self.current_shard = 0
        self.toks = load_toks(self.shards[self.current_shard])
        self.current_position = self.b * self.t * self.rank

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
b, t = 32, 1024 # b=batch size, t=timesteps (sequence length)
# create a tensor from the tokens, adding one extra token for the target shift
assert total_batch_size % (b*t*ddp_world_size) == 0, "total_batch_size must be divisible by b*t*ddp_world_size"
num_grad_accum = total_batch_size // (b*t*ddp_world_size) 
if master_process: # only print once, for the master process 
    print(f"total batch size: {total_batch_size}")
    print(f"gradient accumulation steps: {num_grad_accum} \n")

#-----------------------------------------------------------------------------------------

train_loader = DataLoaderLite(b=8, t=512, rank=ddp_rank, nump=ddp_world_size, split='train') 
val_loader = DataLoaderLite(b=8, t=512, rank=ddp_rank, nump=ddp_world_size, split='val') 

torch.set_float32_matmul_precision('high')

# initialise model 
model = babyGPT(configGPT(vocab_size=50304))
model.to(device)

# learning rate parameters
max_lr = 6e-4
min_lr = max_lr * 0.1
n_warmup = 715 # warm up lr over 378e6 toks -> 378e6/ 2^19 = 715 steps from gpt3 paper 
n_steps = 19073 # 2^19 toks/step -> 10^9 toks to do -> 10^9/2^19 = 19073 steps

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
    # evaluate loss every 100 steps
    if step % 100 == 0:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_total = 0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss/val_loss_steps
                val_loss_total += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_total, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"val loss: {val_loss_total.item():.4f}")

    # train loop
    model.train()
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
    torch.save(model.state_dict(), 'results/babygpt_fw.pth')
    print("saved model")

import sys; sys.exit(0) # exit the script
#-----------------------------------------------------------------------------------------