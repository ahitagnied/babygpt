import tiktoken
from model import *
import time, math

# initialize the gpt2 tokenizer
enc = tiktoken.get_encoding('gpt2')

# read shakespeare text from a file
with open('shakespear.txt', 'r') as f:
    text = f.read()

# take only the first 1000 characters of text for this example
text = text[:1000]
toks = enc.encode(text) # tokenize the text using the gpt2 tokenizer

# set batch size and sequence length
b, t = 4, 32 # b=batch size, t=timesteps (sequence length)

# create a tensor from the tokens, adding one extra token for the target shift
buffer = torch.tensor(toks[:b*t+1])
buffer = buffer.to(device)

# split into input (x) and target (y) tensors
# x contains tokens 0 to b*t-1, reshaped into b batches of t tokens each
x = buffer[:-1].view(b,t)
y = buffer[1:].view(b,t) # y contains tokens 1 to b*t, also reshaped (shifted by 1 position)

#-----------------------------------------------------------------------------------------
class DataLoaderLite:
    def __init__(self, b, t):
        self.b = b
        self.t = t
        # read the entire text file
        with open('shakespear.txt', 'r') as f:
            text = f.read()
        # tokenize it 
        enc = tiktoken.get_encoding('gpt2')
        toks = enc.encode(text)
        self.toks = torch.tensor(toks)
        print(f"loaded {len(self.toks)} tokens")
        # number of epochs = number of toks/ bt
        print(f"1 epoch -> {math.floor(len(self.toks) / (b*t))} batches")
        # state information
        self.current_position = 0
    
    def next_batch(self):
        b, t = self.b, self.t
        buffer = self.toks[self.current_position: self.current_position + b*t + 1]
        x, y = (buffer[:-1].view(b, t)), (buffer[1:].view(b, t)) # inputs and targets
        # next step
        self.current_position += b*t
        # if next batch is out of bounds, then reset:
        if self.current_position + b*t + 1 > len(self.toks):
            self.current_position = 0
        return x, y

#-----------------------------------------------------------------------------------------
print("using device:", device)

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

total_batch_size = 524288 # 2^19 or ~0.5M tokens
# set batch size and sequence length
b, t = 4, 1024 # b=batch size, t=timesteps (sequence length)
# create a tensor from the tokens, adding one extra token for the target shift
assert total_batch_size % (b*t) == 0, "total_batch_size must be divisible by b*t"
num_grad_accum = total_batch_size // (b*t) # number of gradient accumulation steps, 128 <- 524288/ (4*1024)
print(f"total batch size: {total_batch_size}")
print(f"gradient accumulation steps: {num_grad_accum}")

train_loader = DataLoaderLite(b=4, t=1024)

torch.set_float32_matmul_precision('high')

# get logits
model = babyGPT(configGPT(vocab_size=50304))
model.to(device)
model = torch.compile(model) # super ultra fast 

# learning rate parameters
max_lr = 6e-4
min_lr = 3e-5
n_warmup = 10
n_steps = 50

# optimize:
optimizer = model.config_optimizer(weight_decay=0.1, lr=3e-4, betas=(0.9, 0.95), device=device)

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
        # with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss = model(x, y)
        loss /= num_grad_accum # scale the loss by the number of gradient accumulation steps
        lossf += loss.detach()
        loss.backward()
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
    toks = train_loader.b * train_loader.t * num_grad_accum
    toksps = toks / s
    print(f"step: {step:02d} | lr: {lr:.10f} | loss: {lossf:.10f} | norm: {norm:.4f} | time: {s*1000:.2f} ms | toks/s: {toksps:.2f}")

# with torch.profiler.profile(
#     activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
#     record_shapes=True,
#     profile_memory=True,
#     with_stack=True
# ) as prof:
# print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))