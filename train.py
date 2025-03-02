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

train_loader = DataLoaderLite(b=4, t=1024)

# torch.set_float32_matmul_precision('high')

# get logits
model = babyGPT(configGPT(vocab_size=50304))
model.to(device)
model = torch.compile(model) # super ultra fast 

maxlr = 3e-4
minlr = 3e-5
nwarmup = 10
nsteps = 50
def lrf(step):
    # 1: linear warm up 
    if step < nwarmup: 
        return maxlr * (step+1) / nwarmup
    # 2: if steps > nsteps, return minlr
    if step > nsteps:
        return minlr
    # 3: otherwise use cosine decay lr scheduler
    decay_ratio = (step-nwarmup)/(nsteps-nwarmup)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return minlr + coeff * (maxlr - minlr)

# optimize:
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8) # optimisation hyperparams from GPT-3 paper
for step in range(nsteps):
    t0 = time.time()
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    # with torch.autocast(device_type=device, dtype=torch.bfloat16):
    logits, loss = model(x, y)
    loss.backward()
    norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # get next lr using schedule 
    lr = lrf(step)
    for param_grp in optimizer.param_groups: # update lr for each parameter group
        param_grp['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize() # wait till gpu is done
    t1 = time.time()
    ms = (t1-t0)*1000 # in ms
    toksps = (train_loader.b * train_loader.t) / (t1 - t0)
    print(f"step: {step:02d} | loss: {loss.item():.10f} | norm: {norm:.4f} | time: {ms:.2f} ms | toks/s: {toksps:.2f}")

# with torch.profiler.profile(
#     activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
#     record_shapes=True,
#     profile_memory=True,
#     with_stack=True
# ) as prof:
# print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))