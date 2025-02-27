import tiktoken
from model import *

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

# initialize the babygpt model with the default configuration
model = babyGPT(configGPT())
model.to(device)

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
        print(f"1 epoch -> {len(self.toks) / (b*t)} batches")
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

train_loader = DataLoaderLite(b=4, t=32)

# get logits
model = babyGPT(configGPT())
model.to(device)

# optimize:
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(30):
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    print(f"step: {i:02d} | loss: {loss.item():.10f}")
    hello = 1