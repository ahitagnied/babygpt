from model import *
import tiktoken

num_return_sequences = 5 # number of different sequences to generate
max_length = 30 # maximum length of each sequence (in tokens)

model = babyGPT.from_pretrained('gpt2') # load the pretrained gpt2 model
model.eval() # set the model to evaluation mode (disables dropout)

# initialize the gpt2 tokenizer
enc = tiktoken.get_encoding('gpt2') 
# encode the prompt text into tokens
token = enc.encode("i'm lebron james from akron, ohio")
# convert tokens to a pytorch tensor
token = torch.tensor(token, dtype=torch.long)
# add batch dimension and repeat the same prompt for all sequences
token = token.unsqueeze(0).repeat(num_return_sequences, 1)
# move tensor to the appropriate device (gpu or cpu)
x = token.to(device)

torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    with torch.no_grad():
        logits, _ = model(x) # get model predictions (logits and loss, but we only use logits)
        logits = logits[:, -1, :] # get logits for just the last token in each sequence
        probs = F.softmax(logits, dim=-1) # convert logits to probabilities
        topkprobs, topkindices = torch.topk(probs, 50, dim=-1) # get top 50 most likely tokens for each sequence
        # randomly sample from the top tokens based on their probabilities
        ix = torch.multinomial(topkprobs, 1)
        # get the actual token ids that were sampled
        xcol = torch.gather(topkindices, -1, ix)
        # append the new tokens to our sequences
        x = torch.cat((x, xcol), dim=1)

for i in range(num_return_sequences):
    # get the tokens for this sequence
    tokens = x[i, :max_length].tolist()
    # tokens -> text
    decoded = enc.decode(tokens)
    print(">", decoded)