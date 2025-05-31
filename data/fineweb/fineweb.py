"""
load a large ds from hf 
--> tokenize it using gpt-2 bpe 
--> uint16 toks 
--> fixed size 100 million toks 'shards' or .npy files
"""

import os
import numpy as np
import tiktoken
from tqdm import tqdm
import multiprocessing as mp
from datasets import load_dataset

local_data_path = "fwds"
remote_name = "sample-10BT"
shard_size = int(1e8) # 100 million tokens per shard

DATA_CACHE_PATH = os.path.join(os.path.dirname(__file__), local_data_path)
os.makedirs(DATA_CACHE_PATH, exist_ok=True)

# get the dataset from hf
fw = load_dataset("HuggingFaceFW/fineweb", name=remote_name, split="train")

# initialize the tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens["<|endoftext|>"]
def tokenize(data):
    """
    tokenize the text data and return a list of tokens.
    """
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(data["text"])) # skips special tokens/ eot
    toks_np = np.array(tokens)
    assert (0 <= toks_np).all() and (toks_np < 2**16).all(), "tokens must be in the range [0, 2**16]"
    uint16_toks = toks_np.astype(np.uint16)  # convert to uint16
    return uint16_toks

def write_shard(f, data):
    """
    write the tokenized data to a file.
    """
    np.save(f, data)

# tokenize the dataset and write to shards
nprocs = max(1, os.cpu_count()//2)  # number of processes to use for tokenization
with mp.Pool(nprocs) as pool:
    sid = 0 # track which shard we're writing to
    all_toks = np.empty((shard_size,), dtype=np.uint16)  # buffer for tokens
    num_toks = 0  # number of tokens written to the current shard
    progress_bar = None
    for toks in pool.imap(tokenize, fw, chunksize=16):
        if num_toks + len(toks) < shard_size:
            # write the current shard to a file
            all_toks[num_toks:num_toks+len(toks)] = toks
            num_toks += len(toks)
            if progress_bar is None: 
                progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"shard {sid}")
            progress_bar.update(len(toks))
        else: 
            split = "val" if sid == 0 else "train"
            f = os.path.join(DATA_CACHE_PATH, f"fwds_{split}_{sid:06d}")
            remainder = shard_size - num_toks
            progress_bar.update(remainder)
            all_toks[num_toks:num_toks+remainder] = toks[:remainder]
            write_shard(f, all_toks)
            sid += 1
            progress_bar = None
            all_toks[0:len(toks)-remainder] = toks[remainder:]  # move the remaining tokens to the front
            num_toks = len(toks) - remainder  # update the number of tokens written to the current shard

    if num_toks != 0:
        # write the last shard to a file
        split = "val" if sid == 0 else "train"
        f = os.path.join(DATA_CACHE_PATH, f"fwds_{split}_{sid:06d}")
        write_shard(f, all_toks[:num_toks])
        print(f"wrote {num_toks} tokens to {f}")