"""
text generation script for baby gpt: 
loads a trained model and generates text continuation given a prompt
"""

import torch
import tiktoken
import argparse
import os
from model import babyGPT, configGPT

class Colors:
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RESET = '\033[0m'

class TextGenerator:
    def __init__(self, model_path=None, device=None):
        """
        initialize the text generator
        
        args:
            model_path (str): path to saved model checkpoint (e.g., 'results/babygpt_fw.pth')
            device (str): device to run on ('cuda', 'cpu', 'auto')
        """
        # set device
        if device is None or device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        print(f"{Colors.CYAN} using device: {Colors.BOLD}{self.device}{Colors.RESET}\n")
        # initialise tokenizer
        self.tokenizer = tiktoken.get_encoding('gpt2')
        # load model
        if model_path and os.path.exists(model_path):
            self.model = self._load_checkpoint(model_path)
            print(f"{Colors.GREEN}✅ loaded model from: {Colors.BOLD}{model_path}{Colors.RESET}\n")
        else:
            raise FileNotFoundError(f"checkpoint not found at {model_path}")
        self.model.to(self.device)
        self.model.eval()
    
    def _load_checkpoint(self, checkpoint_path):
        """load model from saved checkpoint"""
        # try to load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # handle different checkpoint formats
        if 'model' in checkpoint:
            # full checkpoint with config
            if 'config' in checkpoint:
                config = checkpoint['config']
                model = babyGPT(config)
            else:
                # default config if not saved
                config = configGPT(vocab_size=50304)
                model = babyGPT(config)
            
            state_dict = checkpoint['model']
            
            if 'step' in checkpoint:
                print(f"{Colors.BLUE}📈 checkpoint from step: {Colors.BOLD}{checkpoint['step']}{Colors.RESET}")
            if 'val_loss' in checkpoint:
                print(f"{Colors.BLUE}📊 val loss: {Colors.BOLD}{checkpoint['val_loss']:.4f}{Colors.RESET}")
        else:
            # direct state dict
            config = configGPT(vocab_size=50304)
            model = babyGPT(config)
            state_dict = checkpoint
        
        # handle ddp and torch.compile prefixes
        # remove 'module.' prefix (from ddp) and '_orig_mod.' prefix (from torch.compile)
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            # remove prefixes in order: module._orig_mod. -> _orig_mod. -> clean key
            clean_key = key
            if clean_key.startswith('module.'):
                clean_key = clean_key[7:]  # remove 'module.'
            if clean_key.startswith('_orig_mod.'):
                clean_key = clean_key[10:]  # remove '_orig_mod.'
            cleaned_state_dict[clean_key] = value
        
        model.load_state_dict(cleaned_state_dict)
        return model
    
    def generate(self, prompt, max_new_tokens=50, temperature=1.0, top_k=50, top_p=None, seed=None, stream=False):        
        """
        generate text continuation given a prompt

        args:
            prompt (str): input text prompt
            max_new_tokens (int): max number of new tokens to generate
            temperature (float): sampling temperature (higher = more random)
            top_k (int): top-k sampling (None to disable)
            top_p (float): top-p (nucleus) sampling (None to disable)
            seed (int): random seed for reproducible generation
            stream (bool): whether to show tokens as they're generated  
            
        returns:
            str: generated text (prompt + continuation)
        """
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        # encode the prompt
        tokens = self.tokenizer.encode(prompt)
        tokens = torch.tensor(tokens, dtype=torch.long, device=self.device)
        tokens = tokens.unsqueeze(0)  # + batch dimension
        print(f"{Colors.MAGENTA}💭 prompt:{Colors.RESET} {Colors.DIM}'{prompt}'{Colors.RESET}\n")
        print(f"{Colors.YELLOW}⚡ generating {max_new_tokens} tokens...{Colors.RESET}\n")
        print(f"{Colors.CYAN}{'═' * 60}{Colors.RESET}")
        # generate tokens
        with torch.no_grad():
            for i in range(max_new_tokens):
                # get model predictions
                logits, _ = self.model(tokens)
                logits = logits[:, -1, :]  # get logits for the last position
                # apply temperature
                if temperature != 1.0:
                    logits = logits / temperature
                # apply top-k filtering
                if top_k is not None:
                    # get top-k logits and indices
                    top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
                    # mask out logits below top-k
                    logits_filtered = torch.full_like(logits, float('-inf'))
                    logits_filtered.scatter_(1, top_k_indices, top_k_logits)
                    logits = logits_filtered
                # apply top-p (nucleus) filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    # remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # shift the indices to the right to keep also the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')
                # convert to probabilities and sample
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                # append to sequence
                tokens = torch.cat([tokens, next_token], dim=1)
                # optional: print token as it's generated
                if stream:
                    new_text = self.tokenizer.decode([next_token.item()])
                    print(new_text, end='', flush=True)
        if stream:
            print(f"\n{Colors.CYAN}{'═' * 60}{Colors.RESET}")
        # decode the full sequence
        generated_tokens = tokens[0].tolist()
        generated_text = self.tokenizer.decode(generated_tokens)
        return generated_text
    
    def interactive_mode(self):
        """interactive text generation mode"""
        print("\n=== interactive text generation ===")
        print("type 'quit' to exit, 'help' for commands")
        while True:
            try:
                prompt = input("\nenter prompt: ").strip()
                if prompt.lower() == 'quit':
                    break
                elif prompt.lower() == 'help':
                    self._print_help()
                    continue
                elif not prompt:
                    continue
                # get generation parameters
                try:
                    max_tokens = int(input("max new tokens (default 50): ") or "50")
                    temperature = float(input("temp (default 1.0): ") or "1.0")
                    top_k = input("top-k (default 50): ") or "50"
                    top_k = int(top_k) if top_k.lower() != 'none' else None
                except ValueError:
                    print("invalid input, using defaults...")
                    max_tokens, temperature, top_k = 50, 1.0, 50
                # generate text
                generated = self.generate(
                    prompt=prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    seed=42,  # for reproducible results
                    stream=True  # show tokens as they are generated
                )
            except KeyboardInterrupt:
                print("\nexiting...")
                break
            except Exception as e:
                print(f"error: {e}")
    
    def _print_help(self):
        """print help information"""
        print("""
            available commands:
            - enter any text prompt to generate continuation
            - 'quit' - exit the program  
            - 'help' - show this help message

            params:
            - max new tokens: # of tokens to generate (default: 50)
            - temp: controls randomness (0.1=deterministic, 2.0=very random)  
            - top-k: keep only top k tokens for sampling (higher=more diverse)
        """)

def main():
    parser = argparse.ArgumentParser(description='generate text using trained babyGPT model')
    parser.add_argument('--model-path', type=str, help='path to saved model checkpoint')
    parser.add_argument('--prompt', type=str, help='text prompt for generation')
    parser.add_argument('--max-tokens', type=int, default=50, help='max new tokens to generate')
    parser.add_argument('--temperature', type=float, default=1.0, help='sampling temperature')
    parser.add_argument('--top-k', type=int, default=50, help='top-k sampling')
    parser.add_argument('--seed', type=int, help='random seed for reproducibility')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu', 'auto'], default='auto',
                       help='device to run on')
    parser.add_argument('--interactive', action='store_true', help='run in interactive mode')
    parser.add_argument('--stream', action='store_true', help='show tokens as they are generated')
    args = parser.parse_args()
    # initialise generator
    generator = TextGenerator(
        model_path=args.model_path,
        device=args.device
    )
    if args.interactive:
        generator.interactive_mode()
    elif args.prompt:
        # generate text for provided prompt
        generated_text = generator.generate(
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            seed=args.seed,
            stream=args.stream
        )
        if args.stream:
            print(f"\n{Colors.GREEN}📝 complete generated text:{Colors.RESET}")
            print(f"{Colors.DIM}[PROMPT]{Colors.RESET} {Colors.WHITE}{args.prompt}{Colors.RESET}")
            print(f"{Colors.DIM}[GENERATED]{Colors.RESET} {Colors.YELLOW}{generated_text[len(args.prompt):]}{Colors.RESET}")
            print(f"\n{Colors.MAGENTA}🔗 full text:{Colors.RESET}")
            print(f"{Colors.WHITE}{generated_text}{Colors.RESET}")
        else:
            print(f"\n{Colors.GREEN}📝 generated text:{Colors.RESET}")
            print(f"{Colors.WHITE}{generated_text}{Colors.RESET}")
    else:
        print("no prompt provided. Use --prompt or --interactive mode.")
        print("use --help for more options.")


if __name__ == "__main__":
    main()