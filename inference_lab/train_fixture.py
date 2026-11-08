"""Rebuild the bundled teaching checkpoint on CPU from a tiny, synthetic grammar."""
from pathlib import Path
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

def main():
    torch.manual_seed(17)
    torch.set_num_threads(1)
    words = "<unk> <end> sensor reports hot cold . fan on off".split()
    ids = {word: index for index, word in enumerate(words)}
    backend = Tokenizer(WordLevel(ids, unk_token="<unk>"))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>", eos_token="<end>")
    config = GPT2Config(vocab_size=len(words), n_positions=1024, n_embd=64, n_layer=2, n_head=4,
                        bos_token_id=None, eos_token_id=ids["<end>"], resid_pdrop=0, embd_pdrop=0, attn_pdrop=0)
    model = GPT2LMHeadModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002)
    sentences = ["sensor reports hot . fan on . <end>", "sensor reports cold . fan off . <end>"]
    batch = torch.tensor([tokenizer.encode(text) for text in sentences])
    for _ in range(100):
        optimizer.zero_grad()
        loss = model(batch, labels=batch, use_cache=False).loss
        loss.backward()
        optimizer.step()
    target = Path(__file__).resolve().parent / "fixture"
    model.save_pretrained(target)
    tokenizer.save_pretrained(target)
    print(f"Saved {target.name}; final training loss {loss.item():.4f}")

if __name__ == "__main__":
    main()
