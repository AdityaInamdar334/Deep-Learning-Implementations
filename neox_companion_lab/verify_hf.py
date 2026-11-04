"""Offline parity against a randomly initialized Transformers NeoX."""
import torch
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
from model import Config, TinyNeoX

def main():
    torch.set_num_threads(1)
    torch.manual_seed(19)
    c=Config(hidden_size=32,num_heads=4,num_layers=2,intermediate_size=64,max_seq_len=64)
    hc=GPTNeoXConfig(vocab_size=256,hidden_size=32,num_hidden_layers=2,
        num_attention_heads=4,intermediate_size=64,rotary_pct=0.25,
        rotary_emb_base=10000,max_position_embeddings=64,hidden_act='gelu',
        layer_norm_eps=1e-5,attention_dropout=0.0,hidden_dropout=0.0,
        use_parallel_residual=True,tie_word_embeddings=False)
    hc._attn_implementation='eager'
    reference=GPTNeoXForCausalLM(hc).eval()
    ours=TinyNeoX(c).eval()
    mapping={'embedding.weight':'gpt_neox.embed_in.weight','head.weight':'embed_out.weight',
        'final_norm.weight':'gpt_neox.final_layer_norm.weight',
        'final_norm.bias':'gpt_neox.final_layer_norm.bias'}
    modules={'attn_norm':'input_layernorm','mlp_norm':'post_attention_layernorm',
        'attn.qkv':'attention.query_key_value','attn.out':'attention.dense',
        'mlp.0':'mlp.dense_h_to_4h','mlp.2':'mlp.dense_4h_to_h'}
    for i in range(c.num_layers):
        for dest,src in modules.items():
            for suffix in ('weight','bias'):
                mapping[f'layers.{i}.{dest}.{suffix}']=f'gpt_neox.layers.{i}.{src}.{suffix}'
    state=reference.state_dict()
    ours.load_state_dict({dest:state[src] for dest,src in mapping.items()},strict=True)
    ids=torch.randint(0,256,(2,13))
    with torch.no_grad():
        expected=reference(ids,use_cache=False).logits
        actual,_=ours(ids)
    torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-5)
    print('Random-config HF parity passed. Max absolute error:',(actual-expected).abs().max().item())
    print('This does not verify loading a published 20B or Pythia checkpoint.')

if __name__=='__main__':
    main()
