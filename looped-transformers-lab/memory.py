"""Analytical estimates only; not architecture-specific measurements."""
import argparse, json

def kv_bytes(context, global_layers, local_layers, window, kv_heads, head_dim, bytes_per_element):
    return 2 * kv_heads * head_dim * bytes_per_element * (global_layers * context + local_layers * min(context, window))

def weight_bytes(parameters, bits):
    return parameters * bits / 8
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--context', type=int, default=131072)
    p.add_argument('--global-layers', type=int, default=8)
    p.add_argument('--local-layers', type=int, default=40)
    p.add_argument('--window', type=int, default=1024)
    p.add_argument('--kv-heads', type=int, default=4)
    p.add_argument('--head-dim', type=int, default=128)
    p.add_argument('--bytes-per-element', type=int, default=2)
    p.add_argument('--parameters', type=float, default=26400000000.0)
    p.add_argument('--bits', type=int, default=4)
    a = p.parse_args()
    print(json.dumps({'hypothetical_settings': vars(a), 'kv_GiB': kv_bytes(a.context, a.global_layers, a.local_layers, a.window, a.kv_heads, a.head_dim, a.bytes_per_element) / 2 ** 30, 'raw_weight_GiB': weight_bytes(a.parameters, a.bits) / 2 ** 30}, indent=2))
