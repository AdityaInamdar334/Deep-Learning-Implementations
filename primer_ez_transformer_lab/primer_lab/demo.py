"""Demonstrate the causal properties of spatial depthwise convolution."""
import torch
from .model import CausalDepthwiseConv1d, make_model


def main():
    torch.manual_seed(0)
    conv = CausalDepthwiseConv1d(heads=2, head_dim=4, kernel_size=3)
    a = torch.randn(1, 2, 8, 4)
    b = a.clone()
    b[:, :, 5:, :] = torch.randn_like(b[:, :, 5:, :]) * 100
    da = conv(a)
    db = conv(b)
    print('Causal convolution max prefix difference:', float((da[:, :, :5] - db[:, :, :5]).abs().max().detach()))
    base = make_model('baseline_relu', context=12)
    primer = make_model('primer_ez', context=12)
    tokens = torch.randint(0, 16, (1, 12))
    disturbed = tokens.clone()
    disturbed[:, 6:] = (disturbed[:, 6:] + 3) % 16
    for name, m in [('baseline_relu', base), ('primer_ez', primer)]:
        m.eval()
        with torch.no_grad():
            difference = (m(tokens)[:, :6] - m(disturbed)[:, :6]).abs().max().item()
        print(f'{name} max prefix-logit difference:', difference)


if __name__ == '__main__':
    main()
