"""Optional backend comparison. Requires PyTorch, bitsandbytes and compatible CUDA."""
import torch
from bitsandbytes.nn import Linear8bitLt


@torch.inference_mode()
def main():
    if not torch.cuda.is_available():
        raise SystemExit("This optional check needs CUDA. Use lab.py for the CPU exercise.")
    torch.manual_seed(7)
    linear = torch.nn.Linear(128, 64, dtype=torch.float16)
    x = torch.randn(32, 128, dtype=torch.float16)
    x[::4, 3] = 24
    x[1::5, 91] = -12
    x = x.cuda()
    ref = linear.cuda()(x).float()
    linear.cpu()
    for threshold in (0., 6.):
        layer = Linear8bitLt(128, 64, bias=True, has_fp16_weights=False, threshold=threshold)
        layer.load_state_dict(linear.state_dict())
        layer = layer.eval().cuda()
        out = layer(x)
        assert out.shape == ref.shape and torch.isfinite(out).all()
        print(f"threshold={threshold:g} weights={layer.weight.dtype} "
              f"bias={layer.bias.device} RMSE={(out.float()-ref).square().mean().sqrt().item():.6g}")


if __name__ == "__main__":
    main()
