"""Reversible hooks, activation probes, and a *toy* projection edit."""
from contextlib import contextmanager
import torch
import torch.nn.functional as F


def collect_block_activations(model, tokens):
    captures = []
    handles = []
    for layer in model.layers:
        def hook(module, inputs, output):
            captures.append(output.detach().clone())
        handles.append(layer.register_forward_hook(hook))
    try:
        with torch.no_grad():
            model(tokens)
    finally:
        for h in handles:
            h.remove()
    return torch.stack(captures)  # layers, samples, hidden


def collect_neurons(model, layer_id, tokens):
    buf = []
    h = model.layers[layer_id].mlp.tap.register_forward_hook(
        lambda module, inputs, output: buf.append(output.detach().clone())
    )
    try:
        with torch.no_grad():
            model(tokens)
    finally:
        h.remove()
    return buf[0]


def contrast_direction(model, pos_tokens, neg_tokens, layer_id):
    p = collect_block_activations(model, pos_tokens)[layer_id].mean(0)
    n = collect_block_activations(model, neg_tokens)[layer_id].mean(0)
    direction = p - n
    return F.normalize(direction, dim=0)


@contextmanager
def steer_residual(model, layer_id, direction, strength):
    """Temporary forward hook, installed only inside a with statement."""
    def hook(module, inputs, output):
        return output + float(strength) * direction.to(output.device, output.dtype)
    h = model.layers[layer_id].register_forward_hook(hook)
    try:
        yield
    finally:
        h.remove()


@contextmanager
def steer_neuron(model, layer_id, neuron_id, delta):
    """Modify a single SwiGLU gated intermediate feature, not a residual coordinate."""
    tap = model.layers[layer_id].mlp.tap
    def hook(module, inputs, output):
        shifted = output.clone()
        shifted[..., neuron_id] += float(delta)
        return shifted
    h = tap.register_forward_hook(hook)
    try:
        yield
    finally:
        h.remove()


def project_output_weights(model, layer_id, direction):
    """Edit a copied model's MLP DOWN PROJECTION, not the residual stream.

    Demonstrates one-sided projection; does not remove a feature globally.
    """
    import copy
    model = copy.deepcopy(model)
    d = F.normalize(direction.detach(), dim=0)
    down = model.layers[layer_id].mlp.down
    with torch.no_grad():
        down.weight.sub_(torch.outer(d, d @ down.weight))
        # bias is also an output contribution and must be projected
        down.bias.sub_(d * (d @ down.bias))
    model.eval()
    return model


def categorical_kl(reference_logits, changed_logits):
    return (F.softmax(reference_logits, -1) * (
        F.log_softmax(reference_logits, -1) - F.log_softmax(changed_logits, -1)
    )).sum(-1).mean().item()
