"""Synthetic sequential data. No datasets or external downloads required."""
import torch


def periodic_sequences(batch: int, context: int, period: int, vocab_size: int,
                       generator: torch.Generator, noise: float = 0.025) -> torch.Tensor:
    """Return [batch, context + 1] for next-token prediction.

    Each independent sequence begins with a random `period`-length motif.
    Later steps copy token[t-period], with optional random substitutions.
    No tokens from test sequences are used for training.
    """
    if batch <= 0 or period < 1 or period >= context or not (0.0 <= noise <= 1.0):
        raise ValueError('invalid batch, period, context, or noise')
    if vocab_size < 3:
        raise ValueError('vocab_size must be >=3')
    result = torch.empty((batch, context + 1), dtype=torch.long)
    result[:, :period] = torch.randint(vocab_size, (batch, period), generator=generator)
    for index in range(period, context + 1):
        override = torch.rand(batch, generator=generator) < noise
        replacement = torch.randint(vocab_size, (batch,), generator=generator)
        result[:, index] = torch.where(override, replacement, result[:, index - period])
    return result


def teacher_forcing(batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return batch[:, :-1], batch[:, 1:]
