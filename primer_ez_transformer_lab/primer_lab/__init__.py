"""A CPU-friendly implementation of Primer-EZ components for education."""
from .model import (SquaredReLU, CausalDepthwiseConv1d, CausalSelfAttention,
                    TinyLanguageModel, ModelConfig, make_model, num_parameters)
