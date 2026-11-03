"""Educational weight-space programming experiments, not an LLM model editor."""
from .core import TinyPolicy, make_policy, permute_hidden, rescale_hidden, compile_xor
from .editing import propose_edit, diff_audit
__all__ = ["TinyPolicy", "make_policy", "permute_hidden", "rescale_hidden", "compile_xor", "propose_edit", "diff_audit"]
