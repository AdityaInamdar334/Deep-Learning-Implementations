"""An original, deliberately narrow procedure cache. Python 3.10+, standard library."""
import argparse
from dataclasses import dataclass
import re

@dataclass(frozen=True)
class Request:
    examples: int
    capacity: int
    keep_partial: bool

# Full matches make unsupported qualifiers observable as misses.
TEMPLATES = (
    r"batch (?P<n>[0-9]{1,12}) examples at (?P<b>[0-9]{1,12}) per batch, (?P<policy>keep|drop) the last partial batch",
    r"with batches of (?P<b>[0-9]{1,12}), group (?P<n>[0-9]{1,12}) examples and (?P<policy>keep|drop) the remainder",
)

def extract(text):
    normalized = " ".join(text.lower().split()).removesuffix(".")
    for template in TEMPLATES:
        fields = re.fullmatch(template, normalized)
        if fields and int(fields["b"]) > 0:
            return Request(int(fields["n"]), int(fields["b"]), fields["policy"] == "keep")
    return None

def execute(request):
    if request.examples < 0 or request.capacity <= 0:
        raise ValueError("Counts require nonnegative examples and positive capacity.")
    full, remainder = divmod(request.examples, request.capacity)
    partial = int(bool(remainder) and request.keep_partial)
    return {"full":full, "remainder":remainder, "partial_batches_used":partial,
            "batches":full + partial}

def answer(text):
    request = extract(text)
    return execute(request) if request is not None else None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    result = answer(parser.parse_args().query)
    print(result if result is not None else "MISS: hand off to the source model")
