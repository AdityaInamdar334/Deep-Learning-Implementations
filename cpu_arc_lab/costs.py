"""Table 2 v1, published rounded USD costs and published invocation ratios."""
from decimal import Decimal as D
from math import ceil
ROWS = [
    ("MATH-500","0.0388","1.1818","0.0004",88.7),
    ("AIME 2024","0.1101","1.4198","0.0004",262.4),
    ("AIME 2025","0.1203","1.4517","0.0004",291.1),
    ("AIME 2026","0.1197","1.5477","0.0004",281.4),
    ("GPQA Diamond","0.0708","1.7781","0.0019",38.1),
    ("GSM8K","0.0101","0.8481","0.0006",16.3),
    ("MMLU-Pro","0.0345","1.4084","0.0015",22.9),
    ("SciBench","0.0434","1.2572","0.0005",83.0),
]
def break_even(source,builder,hit):
    if source <= hit: return None
    return ceil(builder/(source-hit))
if __name__=="__main__":
    for name,s,b,c,ratio in ROWS:
        print(f"{name:<13} {ratio:>5.1f}x reported; {break_even(D(s),D(b),D(c)):>2} future hits (rounded inputs)")
