"""Audit Table 1 at the displayed precision, keeping collapse as a distinct outcome."""
import json
from pathlib import Path
from collections import Counter

def compare(data):
    for model,methods in data.items():
        for i, task in enumerate(["AIME24","AIME25","AMC","MATH500","GPQA-D","MMLU-Pro Math"]):
            rivals = [methods[k][i] for k in ("RENT","TTRL","Co-Reward") if methods[k][i] is not None]
            score = methods["PC-Dr.DPO"][i]
            best = max(rivals,default=None)
            if best is None and score is None:
                verdict = "all collapsed"
            elif score is None:
                verdict = "collapse"
            elif best is None or score > best:
                verdict = "higher"
            elif score == best:
                verdict = "numeric tie"
            else:
                verdict = "lower"
            yield model,task,score,best,verdict

if __name__=="__main__":
    rows=list(compare(json.loads(Path(__file__).with_name("table1.json").read_text())))
    for row in rows: print(*row,sep=" | ")
    print(dict(Counter(row[-1] for row in rows)))
