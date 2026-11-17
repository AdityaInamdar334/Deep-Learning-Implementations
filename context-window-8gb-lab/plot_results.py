#!/usr/bin/env python3
"""Generate dependency-free SVG charts from benchmark CSV results."""
from __future__ import annotations
import argparse
import csv
import html
import statistics
from collections import defaultdict
from pathlib import Path


def chart(rows: list[dict], metric: str, title: str, outfile: Path) -> None:
    groups = defaultdict(list)
    for row in rows:
        if row.get("error") or not row.get(metric):
            continue
        try:
            groups[(row["mode"], int(row["configured_context"]))].append(float(row[metric]))
        except (ValueError, TypeError):
            pass
    points = sorted((mode, ctx, statistics.median(vals)) for (mode, ctx), vals in groups.items())
    width, height, left, right, top, bottom = 940, 490, 92, 35, 85, 75
    max_y = max((v for _, _, v in points), default=1) * 1.12 or 1
    contexts = sorted({ctx for _, ctx, _ in points})
    def x(ctx):
        return left + (width-left-right) * (contexts.index(ctx) + .5) / max(1, len(contexts))
    def y(val):
        return height-bottom - (height-top-bottom)*val/max_y
    colors = {"fixed": "#2563eb", "scaled": "#ea580c"}
    elements = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                '<rect width="100%" height="100%" fill="white"/>',
                f'<text x="{left}" y="38" font-size="23" font-family="Arial" font-weight="bold">{html.escape(title)}</text>']
    for i in range(5):
        value = max_y * i/4
        yy = y(value)
        elements.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}" stroke="#e5e7eb"/>')
        elements.append(f'<text x="{left-12}" y="{yy+4:.1f}" text-anchor="end" font-size="13" font-family="Arial" fill="#4b5563">{value:.0f}</text>')
    for ctx in contexts:
        xx = x(ctx)
        elements.append(f'<text x="{xx:.1f}" y="{height-bottom+28}" text-anchor="middle" font-size="14" font-family="Arial">{ctx:,}</text>')
    elements.append(f'<text x="{width/2}" y="{height-18}" text-anchor="middle" font-size="15" font-family="Arial">Configured context (tokens)</text>')
    for mode in ("fixed", "scaled"):
        seq = [(x(ctx), y(val)) for m, ctx, val in points if m == mode]
        if seq:
            coords = " ".join(f"{xx:.1f},{yy:.1f}" for xx, yy in seq)
            elements.append(f'<polyline points="{coords}" fill="none" stroke="{colors[mode]}" stroke-width="3"/>')
            for xx, yy in seq:
                elements.append(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="5" fill="{colors[mode]}"/>')
    for i, mode in enumerate(("fixed", "scaled")):
        xx = width-260 + i*110
        elements.append(f'<circle cx="{xx}" cy="58" r="5" fill="{colors[mode]}"/><text x="{xx+12}" y="63" font-family="Arial" font-size="13">{mode}</text>')
    if not points:
        elements.append('<text x="350" y="260" font-family="Arial" font-size="20">No valid measurements</text>')
    elements.append('</svg>')
    outfile.write_text("\n".join(elements), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("results/context-sweep/runs.csv"))
    parser.add_argument("--out", type=Path, default=Path("results/context-sweep/plots"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with args.csv.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for key, title in [("gpu_peak_mib", "Peak sampled GPU memory (MiB)"),
                       ("gpu_delta_peak_mib", "Peak GPU memory above baseline (MiB)"),
                       ("wall_seconds", "End-to-end request latency (s)"),
                       ("prefill_tps_uncached", "Uncached prompt throughput (tokens/s)"),
                       ("decode_tps", "Generation throughput (tokens/s)")]:
        chart(rows, key, title, args.out / f"{key}.svg")
    print(f"Wrote SVG charts to {args.out}")

if __name__ == "__main__":
    main()
