"""Synthetic capability-versus-oversight pacing experiment. No real AI data."""
import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Scenario:
    name: str
    capability_growth: float
    oversight_growth: float
    initial_capability: float = 0.20
    initial_oversight: float = 0.35


def simulate(scenario: Scenario, months: int = 24):
    if months < 0 or not (0 <= scenario.initial_capability <= 1 and 0 <= scenario.initial_oversight <= 1):
        raise ValueError('Invalid inputs')
    if scenario.capability_growth < 0 or scenario.oversight_growth < 0:
        raise ValueError('Growth rates must be nonnegative')
    rows = []
    for month in range(months + 1):
        capability = min(1.0, scenario.initial_capability + month * scenario.capability_growth)
        oversight = min(1.0, scenario.initial_oversight + month * scenario.oversight_growth)
        gap = max(0.0, capability - oversight)
        rows.append({'scenario': scenario.name, 'month': month, 'capability': round(capability, 6),
                     'oversight': round(oversight, 6), 'uncovered_gap': round(gap, 6)})
    return rows


def examples():
    return [
        Scenario('fast_progress', .032, .009),
        Scenario('paced_progress', .023, .009),
        Scenario('paced_plus_oversight', .023, .025),
    ]


def main():
    parser = argparse.ArgumentParser(description='Synthetic capability/oversight experiment, not a forecast.')
    parser.add_argument('--months', type=int, default=24)
    parser.add_argument('--csv', type=Path, default=Path('results.csv'))
    args = parser.parse_args()
    rows = [row for s in examples() for row in simulate(s, args.months)]
    with args.csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['scenario','month','capability','oversight','uncovered_gap'])
        writer.writeheader(); writer.writerows(rows)
    for s in examples():
        last = simulate(s, args.months)[-1]
        print(f"{s.name:21} month={args.months:2d} capability={last['capability']:.3f} "
              f"oversight={last['oversight']:.3f} uncovered_gap={last['uncovered_gap']:.3f}")
    print(f'Synthetic output saved to {args.csv}')

if __name__ == '__main__':
    main()
