"""Run the canonical compiler comparison benchmark and generate results."""

import sys
from pathlib import Path
from compare import run_comparison


def main():
    print("=" * 70)
    print("Running Trusted Compiler Comparison Benchmark...")
    print("Comparing: reference (standard) vs candidate (-O) vs fast (-OO -B)")
    print("=" * 70)

    lab_dir = Path(__file__).parent.resolve()
    py_exec = sys.executable

    report_path, results_data = run_comparison(
        cwd=str(lab_dir),
        runs=7,
        timeout=60.0,
        reference=f"{py_exec} -u workload.py",
        candidate=f"{py_exec} -O workload.py",
        fast=f"{py_exec} -OO -B workload.py",
        output=str(lab_dir / "results"),
        overwrite=True
    )

    print(f"\n✓ Benchmark complete! Report written to {report_path}")
    print("\nSummary Table:")
    print((lab_dir / "results" / "REPORT.md").read_text())


if __name__ == "__main__":
    main()
