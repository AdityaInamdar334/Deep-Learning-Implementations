"""Automated Benchmark Audit Runner.

Runs comprehensive evaluations across:
1. SciCode original vs verified benchmark suites (cascading dependency recovery).
2. DeepQuestion cognitive complexity degradation across Bloom's taxonomy.
3. Epoch AI formal review rubric across historical benchmark suites.
Exports results to results/results.json and results/REPORT.md.
"""

import json
from pathlib import Path
from audit_engine import BenchmarkAuditor, create_synthetic_scicode_suite, DefectType
from cognitive_complexity import CognitiveComplexityEngine, ModelProfile, create_standard_cognitive_suite
from epoch_rubric import EpochReviewEngine, get_historical_epoch_registry

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 70)
    print("RUNNING BENCHMARK AUDITING & INTEGRITY EVALUATION")
    print("=" * 70)

    # -------------------------------------------------------------
    # 1. SciCode Multi-Step Audit Simulation
    # -------------------------------------------------------------
    print("\n[1/3] Simulating SciCode Original vs Verified Multi-Step Grading...")
    problems = create_synthetic_scicode_suite(num_problems=65, seed=42)
    auditor = BenchmarkAuditor(problems)

    total_subproblems = sum(len(p.subproblems) for p in problems)
    defective_subproblems = sum(
        sum(1 for s in p.subproblems if s.defect_type != DefectType.NONE)
        for p in problems
    )
    defective_problems = sum(
        1 for p in problems if any(s.defect_type != DefectType.NONE for s in p.subproblems)
    )

    print(f"  Total Problems: {len(problems)}")
    print(f"  Total Subproblems: {total_subproblems}")
    print(f"  Defective Subproblems: {defective_subproblems} ({defective_subproblems/total_subproblems*100:.1f}%)")
    print(f"  Problems Affected by Defects: {defective_problems} ({defective_problems/len(problems)*100:.1f}%)")

    # Evaluate Frontier Reasoning Model (produces mathematically correct outputs)
    frontier_reasoning_outputs = {
        s.id: s.analytical_solution for p in problems for s in p.subproblems
    }
    # Note: globals do not contain undocumented helper names by default
    frontier_globals = {}

    sub_orig_pass = 0
    sub_verif_pass = 0
    whole_orig_pass = 0
    whole_verif_pass = 0

    for p in problems:
        w_orig, w_verif, sub_res = auditor.evaluate_whole_problem(
            p, frontier_reasoning_outputs, frontier_globals, is_memorizer=False
        )
        if w_orig:
            whole_orig_pass += 1
        if w_verif:
            whole_verif_pass += 1
        for res in sub_res.values():
            if res.passed_original:
                sub_orig_pass += 1
            if res.passed_verified:
                sub_verif_pass += 1

    sub_orig_acc = (sub_orig_pass / total_subproblems) * 100.0
    sub_verif_acc = (sub_verif_pass / total_subproblems) * 100.0
    whole_orig_acc = (whole_orig_pass / len(problems)) * 100.0
    whole_verif_acc = (whole_verif_pass / len(problems)) * 100.0

    print(f"  Original Benchmark  -> Subproblem: {sub_orig_acc:.1f}% | Whole-Problem: {whole_orig_acc:.1f}%")
    print(f"  Verified Benchmark  -> Subproblem: {sub_verif_acc:.1f}% | Whole-Problem: {whole_verif_acc:.1f}%")
    print(f"  Recovery Delta      -> Subproblem: +{sub_verif_acc - sub_orig_acc:.1f} pp | Whole-Problem: +{whole_verif_acc - whole_orig_acc:.1f} pp")

    # -------------------------------------------------------------
    # 2. DeepQuestion Cognitive Complexity Evaluation
    # -------------------------------------------------------------
    print("\n[2/3] Simulating DeepQuestion Cognitive Complexity (Bloom's Taxonomy)...")
    cog_suite = create_standard_cognitive_suite()
    cog_engine = CognitiveComplexityEngine(cog_suite)

    models = [
        ModelProfile("Frontier-Reasoning-Model", base_knowledge_capacity=0.96, reasoning_depth_factor=0.88),
        ModelProfile("Standard-Chat-Model", base_knowledge_capacity=0.92, reasoning_depth_factor=0.68),
        ModelProfile("Memorizer-Model", base_knowledge_capacity=0.95, reasoning_depth_factor=0.45),
    ]

    cognitive_results = {}
    for m in models:
        scores = cog_engine.evaluate_model(m, seed=123)
        cognitive_results[m.name] = {lvl.name: round(score * 100.0, 1) for lvl, score in scores.items()}
        print(f"  Model: {m.name}")
        for lvl_name, pct in cognitive_results[m.name].items():
            print(f"    - {lvl_name:<12}: {pct:.1f}%")

    # -------------------------------------------------------------
    # 3. Epoch AI Historical Review Registry Audit
    # -------------------------------------------------------------
    print("\n[3/3] Inspecting Epoch Benchmark Review Registry & Verdicts...")
    epoch_registry = get_historical_epoch_registry()
    epoch_summary = []

    for item in epoch_registry:
        epoch_summary.append({
            "benchmark": item.benchmark_name,
            "release_date": item.release_date,
            "sample_size": item.inspected_sample_size,
            "defects": item.score_affecting_defects,
            "verdict": item.verdict.value,
            "rationale": item.rationale,
        })
        print(f"  {item.benchmark_name:<38} | Verdict: {item.verdict.value:<15} | Defects: {item.score_affecting_defects}/{item.inspected_sample_size}")

    # -------------------------------------------------------------
    # Export Artifacts
    # -------------------------------------------------------------
    results_payload = {
        "scicode_audit": {
            "num_problems": len(problems),
            "num_subproblems": total_subproblems,
            "defective_subproblems": defective_subproblems,
            "defective_problems": defective_problems,
            "subproblem_acc_original": round(sub_orig_acc, 2),
            "subproblem_acc_verified": round(sub_verif_acc, 2),
            "whole_problem_acc_original": round(whole_orig_acc, 2),
            "whole_problem_acc_verified": round(whole_verif_acc, 2),
        },
        "cognitive_complexity": cognitive_results,
        "epoch_registry": epoch_summary,
    }

    results_file = RESULTS_DIR / "results.json"
    with open(results_file, "w") as f:
        json.dump(results_payload, f, indent=2)
    print(f"\nSaved raw results to {results_file}")

    # Generate REPORT.md
    report_lines = [
        "# Benchmark Auditing & Measurement Integrity Report",
        "",
        "## 1. SciCode Benchmark Defect Audit & Recovery",
        "",
        "| Metric | Original SciCode | SciCode-Verified | Recovery Delta |",
        "| :--- | :---: | :---: | :---: |",
        f"| **Subproblem Accuracy** | {sub_orig_acc:.1f}% | **{sub_verif_acc:.1f}%** | +{sub_verif_acc - sub_orig_acc:.1f} pp |",
        f"| **Whole-Problem Accuracy** | {whole_orig_acc:.1f}% | **{whole_verif_acc:.1f}%** | **+{whole_verif_acc - whole_orig_acc:.1f} pp** |",
        f"| **Audited Main Problems** | {len(problems)} | {len(problems)} | — |",
        f"| **Defect Prevalence** | {defective_problems/len(problems)*100:.1f}% affected | Corrected | — |",
        "",
        "## 2. DeepQuestion Cognitive Complexity Degradation (Bloom's Taxonomy)",
        "",
        "| Model | Recall (L1) | Application (L2) | Analysis (L3) | Synthesis (L4) | Peak-to-Trough Drop |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]

    for m_name, res in cognitive_results.items():
        drop = res["RECALL"] - res["SYNTHESIS"]
        pct_drop = (drop / res["RECALL"]) * 100.0
        report_lines.append(
            f"| **{m_name}** | {res['RECALL']:.1f}% | {res['APPLICATION']:.1f}% | {res['ANALYSIS']:.1f}% | {res['SYNTHESIS']:.1f}% | -{drop:.1f} pp (-{pct_drop:.0f}%) |"
        )

    report_lines.extend([
        "",
        "## 3. Epoch AI Benchmark Reviews Classification Summary",
        "",
        "| Benchmark Suite | Sample Audited | Score-Affecting Defects | Defect Rate | Formal Verdict |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ])

    for item in epoch_registry:
        rate_str = f"{(item.score_affecting_defects / item.inspected_sample_size)*100:.1f}%" if item.inspected_sample_size > 0 else "N/A"
        report_lines.append(
            f"| **{item.benchmark_name}** | {item.inspected_sample_size} | {item.score_affecting_defects} | {rate_str} | **{item.verdict.value}** |"
        )

    report_file = RESULTS_DIR / "REPORT.md"
    report_file.write_text("\n".join(report_lines) + "\n")
    print(f"Saved formatted markdown report to {report_file}")


if __name__ == "__main__":
    main()

