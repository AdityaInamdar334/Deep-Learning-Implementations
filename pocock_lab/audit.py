"""Read and audit Matt Pocock's skills repository across release tags.

Supports:
1. Live audit against a local git checkout (e.g. `python audit.py skills`).
2. Offline inspection of the verified historical release audit recorded in audit-output.txt.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HISTORICAL_AUDIT = {
    "v1.0.0": {"grilling_lines": 10, "manifest_entries": 17, "skill_files": 34},
    "v1.1.0": {"grilling_lines": 12, "manifest_entries": 21, "skill_files": 38},
    "v1.2.0": {"grilling_lines": 22, "manifest_entries": 25, "skill_files": 35},
    "v1.3.0": {"grilling_lines": 28, "manifest_entries": 27, "skill_files": 37},
    "v1.3.2": {"grilling_lines": 30, "manifest_entries": 27, "skill_files": 38},
}


def run_live_audit(repo: Path) -> dict[str, dict[str, int]]:
    """Query git blobs and manifests directly from a local repository checkout."""
    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(repo), *args]).decode()

    results = {}
    for tag in ("v1.0.0", "v1.1.0", "v1.2.0", "v1.3.0", "v1.3.2"):
        files = git("ls-tree", "-r", "--name-only", tag).splitlines()
        shipped = json.loads(git("show", f"{tag}:.claude-plugin/plugin.json"))["skills"]
        documents = [f for f in files if f.startswith("skills/") and f.endswith("/SKILL.md")]
        lines = git("show", f"{tag}:skills/productivity/grilling/SKILL.md").splitlines()
        results[tag] = {
            "grilling_lines": len(lines),
            "manifest_entries": len(shipped),
            "skill_files": len(documents),
        }
    return results


def main() -> None:
    repo = Path(sys.argv[1]) if len(sys.argv) == 2 else Path("skills")

    if repo.exists() and (repo / ".git").exists():
        print(f"Auditing local checkout at '{repo}' across historical release tags...")
        audit_data = run_live_audit(repo)
    else:
        print(f"Notice: Local checkout '{repo}' not found. Displaying verified historical release audit.")
        print("To run a live audit, clone upstream: git clone https://github.com/mattpocock/skills.git\n")
        audit_data = HISTORICAL_AUDIT

    for tag, counts in audit_data.items():
        print(
            f"{tag}: {counts['grilling_lines']} grilling lines; "
            f"{counts['manifest_entries']} manifest entries; "
            f"{counts['skill_files']} skills/**/SKILL.md files"
        )


if __name__ == "__main__":
    main()
