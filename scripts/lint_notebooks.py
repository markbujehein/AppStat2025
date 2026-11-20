"""Lint notebook markdown cells for Quarto.

Quarto (kRegExYAML in its bundle) reads any line that is exactly '---' and is followed by a
non-blank line as the start of a YAML block, whatever comes before it, and fails with a
YAMLException. '***' renders as the same horizontal rule (a '---' under a text line was a
heading, and becomes a rule).
Usage: lint_notebooks.py [--fix] [notebooks...]   (default: work/**/*.ipynb)
"""

import json
import sys
from pathlib import Path


def yaml_like_rules(lines: list[str]) -> list[int]:
    """Indices of '---' lines that Quarto would read as the start of a YAML block."""
    return [
        j
        for j, line in enumerate(lines)
        if line.rstrip() == "---" and j + 1 < len(lines) and lines[j + 1].strip()
    ]


def lint(path: Path, fix: bool = False) -> list[tuple[int, int]]:
    """Return (cell, line) of every offending '---'; replace them with '***' if fix."""
    nb = json.loads(path.read_text(encoding="utf8"))
    bad = []
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "markdown":
            continue
        is_list = isinstance(cell["source"], list)
        lines = "".join(cell["source"]).splitlines(keepends=True)
        for j in yaml_like_rules(lines):
            bad.append((i, j))
            lines[j] = lines[j].replace("---", "***", 1)
        if fix and any(c == i for c, _ in bad):
            cell["source"] = lines if is_list else "".join(lines)
    if fix and bad:
        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf8")
    return bad


if __name__ == "__main__":
    args = sys.argv[1:]
    fix = "--fix" in args
    paths = [Path(a) for a in args if a != "--fix"] or sorted(Path("work").glob("**/*.ipynb"))
    failed = False
    for p in paths:
        if bad := lint(p, fix):
            print(f"{p}: (cell, line) with YAML-like '---': {bad}{' (fixed)' if fix else ''}")
            failed |= not fix
    sys.exit(failed)
