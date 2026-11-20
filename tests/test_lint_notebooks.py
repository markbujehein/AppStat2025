import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from lint_notebooks import lint, yaml_like_rules  # noqa: E402


def test_rule_detection_follows_quarto():
    assert yaml_like_rules(["---\n", "## Title"]) == [0]  # start of cell
    assert yaml_like_rules(["a\n", "  \n", "---\n", "_later_\n"]) == [2]  # mid-cell
    assert yaml_like_rules(["a\n", "---\n", "b"]) == [1]  # under text: heading, also opens YAML
    assert yaml_like_rules(["a\n", "\n", "---\n", "\n", "b"]) == []  # rule followed by blank
    assert yaml_like_rules(["a\n", "\n", "----\n", "b"]) == []  # four dashes


def test_fix_replaces_only_offending_lines(tmp_path):
    cells = [
        {
            "cell_type": "markdown",
            "source": ["Q:\n", "\n", "---\n", "_later_\n", "\n", "---\n", "x"],
        },
        {"cell_type": "markdown", "source": ["## Fine"]},
        {"cell_type": "code", "source": ["---"]},
    ]
    nb = tmp_path / "a.ipynb"
    nb.write_text(json.dumps({"cells": cells}))
    assert lint(nb) == [(0, 2), (0, 5)]
    lint(nb, fix=True)
    assert lint(nb) == []
    assert json.loads(nb.read_text())["cells"][0]["source"][2] == "***\n"
