"""Export the Week1 to Week8 marimo notebooks as interactive pages in ``_site/demos/<name>/``.

The pages run in the browser (Pyodide), where ``appstat`` is not installed. So each export uses a
copy of the notebook that first downloads ``appstat`` from ``public/appstat/`` next to the page.
The browser's Python (Pyodide) cannot parse ``lazy import`` ("SyntaxError: invalid syntax"), so that
downloaded copy has plain imports. The files in ``src/`` and ``work/`` are not changed.
"""

import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "_site" / "demos"

BOOTSTRAP = """    import sys

    import iminuit  # noqa: F401  # imported only by appstat, listed so the browser installs them
    import scipy  # noqa: F401

    if sys.platform == "emscripten":
        from pathlib import Path
        from urllib.request import urlopen

        _package = Path("/tmp/appstat_src/appstat")
        _package.mkdir(parents=True, exist_ok=True)
        for _name in __MODULES__:
            _url = str(mo.notebook_location() / "public" / "appstat" / f"{_name}.py")
            (_package / f"{_name}.py").write_text(urlopen(_url).read().decode())
        sys.path.insert(0, "/tmp/appstat_src")
"""


# Data files each notebook reads. Listed on purpose: the folders also hold private (student) or very
# large files, which must not end up on the public site.
DATA = {
    "TableMeasurements": ["data_TableMeasurements*.txt"],
    "CompanySize": ["TableData_Denmark2020.txt"],
    "RandomDigitsTest": ["data_RandomDigits2025_?.txt", "pi1000000.txt"],
    "ATLAStestbeam": ["DataSet_AtlasPid_ElectronPion_2GeV.txt"],
    "Calibration": ["data_calib.txt"],
    "FisherDiscriminant": ["DataSet_AndersonFisherIris.txt"],
}


def plain(text: str) -> str:
    """Notebook markdown without LaTeX, bold and code marks, for search and plain display."""
    return re.sub(r"[$*`\\]", "", text).strip()


def lessons(notebook: Path) -> tuple[list[str], list[str]]:
    """The section headings and the numbered "What to take away" points of a marimo notebook."""
    lines = notebook.read_text().splitlines()
    headings = [plain(m[1]) for line in lines if (m := re.match(r"    ##+ (.+)", line))]
    start = next(
        (i for i, line in enumerate(lines) if line.strip() == "## What to take away"), None
    )
    points: list[str] = []
    for line in lines[start + 1 :] if start is not None else []:
        if line.strip() == '""")':
            break
        if m := re.match(r"\s*\d+\. (.+)", line):
            points.append(m[1])
        elif points and line.strip():
            points[-1] += " " + line.strip()
    return [h for h in headings if h != "What to take away"], [plain(p) for p in points]


def write_search_index() -> None:
    """``demos/search-index.json``: what the Lessons page (methods.qmd) searches."""
    index = []
    for demo in yaml.safe_load((ROOT / "demos.yml").read_text()):
        name = demo["path"].strip("/").removeprefix("demos/")
        notebook = next((ROOT / "work").glob(f"Week[0-9]/*/{name}_marimo.py"))
        headings, points = lessons(notebook)
        index.append(
            {
                "title": demo["title"],
                "description": demo["description"],
                "path": demo["path"],
                "categories": demo["categories"],
                "headings": headings,
                "takeaways": points,
            }
        )
    (SITE / "search-index.json").write_text(json.dumps(index, ensure_ascii=False))


MODULES = tuple(sorted(p.stem for p in (ROOT / "src" / "appstat").glob("*.py")))


def main() -> None:
    shutil.rmtree(SITE, ignore_errors=True)
    package = SITE / "_appstat" / "appstat"
    for source in sorted((ROOT / "src" / "appstat").glob("*.py")):
        text = re.sub(r"^lazy (import|from)", r"\1", source.read_text(), flags=re.M)
        package.mkdir(parents=True, exist_ok=True)
        (package / source.name).write_text(text)

    for notebook in sorted((ROOT / "work").glob("Week[0-9]/*/*_marimo.py")):
        name = notebook.stem.removesuffix("_marimo")
        out = SITE / name
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / notebook.name
            copy.write_text(
                notebook.read_text().replace(
                    "    import marimo as mo\n",
                    "    import marimo as mo\n" + BOOTSTRAP.replace("__MODULES__", repr(MODULES)),
                    1,
                )
            )
            subprocess.run(
                ["marimo", "export", "html-wasm", str(copy), "-o", str(out), "--mode", "run", "-f"],
                check=True,
            )
        shutil.copytree(package, out / "public" / "appstat")
        for pattern in DATA.get(name, []):  # notebooks read these through public/
            for data in notebook.parent.glob(pattern):
                shutil.copy(data, out / "public" / data.name)
        print(f"built {out.relative_to(ROOT)}")
    shutil.rmtree(SITE / "_appstat")
    write_search_index()


if __name__ == "__main__":
    sys.exit(main())
