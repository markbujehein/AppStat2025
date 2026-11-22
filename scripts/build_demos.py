"""Export the Week1 to Week8 marimo notebooks as interactive pages in ``_site/demos/<name>/``.

The pages run in the browser (Pyodide), where ``appstat`` is not installed. So each export uses a
copy of the notebook that first downloads ``appstat`` from ``public/appstat.txt`` next to the page.
The browser's Python (Pyodide) cannot parse ``lazy import`` ("SyntaxError: invalid syntax"), so that
downloaded copy has plain imports. The files in ``src/`` and ``work/`` are not changed.
"""

import hashlib
import html
import json
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "_site" / "demos"

BOOTSTRAP = """    import sys
__IMPORTS__
    if sys.platform == "emscripten":
        from pathlib import Path
        from urllib.request import urlopen

        # The frontend is shared, so mo.notebook_location() is demos/, and public/ is shared too.
        # One file holds all modules, as "name, newline, source", separated by form feeds.
        _package = Path("/tmp/appstat_src/appstat")
        _package.mkdir(parents=True, exist_ok=True)
        _url = str(mo.notebook_location() / "public" / "appstat" / "__NAME__.txt")
        for _part in urlopen(_url).read().decode().split("\\f"):
            _name, _, _source = _part.partition("\\n")
            (_package / f"{_name}.py").write_text(_source)
        sys.path.insert(0, "/tmp/appstat_src")
"""

# Packages that appstat imports. The browser installs a package only if a notebook imports it.
HEAVY = ("scipy", "iminuit")


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


def write_landing() -> None:
    """``_site/index.html``: the landing page. It embeds the demo data, so nothing is fetched."""
    demos = []
    for demo in yaml.safe_load((ROOT / "demos.yml").read_text()):
        name = demo["path"].strip("/").removeprefix("demos/")
        headings, points = lessons(next((ROOT / "work").glob(f"Week[0-9]/*/{name}_marimo.py")))
        demos.append({**demo, "headings": headings, "takeaways": points})
    # "</" must not end the script block early
    data = json.dumps(demos, ensure_ascii=False).replace("</", "<\\/")
    page = (ROOT / "site" / "landing.html").read_text()
    header = (ROOT / "site" / "header.html").read_text().replace('"../', '"./')
    page = page.replace("{{header}}", header).replace("{{data}}", data)
    (SITE.parent / "index.html").write_text(page)


def appstat_modules(text: str, sources: dict[str, str]) -> set[str]:
    """The appstat modules ``text`` imports, the ones those import, and so on."""
    found: set[str] = set()
    todo = re.findall(r"from appstat\.(\w+) import", text)
    while todo:
        name = todo.pop()
        if name not in found:
            found.add(name)
            todo += re.findall(r"from appstat\.(\w+) import", sources[name])
    return found


def with_bootstrap(name: str, text: str, modules: set[str], sources: dict[str, str]) -> str:
    """The notebook with a first step that downloads ``appstat``, and imports only what it needs."""
    heavy = sorted(
        pkg
        for pkg in HEAVY
        if any(re.search(rf"^(from|import) {pkg}\b", sources[m], re.M) for m in modules)
    )
    imports = "".join(f"    import {pkg}  # noqa: F401  # imported by appstat\n" for pkg in heavy)
    bootstrap = BOOTSTRAP.replace("__IMPORTS__\n", imports).replace("__NAME__", name)
    return text.replace("    import marimo as mo\n", "    import marimo as mo\n" + bootstrap, 1)


def tree_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for file in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(file.relative_to(path).as_posix().encode())
        digest.update(file.read_bytes())
    return digest.hexdigest()


def share_assets(apps: list[Path]) -> None:
    """Keep one copy of the marimo frontend in ``demos/assets``.

    Every export ships the same 28 MB, so the browser would download it again for each demo.
    An app whose copy differs keeps its own.
    """
    reference = tree_hash(apps[0] / "assets")
    shutil.copytree(apps[0] / "assets", SITE / "assets")
    for app in apps:
        if tree_hash(app / "assets") != reference:
            print(f"{app.parent.name}: assets differ from {apps[0].parent.name}, not shared")
            continue
        shutil.rmtree(app / "assets")
        page = app / "index.html"
        page.write_text(page.read_text().replace("./assets/", "../../assets/"))


def build_demo(notebook: Path, sources: dict[str, str]) -> Path:
    """Export one notebook as a static page and as a Pyodide app. Returns the app folder."""
    name = notebook.stem.removesuffix("_marimo")
    out = SITE / name
    text = notebook.read_text()
    modules = appstat_modules(text, sources)
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / notebook.name
        copy.write_text(with_bootstrap(name, text, modules, sources) if modules else text)
        subprocess.run(
            ["marimo", "export", "html-wasm", str(copy), "-o", str(out / "app"), "--mode", "run"]
            + ["-f"],
            check=True,
            capture_output=True,
        )
    # The static page runs the notebook here, where appstat is installed. Without it the demo
    # page offers only the interactive tab.
    static = subprocess.run(
        ["marimo", "export", "html", str(notebook), "-o", str(out / "static.html")]
        + ["--no-include-code", "-f"],
        capture_output=True,
    )
    if static.returncode:
        print(f"{name}: static export failed, interactive only")
    if modules:  # one file to download instead of one per module
        bundle = "\f".join(f"{m}\n{sources[m]}" for m in ("__init__", *sorted(modules)))
        (SITE / "public" / "appstat" / f"{name}.txt").write_text(bundle)
    for pattern in DATA.get(name, []):  # notebooks read these through public/
        for data in notebook.parent.glob(pattern):
            shutil.copy(data, SITE / "public" / data.name)
    write_demo_page(name, has_static=not static.returncode)
    print(f"built {name}")
    return out / "app"


def write_demo_page(name: str, *, has_static: bool) -> None:
    """``demos/<name>/index.html``: title, a Read tab and a Run tab around the two exports."""
    demo = next(
        d for d in yaml.safe_load((ROOT / "demos.yml").read_text()) if d["path"] == f"demos/{name}/"
    )
    tags = "".join(f'<span class="tag">{t}</span>' for t in demo["categories"])
    header = (ROOT / "site" / "header.html").read_text().replace('"../', '"../../')
    page = (ROOT / "site" / "demo.html").read_text()
    for key, value in {
        "header": header,
        "title": html.escape(demo["title"]),
        "description": html.escape(demo["description"]),
        "tags": tags,
        "static_hidden": "" if has_static else " hidden",
    }.items():
        page = page.replace("{{" + key + "}}", value)
    (SITE / name / "index.html").write_text(page)


def main() -> None:
    shutil.rmtree(SITE, ignore_errors=True)
    (SITE / "public" / "appstat").mkdir(parents=True)
    sources = {
        source.stem: re.sub(r"^lazy (import|from)", r"\1", source.read_text(), flags=re.M)
        for source in sorted((ROOT / "src" / "appstat").glob("*.py"))
    }
    notebooks = sorted((ROOT / "work").glob("Week[0-9]/*/*_marimo.py"))
    with ThreadPoolExecutor(4) as pool:  # each export waits mostly on a notebook kernel
        apps = list(pool.map(lambda nb: build_demo(nb, sources), notebooks))
    share_assets(apps)
    write_landing()
    (SITE.parent / "site").mkdir(exist_ok=True)
    for name in ("demo.js", "landing.js", "site.css"):
        shutil.copy(ROOT / "site" / name, SITE.parent / "site" / name)
    shutil.copy(ROOT / "site" / "sw.js", SITE.parent / "sw.js")


if __name__ == "__main__":
    sys.exit(main())
