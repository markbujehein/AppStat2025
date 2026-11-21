"""Export the Week1 to Week3 marimo notebooks as interactive pages in ``_site/demos/<name>/``.

The pages run in the browser (Pyodide), where ``appstat`` is not installed. So each export uses a
copy of the notebook that first downloads ``appstat`` from ``public/appstat/`` next to the page.
The browser's Python (Pyodide) cannot parse ``lazy import`` ("SyntaxError: invalid syntax"), so that
downloaded copy has plain imports. The files in ``src/`` and ``work/`` are not changed.
"""

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

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


MODULES = tuple(sorted(p.stem for p in (ROOT / "src" / "appstat").glob("*.py")))


def main() -> None:
    shutil.rmtree(SITE, ignore_errors=True)
    package = SITE / "_appstat" / "appstat"
    for source in sorted((ROOT / "src" / "appstat").glob("*.py")):
        text = re.sub(r"^lazy (import|from)", r"\1", source.read_text(), flags=re.M)
        package.mkdir(parents=True, exist_ok=True)
        (package / source.name).write_text(text)

    for notebook in sorted((ROOT / "work").glob("Week[123]/*/*_marimo.py")):
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
        for data in notebook.parent.glob("*.txt"):  # notebooks read these through public/
            shutil.copy(data, out / "public" / data.name)
        print(f"built {out.relative_to(ROOT)}")
    shutil.rmtree(SITE / "_appstat")


if __name__ == "__main__":
    sys.exit(main())
