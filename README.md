# Applied Statistics 2025

Fork of the [Applied Statistics 2025/26](https://www.nbi.dk/~petersen/Teaching/AppliedStatistics2025.html) course repository (NBI, KU).

* `Week0`–`Week8`, `Project`, `DataAndCodeFor*`: course material (only fixes for current numpy/iminuit).
* `work/`: my own copies of the exercises (`*_mark.ipynb`, or `*_marimo.py` once converted).
* `src/appstat/`: helper functions I wrote, tested in `tests/`.

## Run

```bash
pixi install
pixi run jupyter notebook   # open a notebook
pixi run check              # lint, format check, tests
pixi run docs               # build the API reference and the site (runs the notebooks in work/) in _site/
pixi run hooks              # once: install the pre-commit hooks (strips notebook outputs)
```

Rendered notebooks: <https://markbujehein.github.io/AppStat2025/>

Licence: see `LICENSE`.
