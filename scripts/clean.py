"""Delete what the docs build and notebook runs generate (git-ignored or untracked)."""

import shutil
from pathlib import Path

GENERATED = [
    "_site",
    ".quarto",
    "site_libs",
    "reference",  # quartodoc API pages
    "objects.json",
    "index.html",
    "index-listing.json",
    ".pytest_cache",
    ".ruff_cache",
    "work/*/*/*_mark.html",  # left behind by an interrupted quarto render
    "work/*/*/*_mark_files",
]

for pattern in GENERATED:
    for path in Path().glob(pattern):
        shutil.rmtree(path) if path.is_dir() else path.unlink()
        print("removed", path)
