"""Reading data files that sit next to a notebook, natively and in the browser."""

import sys
from urllib.request import urlopen


def read_text(location, name: str) -> str:
    """
    Text of the data file ``name`` that belongs to a notebook.

    On a computer the file is next to the notebook. In the browser (Pyodide) it is in the page's
    ``public/`` folder, from where the site build puts it.

    Parameters
    ----------
    location : path-like
        The notebook's folder, from ``mo.notebook_location()``.
    name : str
        File name.

    Returns
    -------
    str
        The content of the file.
    """
    if sys.platform == "emscripten":  # shortcut: the browser branch has no test, check in a browser
        return urlopen(str(location / "public" / name)).read().decode()
    return (location / name).read_text()
