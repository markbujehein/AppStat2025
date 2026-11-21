"""Tests for reading data files the same way natively and in the browser."""

from pathlib import Path

from appstat.data import read_text


def test_read_text_reads_a_file_next_to_the_notebook(tmp_path: Path):
    # IF a data file in the notebook's folder
    (tmp_path / "table.txt").write_text("a b\n1 2\n")

    # THEN
    text = read_text(tmp_path, "table.txt")

    # EXPECT its content
    assert text == "a b\n1 2\n"
