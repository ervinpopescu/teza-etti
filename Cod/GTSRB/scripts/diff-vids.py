#!/bin/python
import difflib
import itertools
import json
import os
import pathlib
import subprocess
import sys
from typing import Iterator, List, Tuple

import ffmpeg


class Sdiffer:
    def __init__(self, max_width: int = 80):
        # Two columns with a gutter
        self._col_width = (max_width - 3) // 2
        assert self._col_width > 0

    def _fit(self, s: str) -> str:
        s = s.rstrip()[: self._col_width]
        return f"{s: <{self._col_width}}"

    def sdiff(self, a: List[str], b: List[str]) -> Iterator[str]:
        diff_lines = difflib.Differ().compare(a, b)
        diff_table: List[Tuple[str, List[str], List[str]]] = []
        for diff_type, line_group in itertools.groupby(
            diff_lines, key=lambda ln: ln[:1]
        ):
            lines = [ln[2:] for ln in line_group]
            if diff_type == " ":
                diff_table.append((" ", lines, lines))
            else:
                if not diff_table or diff_table[-1][0] != "|":
                    diff_table.append(("|", [], []))
                if diff_type == "-":
                    # Lines only in `a`
                    diff_table[-1][1].extend(lines)
                elif diff_type == "+":
                    # Lines only in `b`
                    diff_table[-1][2].extend(lines)

        for diff_type, cell_a, cell_b in diff_table:
            for left, right in itertools.zip_longest(cell_a, cell_b, fillvalue=""):
                yield f"{self._fit(left)} {diff_type} {self._fit(right)}"

    def print_sdiff(self, a: List[str], b: List[str]) -> Iterator[str]:
        print("\n".join(self.sdiff(a, b)))

current_file_path = pathlib.Path(__file__).parent
try:
    input_vid = ffmpeg.probe(sys.argv[1])
    output_vid = ffmpeg.probe(sys.argv[2])
    dump_a = json.dumps(input_vid, indent=2).splitlines(keepends=True)
    dump_b = json.dumps(output_vid, indent=2).splitlines(keepends=True)
    Sdiffer(max_width=150).print_sdiff(dump_a, dump_b)
except ffmpeg.Error as error:
    print(error.stderr.decode())

