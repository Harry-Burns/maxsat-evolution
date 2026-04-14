"""
MSE17 optimum lookup.

Usage:
    from mse17_lookup import lookup, lookup_nsat

    # Returns dict with optimum_cost (unsatisfied clauses), family, best_time, etc.
    result = lookup("keller4.clq.wcnf")

    # Returns optimal number of SATISFIED clauses for a given instance.
    # Needs total clause count (m from the p-line) since the table stores unsatisfied.
    nsat = lookup_nsat("keller4.clq.wcnf", total_clauses=5271)
    # -> 5111
"""

import os
import re
from html.parser import HTMLParser

# Path to the downloaded HTML file. Adjust if needed.
_DEFAULT_HTML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mse-table.html")

_DB = None  # lazy-loaded cache


class _TableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_cell = False
        self.rows = []
        self.current_row = []
        self.current_cell = ""

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.current_row = []
        elif tag in ("td", "th"):
            self.in_cell = True
            self.current_cell = ""

    def handle_endtag(self, tag):
        if tag in ("td", "th"):
            self.current_row.append(self.current_cell.strip())
            self.in_cell = False
        elif tag == "tr" and self.current_row:
            self.rows.append(self.current_row)

    def handle_data(self, data):
        if self.in_cell:
            self.current_cell += data


def _load(html_path=None):
    global _DB
    if _DB is not None:
        return _DB

    path = html_path or _DEFAULT_HTML
    with open(path, encoding="utf-8") as f:
        html = f.read()

    parser = _TableParser()
    parser.feed(html)

    db = {}
    for row in parser.rows[1:]:  # skip header
        benchmark = row[0]
        family = benchmark.split("/")[0] if "/" in benchmark else ""
        filename = benchmark.split("/")[-1] if "/" in benchmark else benchmark

        optimum_cost = None
        best_time = None
        num_solvers = 0

        for cell in row[1:]:
            m = re.match(r"([\d.]+)\s+\((\d+|-)\)", cell)
            if m and m.group(2) != "-":
                cost = int(m.group(2))
                time_val = float(m.group(1))
                if optimum_cost is None:
                    optimum_cost = cost
                num_solvers += 1
                if best_time is None or time_val < best_time:
                    best_time = round(time_val, 3)

        db[benchmark] = {
            "optimum_cost": optimum_cost,
            "family": family,
            "filename": filename,
            "best_time": best_time,
            "num_solvers": num_solvers,
        }

    _DB = db
    return _DB


def lookup(query, html_path=None):
    """
    Look up a benchmark. Accepts:
      - "maxclique/keller4.clq.wcnf"  (full key)
      - "keller4.clq.wcnf"            (just filename)
      - "C:\\data\\maxclique\\keller4.clq.wcnf"  (full path)

    Returns a dict or None if not found:
      {
        "optimum_cost": int or None,   # unsatisfied clauses at optimum
        "family": str,
        "filename": str,
        "best_time": float or None,
        "num_solvers": int,
      }
    """
    db = _load(html_path)

    if query in db:
        return db[query]

    # Try last two path components as family/filename
    parts = query.replace("\\", "/").rstrip("/").split("/")
    if len(parts) >= 2:
        key = f"{parts[-2]}/{parts[-1]}"
        if key in db:
            return db[key]

    # Match by filename alone
    basename = os.path.basename(query)
    matches = [k for k, v in db.items() if v["filename"] == basename]
    if len(matches) == 1:
        return db[matches[0]]
    elif len(matches) > 1:
        # Multiple families may share a filename (e.g. brock200_1.clq.wcnf in maxclique vs maxcut)
        return db[matches[0]]

    return None


def lookup_nsat(query, total_clauses, html_path=None):
    """
    Returns the optimal number of satisfied clauses, or None if unknown.
    total_clauses is m from the 'p wcnf n m' line of the .wcnf file.
    """
    result = lookup(query, html_path)
    if result is None or result["optimum_cost"] is None:
        return None
    return total_clauses - result["optimum_cost"]