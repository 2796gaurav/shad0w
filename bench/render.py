"""Render bench/results/results.json into bench/site/index.html: one self-contained page (data, styles and script
inlined; it opens from disk with no server and no network).

    python bench/render.py              # bench/site/index.html (a complete HTML document)
    python bench/render.py --fragment   # also bench/site/fragment.html: the same page without <html>/<head>/<body>
"""
from __future__ import annotations

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

TEMPLATE = os.path.join(HERE, "site_template.html")  # the page; __DATA__ is replaced by results.json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fragment", action="store_true")
    a = ap.parse_args()
    data = json.load(open(os.path.join(HERE, "results", "results.json")))
    body = open(TEMPLATE, encoding="utf-8").read().replace("__DATA__", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
    os.makedirs(os.path.join(HERE, "site"), exist_ok=True)
    doc = ("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
           "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
           + body.replace("<main", "</head>\n<body>\n<main", 1) + "\n</body>\n</html>\n")
    open(os.path.join(HERE, "site", "index.html"), "w", encoding="utf-8").write(doc)
    if a.fragment:
        open(os.path.join(HERE, "site", "fragment.html"), "w", encoding="utf-8").write(body)
    print("wrote bench/site/index.html", f"({len(doc) // 1024} KB)")


if __name__ == "__main__":
    main()
