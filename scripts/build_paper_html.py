#!/usr/bin/env python3
"""Build publication-ready HTML version of the full research paper."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src = ROOT / "report" / "FULL_RESEARCH_PAPER.md"
dst = ROOT / "report" / "FULL_RESEARCH_PAPER.html"

text = src.read_text(encoding="utf-8")

html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Leakage-Aware, Recall-Constrained Evaluation of Tabular Models for Cardiovascular Disease Risk Screening</title>
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<style>
  body {
    font-family: 'Times New Roman', Times, serif, -apple-system, BlinkMacSystemFont;
    line-height: 1.65;
    color: #111;
    max-width: 900px;
    margin: 40px auto;
    padding: 0 30px;
    background: #fff;
    font-size: 16px;
  }
  h1 { font-size: 2.1em; text-align: center; margin-bottom: 0.2em; color: #0d1117; }
  h2 { font-size: 1.5em; border-bottom: 1.5px solid #333; padding-bottom: 4px; margin-top: 1.8em; color: #161b22; }
  h3 { font-size: 1.2em; margin-top: 1.4em; color: #24292f; }
  .authors { text-align: center; font-style: italic; margin-bottom: 25px; color: #555; }
  .abstract-box {
    background: #f8f9fa;
    border-left: 4px solid #0969da;
    padding: 15px 20px;
    margin: 25px 0;
    border-radius: 0 4px 4px 0;
  }
  table {
    border-collapse: collapse;
    width: 100%;
    margin: 25px 0;
    font-size: 0.9em;
  }
  th, td {
    border: 1px solid #c0c0c0;
    padding: 8px 10px;
    text-align: left;
  }
  th { background-color: #f1f3f5; font-weight: bold; }
  tr:nth-child(even) { background-color: #fafbfc; }
  code {
    background: #f1f3f5;
    padding: 2px 5px;
    border-radius: 3px;
    font-family: Courier, monospace;
    font-size: 0.9em;
  }
  hr { border: 0; border-top: 1px solid #ccc; margin: 30px 0; }
  @media print {
    body { max-width: 100%; margin: 15px; font-size: 12pt; }
    h2 { page-break-after: avoid; }
    table { page-break-inside: avoid; }
  }
</style>
</head>
<body>
"""

in_table = False
for line in text.split("\n"):
    if line.startswith("# Leakage-Aware"):
        html += f"<h1>{line[2:]}</h1>\n<div class='authors'>A Collaborative 4-Member Research Implementation</div>\n"
        continue
    if line.startswith("### Abstract"):
        html += "<div class='abstract-box'><strong>Abstract—</strong>"
        continue
    if line.startswith("Recent literature has proposed"):
        html += f"{line}</div>\n"
        continue
    if line.startswith("# Section"):
        continue
    if line.startswith("# Author:"):
        continue
    if line.startswith("## "):
        html += f"<h2>{line[3:]}</h2>\n"
        continue
    if line.startswith("### "):
        html += f"<h3>{line[4:]}</h3>\n"
        continue
    if line.startswith("---"):
        html += "<hr>\n"
        continue
    if line.startswith("|") and line.endswith("|"):
        if not in_table:
            html += "<table>\n"
            in_table = True
        cells = [c.strip() for c in line.strip("|").split("|")]
        if all(re.match(r"^:?-+:?$", c) for c in cells):
            continue
        is_header = any(k in line for k in ["Architecture", "Model", "Comparison", "Metric", "Stratum", "Slice", "Category"])
        tag = "th" if is_header else "td"
        formatted = []
        for cell in cells:
            c = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", cell)
            c = re.sub(r"`(.*?)`", r"<code>\1</code>", c)
            formatted.append(f"<{tag}>{c}</{tag}>")
        html += "<tr>" + "".join(formatted) + "</tr>\n"
        continue
    else:
        if in_table:
            html += "</table>\n"
            in_table = False
        if line.strip():
            l = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", line)
            l = re.sub(r"`(.*?)`", r"<code>\1</code>", l)
            html += f"<p>{l}</p>\n"

if in_table:
    html += "</table>\n"

html += "</body>\n</html>"
dst.write_text(html, encoding="utf-8")
print(f"Exported HTML paper to {dst}")

