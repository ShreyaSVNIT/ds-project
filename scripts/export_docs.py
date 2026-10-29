#!/usr/bin/env python3
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src = ROOT / "TEAM_WORK_DIVISION.md"
dst = ROOT / "TEAM_WORK_DIVISION.html"

text = src.read_text(encoding="utf-8")

html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Cardiovascular Research - Team Work Division & Professor Questions</title>
<style>
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    line-height: 1.6;
    color: #1f2328;
    max-width: 960px;
    margin: 40px auto;
    padding: 0 24px;
    background: #ffffff;
  }
  h1, h2, h3, h4 { color: #0969da; margin-top: 1.6em; }
  h1 { border-bottom: 2px solid #d0d7de; padding-bottom: 0.3em; font-size: 2em; }
  h2 { border-bottom: 1px solid #d0d7de; padding-bottom: 0.3em; font-size: 1.5em; }
  table {
    border-collapse: collapse;
    width: 100%;
    margin: 20px 0;
    font-size: 0.92em;
  }
  th, td {
    border: 1px solid #d0d7de;
    padding: 10px 12px;
    text-align: left;
    vertical-align: top;
  }
  th { background-color: #f6f8fa; font-weight: 600; }
  tr:nth-child(even) { background-color: #fcfcfc; }
  code {
    background-color: #eff1f3;
    padding: 2px 6px;
    border-radius: 4px;
    font-family: ui-monospace, SFMono-Regular, SF Mono, Menlo, Consolas, Liberation Mono, monospace;
    font-size: 0.88em;
  }
  pre {
    background-color: #f6f8fa;
    padding: 16px;
    border-radius: 6px;
    overflow-x: auto;
    border: 1px solid #d0d7de;
  }
  blockquote {
    border-left: 4px solid #0969da;
    padding: 10px 16px;
    color: #57606a;
    background-color: #f6f8fa;
    margin: 16px 0;
    border-radius: 0 6px 6px 0;
  }
  hr { border: 0; border-top: 1px solid #d0d7de; margin: 30px 0; }
  @media print {
    body { max-width: 100%; margin: 15px; }
    h1, h2, h3 { page-break-after: avoid; }
    table { page-break-inside: avoid; }
  }
</style>
<script type="module">
  import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
  mermaid.initialize({ startOnLoad: true });
</script>
</head>
<body>
"""

in_code = False
in_table = False
in_mermaid = False

for line in text.split("\n"):
    if line.startswith("```mermaid"):
        html += '<pre class="mermaid">\n'
        in_mermaid = True
        continue
    elif in_mermaid and line.startswith("```"):
        html += "</pre>\n"
        in_mermaid = False
        continue
    elif in_mermaid:
        html += line + "\n"
        continue

    if line.startswith("```"):
        if in_code:
            html += "</code></pre>\n"
            in_code = False
        else:
            html += "<pre><code>\n"
            in_code = True
        continue
    if in_code:
        html += line + "\n"
        continue

    if line.startswith("# "):
        html += f"<h1>{line[2:]}</h1>\n"
    elif line.startswith("## "):
        html += f"<h2>{line[3:]}</h2>\n"
    elif line.startswith("### "):
        html += f"<h3>{line[4:]}</h3>\n"
    elif line.startswith("#### "):
        html += f"<h4>{line[5:]}</h4>\n"
    elif line.startswith("> "):
        html += f"<blockquote>{line[2:]}</blockquote>\n"
    elif line.startswith("---"):
        html += "<hr>\n"
    elif line.startswith("|") and line.endswith("|"):
        if not in_table:
            html += "<table>\n"
            in_table = True
        cells = [c.strip() for c in line.strip("|").split("|")]
        if all(re.match(r"^:?-+:?$", c) for c in cells):
            continue
        is_header = any(keyword in line for keyword in ["Category", "Question", "Member", "Task", "#", "Person", "Focus", "Priority"])
        tag = "th" if is_header else "td"
        formatted_cells = []
        for cell in cells:
            c = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", cell)
            c = re.sub(r"`(.*?)`", r"<code>\1</code>", c)
            formatted_cells.append(f"<{tag}>{c}</{tag}>")
        html += "<tr>" + "".join(formatted_cells) + "</tr>\n"
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
print(f"Exported to {dst}")

