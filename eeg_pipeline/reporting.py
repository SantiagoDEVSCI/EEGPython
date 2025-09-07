from __future__ import annotations
import pandas as pd
from pathlib import Path

def save_metrics_csv(subject: str, metrics: dict, out_csv: str):
    k = len(metrics["coverage"])
    df = pd.DataFrame({
        "class": list(range(k)),
        "coverage": [float(x) for x in metrics["coverage"]],
        "duration_s": [float(x) for x in metrics["duration"]],
        "occurrence_hz": [float(x) for x in metrics["occurrence"]],
    })
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

def save_report_html(subject: str, gev: float, metrics: dict, out_html: str):
    Path(out_html).parent.mkdir(parents=True, exist_ok=True)
    rows = "".join(
        f"<tr><td>{i}</td><td>{float(c):.3f}</td><td>{float(d):.3f}</td><td>{float(o):.3f}</td></tr>"
        for i,(c,d,o) in enumerate(zip(metrics['coverage'], metrics['duration'], metrics['occurrence']))
    )
    html = f"""
    <html><head><meta charset='utf-8'><title>{subject} Microstates</title></head>
    <body>
      <h2>Subject: {subject}</h2>
      <p>GEV: {gev:.4f}</p>
      <table border="1" cellpadding="6" cellspacing="0">
        <thead><tr><th>Class</th><th>Coverage</th><th>Duration (s)</th><th>Occurrence (Hz)</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </body></html>
    """
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)