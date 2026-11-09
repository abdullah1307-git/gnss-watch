"""Render a static PNG preview of a report (used for the README image).

    python scripts/make_preview.py reports/demo docs/demo_map.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

COLORS = {"low": "#2e9e6a", "medium": "#e0a526", "high": "#e5484d"}


def main(report_dir: str, out_png: str) -> None:
    geo = json.loads((Path(report_dir) / "map.geojson").read_text())
    summary = json.loads((Path(report_dir) / "summary.json").read_text())

    fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
    fig.patch.set_facecolor("#0f1419")
    ax.set_facecolor("#0b0f13")
    for f in geo["features"]:
        p, g = f["properties"], f["geometry"]
        if p["layer"] == "jamming" and p["level"] in COLORS:
            alpha = 0.25 if p["level"] == "low" else 0.75
            ax.add_patch(Polygon(g["coordinates"][0], closed=True, facecolor=COLORS[p["level"]],
                                 edgecolor="none", alpha=alpha))
        elif p["layer"] == "position_jump":
            (x0, y0), (x1, y1) = g["coordinates"]
            ax.plot([x0, x1], [y0, y1], color="#a855f7", lw=0.9, ls="--", alpha=0.85)
            ax.plot([x1], [y1], "o", color="#a855f7", ms=2)
        elif p["layer"] == "spoofing_hotspot":
            x, y = g["coordinates"]
            ax.plot([x], [y], "o", ms=22, mfc="none", mec="#a855f7", mew=2)
            ax.annotate(f"spoofing hotspot\n{p['n_aircraft']} aircraft", (x, y), xytext=(12, -28),
                        textcoords="offset points", color="#e6edf3", fontsize=9)

    s = summary["settings"]["region"]
    ax.set_xlim(s["min_lon"], s["max_lon"])
    ax.set_ylim(s["min_lat"], s["max_lat"])
    ax.set_aspect(1.7)
    ax.tick_params(colors="#8b98a5", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#26313b")
    ax.set_xlabel("longitude", color="#8b98a5")
    ax.set_ylabel("latitude", color="#8b98a5")
    ax.set_title("GNSS Watch: jamming indicator (cells) and spoofing (purple), synthetic demo",
                 color="#e6edf3", fontsize=11)
    handles = [plt.Line2D([], [], marker="s", ls="", color=c, label=k) for k, c in COLORS.items()]
    handles.append(plt.Line2D([], [], color="#a855f7", ls="--", label="impossible jump"))
    leg = ax.legend(handles=handles, loc="lower left", facecolor="#161d24", edgecolor="#26313b",
                    fontsize=8)
    for t in leg.get_texts():
        t.set_color("#e6edf3")
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, facecolor=fig.get_facecolor())
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
