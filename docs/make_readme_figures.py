#!/usr/bin/env python3
"""Render the README summary figures from the values reported in README.md.

These figures illustrate the current results table by table; they are not a
substitute for the pipeline figures, which ``scripts/06_figures.py`` regenerates
from the CSV files written by ``scripts/run_all.py``.

    python docs/make_readme_figures.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / "figures"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#ffffff"
BLUE, ORANGE = "#2a78d6", "#eb6834"
CHANCE, SIGNIF, DETECT = 6.67, 7.63, 8.13

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
})


def bounds(ax, horizontal=True, label=True):
    for v, name, ls in ((CHANCE, "chance 6.67 %", "-"), (SIGNIF, "significance 7.63 %", ":"),
                        (DETECT, "detectability (80 % power) 8.13 %", "--")):
        (ax.axhline if horizontal else ax.axvline)(v, color=MUTED, lw=1, ls=ls, zorder=1,
                                                   label=name if label else None)


# 1. headline: fifteen words, both conditions -------------------------------
rows = [("Articulated, within session", 43.33), ("Articulated, across sessions (LOSO)", 30.47),
        ("Imagined, across sessions (LOSO)", 8.18)]
fig, ax = plt.subplots(figsize=(7.2, 2.6))
for i, (name, v) in enumerate(rows):
    ax.barh(i, v, height=0.5, color=BLUE if i < 2 else ORANGE, zorder=2)
    ax.text(v + 0.6, i, f"{v:.2f} %", va="center", color=INK)
ax.set_yticks(range(len(rows)), [r[0] for r in rows])
ax.invert_yaxis()
bounds(ax, horizontal=False, label=False)
ax.text(CHANCE, -0.55, "chance", color=INK2, ha="center", fontsize=8)
ax.set_xlim(0, 48)
ax.set_xlabel("Balanced accuracy, 15 words (%)")
ax.grid(axis="x", color=GRID, zorder=0)
fig.tight_layout()
fig.savefig(OUT / "headline_15_words.png", dpi=160)

# 2. rank sweep ---------------------------------------------------------------
ranks = [8, 16, 24, 32, 40, 48, 56, 64]
subblock = [8.29, 7.96, 8.13, 7.46, 7.73, 7.40, 7.29, 7.07]
loso = {8: 7.23, 32: 8.18, 40: 6.90, 48: 6.29, 56: 6.96, 64: 6.73}
fig, ax = plt.subplots(figsize=(7.2, 3.4))
bounds(ax)
ax.plot(ranks, subblock, color=BLUE, lw=2, marker="o", ms=7, label="grouped by sub-block", zorder=3)
ax.plot(list(loso), list(loso.values()), color=ORANGE, lw=2, marker="s", ms=7, ls="-",
        label="leave-one-session-out", zorder=3)
ax.set_xticks(ranks)
ax.set_xlabel("Retained rank (components)")
ax.set_ylabel("Balanced accuracy (%)")
ax.set_title("Imagined speech, 15 words — accuracy vs. rank", loc="left", color=INK, fontsize=11)
ax.grid(axis="y", color=GRID, zorder=0)
ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper right")
ax.set_ylim(5.8, 9.2)
fig.tight_layout()
fig.savefig(OUT / "rank_sweep.png", dpi=160)

# 3. xDAWN asymmetry ----------------------------------------------------------
xd = [("Articulated, full montage", 16.32, 30.47), ("Articulated, jaw ring (10)", 16.19, 27.17),
      ("Articulated, no jaw/occiput (46)", 10.97, 21.55), ("Articulated, occipital (8)", 8.91, 11.97),
      ("Imagined, full montage", 7.68, 7.90)]
fig, ax = plt.subplots(figsize=(7.2, 3.0))
for i, (name, a, b) in enumerate(xd):
    ax.plot([a, b], [i, i], color=GRID, lw=3, zorder=1, solid_capstyle="round")
    ax.plot(a, i, "o", color=BLUE, ms=8, zorder=3, label="without xDAWN" if i == 0 else None)
    ax.plot(b, i, "D", color=ORANGE, ms=8, zorder=3, label="with xDAWN" if i == 0 else None)
    ax.text(max(a, b) + 0.8, i, f"{b - a:+.1f} pt" if i < 4 else "no gain (n.s.)",
            va="center", color=INK2, fontsize=9)
ax.set_yticks(range(len(xd)), [r[0] for r in xd])
ax.invert_yaxis()
ax.axvline(CHANCE, color=MUTED, lw=1)
ax.set_xlim(0, 40)
ax.set_xlabel("Balanced accuracy, 15 words, LOSO (%)")
ax.grid(axis="x", color=GRID, zorder=0)
ax.legend(frameon=False, fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(OUT / "xdawn_asymmetry.png", dpi=160)
print("written to", OUT)
