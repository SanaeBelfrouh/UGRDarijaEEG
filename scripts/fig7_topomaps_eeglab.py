#!/usr/bin/env python3
"""Figure 7 dans le style EEGLAB (topoplot), depuis ``topo_electrodes.csv``.

    python scripts/fig7_topomaps_eeglab.py --csv results/topo_electrodes.csv \\
        --out figures/fig7-topomaps.pdf

Ce que le style EEGLAB change par rapport a ``06_figures.py`` :

- la carte couvre TOUTES les electrodes, y compris celles sous la ligne des
  oreilles (FT9, FT10, TP9, TP10) : le disque colore depasse le contour de tete,
  dessine plus petit a l'interieur ;
- l'interpolation est la spline biharmonique d'EEGLAB (``griddata`` v4,
  Sandwell 1987), lisse et exacte aux electrodes ; elle peut depasser legerement
  les valeurs mesurees entre deux electrodes (``--interp idw`` pour une
  interpolation bornee) ; un garde-fou maintient chaque point de la carte dans
  l'etendue de ses quatre electrodes les plus proches, pour que la spline
  n'invente ni pic ni creux, en particulier au bord du disque ;
- lignes de niveau, electrodes en losanges, echelle ``jet``.

``jet`` n'est pas lisible en vision des couleurs deficiente ; ``--cmap RdBu_r``
donne une echelle bleu-blanc-rouge centree sur la borne de detectabilite.

Toutes les cartes partagent une seule echelle : c'est l'ecart entre elles qui est
le resultat. Les electrodes qui survivent a la correction de Holm sont entourees.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

#: rayon du contour de tete dans les coordonnees du CSV (T7 et T8 a 1,0)
RAYON_TETE = 1.0
FENETRES = {"production": "production", "avant": "pre-production"}


def biharmonic(xy: np.ndarray, v: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Spline biharmonique (MATLAB griddata 'v4', celle de topoplot)."""
    def green(d):
        with np.errstate(divide="ignore", invalid="ignore"):
            g = d ** 2 * (np.log(d) - 1.0)
        return np.where(d > 0, g, 0.0)

    d = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
    w = np.linalg.lstsq(green(d), v, rcond=None)[0]
    dq = np.hypot(q[:, None, 0] - xy[None, :, 0], q[:, None, 1] - xy[None, :, 1])
    return green(dq) @ w


def idw(xy: np.ndarray, v: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Inverse du carre de la distance : bornee par les valeurs mesurees."""
    d2 = (q[:, None, 0] - xy[None, :, 0]) ** 2 + (q[:, None, 1] - xy[None, :, 1]) ** 2
    p = 1.0 / np.maximum(d2, 1e-9)
    return (p * v).sum(1) / p.sum(1)


def topoplot(ax, v, xy, *, cmap, norm, interp, levels, rayon, surligne=(), n=201):
    g = np.linspace(-rayon, rayon, n)
    xx, yy = np.meshgrid(g, g)
    q = np.c_[xx.ravel(), yy.ravel()]
    z = (biharmonic if interp == "biharmonic" else idw)(xy, v, q)
    # garde-fou : en chaque point, la carte reste dans l'etendue de ses quatre
    # electrodes les plus proches. Sans lui, la spline invente des extremes entre
    # les electrodes et surtout au bord du disque, ou aucune electrode n'appuie.
    from scipy.spatial import cKDTree
    _, voisins = cKDTree(xy).query(q, k=min(4, len(v)))
    z = np.clip(z, v[voisins].min(1), v[voisins].max(1)).reshape(xx.shape)
    z[np.hypot(xx, yy) > rayon] = np.nan
    im = ax.imshow(z, origin="lower", extent=(-rayon, rayon, -rayon, rayon),
                   cmap=cmap, norm=norm, interpolation="bilinear", zorder=1)
    ax.contour(xx, yy, z, levels=levels, colors="black", linewidths=0.45,
               alpha=0.7, zorder=2)
    # contour de tete (plus petit que le disque colore), nez, oreilles
    t = np.linspace(0, 2 * np.pi, 300)
    r = RAYON_TETE
    kw = dict(color="black", lw=1.6, zorder=4, solid_capstyle="round")
    ax.plot(r * np.cos(t), r * np.sin(t), **kw)
    ax.plot([-0.10 * r, 0, 0.10 * r], [0.995 * r, 1.11 * r, 0.995 * r], **kw)
    u = np.linspace(-np.pi / 2, np.pi / 2, 60)
    for s in (-1, 1):                              # oreilles en demi-ellipses
        ax.plot(s * r * (1 + 0.07 * np.cos(u)), 0.17 * r * np.sin(u), **kw)
    ax.plot(rayon * np.cos(t), rayon * np.sin(t), color="black", lw=0.6, zorder=4)
    ax.scatter(xy[:, 0], xy[:, 1], marker="D", s=9, c="black", zorder=5,
               linewidths=0)
    for i in surligne:
        ax.scatter(xy[i, 0], xy[i, 1], s=60, facecolor="none", edgecolor="black",
                   lw=1.2, zorder=6)
    ax.set(xlim=(-rayon * 1.04, rayon * 1.04), ylim=(-rayon * 1.04, rayon * 1.08))
    ax.set_aspect("equal")
    ax.axis("off")
    return im


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.colors import Normalize

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default="results/topo_electrodes.csv")
    p.add_argument("--out", default="figures/fig7-topomaps.pdf")
    p.add_argument("--cmap", default="jet",
                   help="jet (EEGLAB) ; RdBu_r pour une echelle centree sur la borne")
    p.add_argument("--interp", choices=("biharmonic", "idw"), default="biharmonic")
    p.add_argument("--levels", type=int, default=8, help="nombre de lignes de niveau")
    args = p.parse_args()

    t = pd.read_csv(args.csv)
    if "fenetre" not in t.columns:
        t = t.assign(fenetre="production")
    conds = [c for c in ("articulated", "imagined") if c in set(t.condition)]
    fens = [f for f in FENETRES if f in set(t.fenetre)]
    fens += [f for f in sorted(set(t.fenetre)) if f not in fens]
    seuil = float(t.seuil_eq.iloc[0])
    vmin, vmax = float(t.equilibree.min()), float(t.equilibree.max())
    if args.cmap.lower().startswith("rdbu"):        # centree sur la borne
        demi = max(vmax - seuil, seuil - vmin)
        norm = Normalize(seuil - demi, seuil + demi)
    else:
        norm = Normalize(vmin, vmax)
    levels = np.linspace(vmin, vmax, args.levels + 2)[1:-1]
    rayon = float(np.hypot(t.x, t.y).max()) * 1.04

    fig, axes = plt.subplots(len(fens), len(conds), squeeze=False,
                             figsize=(3.3 * len(conds) + 1.6, 3.5 * len(fens)))
    im = None
    for li, fen in enumerate(fens):
        for ci, cond in enumerate(conds):
            ax = axes[li][ci]
            g = t[(t.fenetre == fen) & (t.condition == cond)].reset_index(drop=True)
            if not len(g):
                ax.axis("off")
                continue
            sig = [i for i, s in enumerate(g.significatif_holm) if bool(s)]
            im = topoplot(ax, g.equilibree.values, g[["x", "y"]].values,
                          cmap=args.cmap, norm=norm, interp=args.interp,
                          levels=levels, rayon=rayon, surligne=sig)
            best = g.loc[g.equilibree.idxmax()]
            if li == 0:
                ax.set_title(cond, fontsize=10, pad=4)
            ax.annotate(f"max {best.equilibree:.3f} ({best.electrode})\n"
                        f"{len(sig)} electrode{'' if len(sig) == 1 else 's'} after Holm",
                        (0.5, -0.01), xycoords="axes fraction", ha="center",
                        va="top", fontsize=7.5)
            if ci == 0:
                ax.annotate(FENETRES.get(fen, fen), (-0.04, 0.5),
                            xycoords="axes fraction", rotation=90, ha="right",
                            va="center", fontsize=10)
    if im is None:
        raise SystemExit("aucune carte tracee")

    fig.subplots_adjust(left=0.06, right=0.80, top=0.94, bottom=0.08,
                        wspace=0.06, hspace=0.22)
    cax = fig.add_axes([0.885, 0.18, 0.022, 0.62])
    sm = plt.cm.ScalarMappable(norm=Normalize(vmin, vmax),
                               cmap=_portion(args.cmap, norm, vmin, vmax))
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("balanced accuracy, single electrode", fontsize=8, labelpad=2)
    cb.ax.axhline(seuil, color="black", lw=1.6, ls=(0, (3.2, 1.6)))
    cb.ax.annotate("detectability\nbound", (-0.6, seuil),
                   xycoords=("axes fraction", "data"), fontsize=7,
                   va="center", ha="right")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"ecrit : {out}   (echelle {args.cmap}, {args.interp}, "
          f"{vmin:.4f} a {vmax:.4f}, borne {seuil:.4f})")
    return 0


def _portion(nom, norm, vmin, vmax):
    """La part de l'echelle couverte par les donnees, pour la barre de couleur."""
    from matplotlib import colormaps
    from matplotlib.colors import ListedColormap
    a, b = norm(vmin), norm(vmax)
    return ListedColormap(colormaps[nom](np.linspace(a, b, 256)))


if __name__ == "__main__":
    raise SystemExit(main())
