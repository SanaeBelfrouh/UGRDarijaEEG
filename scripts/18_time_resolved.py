#!/usr/bin/env python3
"""Etape 18 : decodage resolu dans le temps, fenetres glissantes de 250 ms.

    python scripts/18_time_resolved.py --epochs epochs.npz --out results

Question d'un relecteur (Jose, p. 8) : une fenetre de 2 s peut diluer un
evenement bref. Le meme decodeur est donc applique a des fenetres de 250 ms
glissant par pas de 50 ms, de 500 ms avant le go a 2 s apres.

Decodeur : base de rang commune (32 composantes, calculee sans etiquette sur
l'epoque entiere de toutes les sessions de la condition), covariance OAS de la
fenetre, espace tangent et LDA a retrecissement (``cov_tangent_lda``), sans et
avec recentrage riemannien par session, leave-one-session-out, 15 mots. Le banc
de filtres du manuscrit n'a pas de sens sur 250 ms (la bande 1-8 Hz y tient un
quart de cycle a 1 Hz) : la covariance large bande le remplace.

Statistique : permutation par clusters temporels (Maris et Oostenveld 2007). Les
etiquettes sont permutees a l'interieur de chaque session, la MEME permutation
pour toutes les fenetres, ce qui conserve la dependance entre fenetres voisines
qui se chevauchent. Le seuil de formation d'un cluster est, fenetre par fenetre,
le 95e centile de la distribution nulle ; la masse d'un cluster est la somme de
ses exces sur ce seuil, comparee au maximum des masses de chaque permutation.
Le p d'un cluster controle donc l'erreur de famille sur toutes les fenetres.

Sorties : ``decodage_temporel.csv`` (une ligne par condition x recentrage x
fenetre : exactitude equilibree, seuil de la fenetre, p non corrige, cluster) et
``decodage_temporel_clusters.csv`` (une ligne par cluster).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from darija.evaluate import evaluate_many                                       # noqa: E402
from darija.features import rank_reduce                                         # noqa: E402
from darija.riemann import covariances                                          # noqa: E402
from darija.stats import minimum_detectable_balanced_accuracy                   # noqa: E402

MODEL = "cov_tangent_lda"


def balanced_from_codes(true_codes, pred_codes, counts):
    hits = np.bincount(true_codes[true_codes == pred_codes], minlength=len(counts))
    return float(np.mean(hits / counts))


def clusters_above(values, threshold):
    """Suites de fenetres consecutives au-dessus du seuil : [(debut, fin incluse, masse)]."""
    above = values > threshold
    out, start = [], None
    for i, a in enumerate(np.append(above, False)):
        if a and start is None:
            start = i
        elif not a and start is not None:
            out.append((start, i - 1, float(np.sum(values[start:i] - threshold[start:i]))))
            start = None
    return out


def temporal_cluster_test(y, preds, groups, n_perm=1000, seed=0):
    """Exactitudes, seuils, p par fenetre et clusters, sous une permutation commune.

    ``preds`` est de forme (fenetres, essais). Renvoie (acc, seuil, p, clusters)
    avec clusters = [(debut, fin, masse, p)].
    """
    labels = sorted(np.unique(y).tolist())
    index = {v: i for i, v in enumerate(labels)}
    true_codes = np.array([index[v] for v in y])
    pred_codes = np.array([[index.get(v, -1) for v in p] for p in preds])
    counts = np.bincount(true_codes, minlength=len(labels)).astype(float)
    acc = np.array([balanced_from_codes(true_codes, pc, counts) for pc in pred_codes])

    rng = np.random.default_rng(seed)
    blocks = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    null = np.empty((n_perm, len(preds)))
    permuted = true_codes.copy()
    for k in range(n_perm):
        for b in blocks:
            permuted[b] = rng.permutation(true_codes[b])
        null[k] = [balanced_from_codes(permuted, pc, counts) for pc in pred_codes]
    threshold = np.quantile(null, 0.95, axis=0)
    p_window = (1 + (null >= acc).sum(axis=0)) / (n_perm + 1)
    null_max = np.array([max([m for _, _, m in clusters_above(row, threshold)], default=0.0)
                         for row in null])
    found = [(s, e, mass, float((1 + np.sum(null_max >= mass)) / (n_perm + 1)))
             for s, e, mass in clusters_above(acc, threshold)]
    return acc, threshold, p_window, found


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", required=True, help="archive de reference")
    p.add_argument("--out", default="results")
    p.add_argument("--conditions", nargs="+", default=("imagined", "articulated"))
    p.add_argument("--length", type=float, default=0.25, help="longueur de fenetre (s)")
    p.add_argument("--step", type=float, default=0.05, help="pas (s)")
    p.add_argument("--n-perm", type=int, default=1000)
    p.add_argument("--max-hours", type=float, default=None, help="ignore, pour run_all")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    a = np.load(args.epochs, allow_pickle=True)
    times = a["times"]
    sf = float(a["sfreq"])
    n_win = int(round(args.length * sf))
    # par indices : chaque fenetre tient entierement dans l'epoque
    first = np.arange(0, len(times) - n_win + 1, int(round(args.step * sf)))
    starts = times[first]
    rows, clusters = [], []
    for condition in args.conditions:
        m = a["cond"].astype(str) == condition
        y = a["y"][m].astype(str)
        ses = a["session"][m].astype(str)
        tix = a["trial_index"][m].astype(int)
        X = rank_reduce(a["X"][m].astype(np.float64), ses, n_components=32, scope="global")
        seuil80 = minimum_detectable_balanced_accuracy(np.unique(y, return_counts=True)[1])
        print(f"\n{condition} : {len(y)} essais, {len(starts)} fenetres de {1000 * args.length:.0f} ms",
              flush=True)
        for recenter in (False, True):
            preds, t0 = [], time.time()
            for i0 in first:
                reps = {"cov": covariances(X[:, :, i0:i0 + n_win])}
                r = evaluate_many(reps, y, ses, tix, model=MODEL, scheme="loso",
                                  contrasts=("mots15",), recenter=recenter, n_perm=1)[0]
                if not np.array_equal(np.asarray(r.y_true), y):
                    raise SystemExit("ordre des essais modifie par l'evaluation")
                preds.append(np.asarray(r.y_pred, dtype=object))
            acc, thr, pw, found = temporal_cluster_test(y, np.array(preds), ses, n_perm=args.n_perm)
            in_cluster = np.full(len(starts), np.nan)
            for s, e, mass, pc in found:
                in_cluster[s:e + 1] = pc
                clusters.append(dict(condition=condition, modele=MODEL, recentre=recenter,
                                     debut_s=float(starts[s]), fin_s=float(starts[e] + n_win / sf),
                                     fenetres=e - s + 1, masse=mass, p_cluster=pc,
                                     pic=float(acc[s:e + 1].max()),
                                     pic_centre_s=float(starts[s + np.argmax(acc[s:e + 1])] + n_win / sf / 2)))
            for i, t in enumerate(starts):
                rows.append(dict(condition=condition, modele=MODEL, recentre=recenter, schema="loso",
                                 contraste="mots15", debut_s=float(t), centre_s=float(t + n_win / sf / 2),
                                 essais=len(y), equilibree=float(acc[i]), seuil_fenetre=float(thr[i]),
                                 seuil_eq=seuil80, p_perm=float(pw[i]), p_cluster=in_cluster[i]))
            sig = [f"{starts[s]:+.2f} a {starts[e] + n_win / sf:+.2f} s (p = {pc:.4f})"
                   for s, e, _, pc in found if pc < 0.05]
            print(f"  recentre={int(recenter)} pic {100 * acc.max():.2f} % a "
                  f"{starts[acc.argmax()] + n_win / sf / 2:+.3f} s ; clusters significatifs : "
                  f"{', '.join(sig) or 'aucun'} ({time.time() - t0:.0f} s)", flush=True)
        del X
    pd.DataFrame(rows).to_csv(out / "decodage_temporel.csv", index=False)
    pd.DataFrame(clusters, columns=["condition", "modele", "recentre", "debut_s", "fin_s", "fenetres",
                                    "masse", "p_cluster", "pic", "pic_centre_s"]
                 ).to_csv(out / "decodage_temporel_clusters.csv", index=False)
    print(f"\nsorties : decodage_temporel.csv, decodage_temporel_clusters.csv dans {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
