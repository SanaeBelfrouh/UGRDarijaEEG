#!/usr/bin/env python3
"""Etape 5 : comparaison APPARIEE entre deux configurations.

    python scripts/05_paired_compare.py --predictions results/controle_emg.predictions.npz

Deux montages evalues sur les MEMES essais ne se comparent pas en regardant si
leurs intervalles de confiance se chevauchent. Ces intervalles sont marginaux :
ils ignorent que les deux configurations voient les memes essais, donc que leurs
erreurs sont correlees. Deux intervalles qui se chevauchent peuvent cacher une
difference nette, et deux intervalles disjoints peuvent exagerer une difference
fragile.

Le test correct echange les predictions des deux configurations, essai par
essai, a l'interieur de chaque session. Sous l'hypothese nulle les deux sont
interchangeables, donc cet echange laisse la difference d'exactitude inchangee
en distribution. C'est l'equivalent apparie du test de permutation employe
ailleurs dans ce paquet.

Requiert que le script 02 ait tourne avec --save-predictions.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from darija.stats import balanced_accuracy, holm_bonferroni   # noqa: E402


def paired_permutation(y_true, pred_a, pred_b, groups, n_perm: int = 10000,
                       seed: int = 0) -> tuple[float, float, float]:
    """Difference d'exactitude equilibree entre deux jeux de predictions.

    Renvoie (difference observee, p bilateral, moyenne de la nulle). La
    permutation echange pred_a et pred_b essai par essai ; les groupes servent
    seulement a verifier que les deux jeux portent bien sur les memes essais.
    """
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a, dtype=object)
    pred_b = np.asarray(pred_b, dtype=object)
    labels = sorted(np.unique(y_true).tolist())

    observed = (balanced_accuracy(y_true, pred_a, labels)
                - balanced_accuracy(y_true, pred_b, labels))

    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        swap = rng.random(len(y_true)) < 0.5
        a = np.where(swap, pred_b, pred_a)
        b = np.where(swap, pred_a, pred_b)
        null[i] = (balanced_accuracy(y_true, a, labels)
                   - balanced_accuracy(y_true, b, labels))

    pvalue = (1.0 + np.sum(np.abs(null) >= abs(observed))) / (n_perm + 1.0)
    return float(observed), float(pvalue), float(null.mean())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", required=True,
                        help="fichier .predictions.npz produit par --save-predictions")
    parser.add_argument("--n-perm", type=int, default=10000)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    store = np.load(args.predictions, allow_pickle=True)
    keys = sorted({k.rsplit("|", 1)[0] for k in store.files})
    print(f"{len(keys)} configurations dans {Path(args.predictions).name}\n")
    for k in keys:
        print(f"  {k}")

    rows = []
    for a, b in itertools.combinations(keys, 2):
        ya, yb = store[f"{a}|vrai"], store[f"{b}|vrai"]
        # deux configurations ne sont comparables en apparie que si elles portent
        # sur exactement les memes essais, dans le meme ordre
        if len(ya) != len(yb) or not np.array_equal(ya.astype(str), yb.astype(str)):
            continue
        # on ne compare que ce qui ne differe que par UNE dimension
        fa, fb = a.split("|"), b.split("|")
        differing = [i for i in range(len(fa)) if fa[i] != fb[i]]
        if len(differing) != 1:
            continue

        diff, pvalue, null_mean = paired_permutation(
            ya, store[f"{a}|pred"], store[f"{b}|pred"], ya, n_perm=args.n_perm
        )
        rows.append(dict(
            configuration_a=a, configuration_b=b,
            dimension=("modele", "schema", "contraste", "recentre", "montage")[differing[0]],
            difference=round(diff, 4), p_apparie=round(pvalue, 4),
            nulle_moy=round(null_mean, 5), n=len(ya),
        ))

    if not rows:
        print("\naucune paire comparable : il faut deux configurations sur les memes "
              "essais ne differant que par une dimension")
        return 0

    table = pd.DataFrame(rows)
    for _, block in table.groupby("dimension"):
        table.loc[block.index, "p_holm"] = holm_bonferroni(block.p_apparie.values)
    table["significatif"] = table.p_holm < 0.05

    print("\n" + "=" * 100)
    print("COMPARAISONS APPARIEES")
    print("=" * 100)
    print(table.to_string(index=False))
    print("\nLa moyenne de la nulle doit valoir ~0 : c'est ce que garantit l'echange")
    print("des predictions sous l'hypothese nulle d'interchangeabilite.")

    out = args.out or str(Path(args.predictions).with_name("paires_appariees.csv"))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(f"\nresultats -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
