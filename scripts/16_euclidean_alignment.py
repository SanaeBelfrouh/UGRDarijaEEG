#!/usr/bin/env python3
"""Etape 16 : l'alignement euclidien entre sessions change-t-il le decodage ?

    python scripts/16_euclidean_alignment.py --epochs epochs.npz --out results

Question d'un relecteur (Jose, p. 8 du manuscrit annote) : aucune compensation du
decalage entre sessions autre que le recentrage riemannien n'a ete essayee.
L'alignement euclidien (He et Wu, IEEE TBME 2019) blanchit chaque session par la
moyenne arithmetique de ses covariances d'essais, R_s^{-1/2} X, avant toute
representation. Comme le recentrage riemannien, il n'utilise aucune etiquette et
voit la session de test : c'est une adaptation de domaine non supervisee.

Configuration du manuscrit : 15 mots, fenetre 0-2 s apres le go, base de rang
commune (32 composantes, calculee sur toutes les sessions de la condition, sans
etiquette), banc de bandes du manuscrit, leave-one-session-out. L'alignement
s'applique aux 32 composantes, avant le banc de filtres.

Pour chaque condition et chaque modele, trois variantes sur les memes essais :

- ``aucun`` : ni alignement ni recentrage (la reference) ;
- ``euclidien`` : alignement euclidien par session ;
- ``riemannien`` : recentrage riemannien par session (modeles a covariances).

Sorties : ``alignement.csv`` (exactitude equilibree, seuil de detectabilite,
p par permutation dans les sessions, Holm par condition) et
``alignement_apparie.csv`` (euclidien moins aucun, euclidien moins riemannien,
permutation appariee, Holm par condition).
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from darija.evaluate import evaluate_many                                       # noqa: E402
from darija.features import MANUSCRIPT_BANK, rank_reduce                        # noqa: E402
from darija.pipelines import build_representations, representation_for         # noqa: E402
from darija.preprocess import crop                                              # noqa: E402
from darija.stats import holm_bonferroni, minimum_detectable_balanced_accuracy  # noqa: E402

MODELS = ("bankcov_tangent_lda", "cov_tangent_lda", "csp_lda", "puissance_lda")
COVARIANCE = ("bankcov", "cov")


def euclidean_alignment(X: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Blanchit chaque groupe par l'inverse de la racine de sa covariance moyenne.

    La moyenne est arithmetique, sur les covariances empiriques des essais du
    groupe (He et Wu 2019). Apres alignement, la covariance moyenne de chaque
    groupe est l'identite.
    """
    X = np.asarray(X, dtype=np.float64)
    out = np.empty_like(X)
    for g in np.unique(groups):
        m = groups == g
        Xc = X[m] - X[m].mean(axis=-1, keepdims=True)
        R = np.einsum("nct,ndt->cd", Xc, Xc) / (Xc.shape[0] * (Xc.shape[-1] - 1))
        w, V = np.linalg.eigh(R)
        w = np.maximum(w, w.max() * 1e-10)
        out[m] = np.einsum("cd,ndt->nct", (V / np.sqrt(w)) @ V.T, X[m])
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", required=True, help="archive de reference")
    p.add_argument("--out", default="results")
    p.add_argument("--conditions", nargs="+", default=("imagined", "articulated"))
    p.add_argument("--n-perm", type=int, default=5000)
    p.add_argument("--max-hours", type=float, default=None, help="ignore, pour run_all")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    spec = importlib.util.spec_from_file_location("pc", HERE / "05_paired_compare.py")
    pc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pc)

    a = np.load(args.epochs, allow_pickle=True)
    sf = float(a["sfreq"])
    rows, pairs = [], []
    for condition in args.conditions:
        m = a["cond"].astype(str) == condition
        y = a["y"][m].astype(str)
        ses = a["session"][m].astype(str)
        tix = a["trial_index"][m].astype(int)
        X, _ = crop(a["X"][m].astype(np.float64), a["times"], 0.0, 2.0)
        X = rank_reduce(X, ses, n_components=32, scope="global")
        t0 = time.time()
        reps = {
            "aucun": build_representations(X, sf, bands=MANUSCRIPT_BANK,
                                           include=("bankcov", "cov", "puissance", "brut"),
                                           n_components=None),
            "euclidien": build_representations(euclidean_alignment(X, ses), sf, bands=MANUSCRIPT_BANK,
                                               include=("bankcov", "cov", "puissance", "brut"),
                                               n_components=None),
        }
        del X
        print(f"\n{condition} : {len(y)} essais, representations pretes ({time.time() - t0:.0f} s)",
              flush=True)
        preds = {}
        for model in MODELS:
            variants = [("aucun", "aucun", False), ("euclidien", "euclidien", False)]
            if representation_for(model) in COVARIANCE:
                variants.append(("riemannien", "aucun", True))
            for name, rep, recenter in variants:
                t0 = time.time()
                r = evaluate_many(reps[rep], y, ses, tix, model=model, scheme="loso",
                                  contrasts=("mots15",), recenter=recenter, n_perm=args.n_perm)[0]
                preds[(model, name)] = (np.asarray(r.y_true, dtype=object),
                                        np.asarray(r.y_pred, dtype=object))
                rows.append(dict(condition=condition, modele=model, alignement=name, schema="loso",
                                 contraste="mots15", essais=r.n_trials, equilibree=r.balanced_accuracy,
                                 hasard_eq=r.chance_balanced, seuil_eq=r.mde_balanced,
                                 p_perm=r.pvalue, nulle_moy=r.null_mean))
                print(f"  {model:20s} {name:11s} {100 * r.balanced_accuracy:6.2f} % "
                      f"(p = {r.pvalue:.4f}) {time.time() - t0:.0f} s", flush=True)
        for model in MODELS:
            yt, ea = preds[(model, "euclidien")]
            for ref in ("aucun", "riemannien"):
                if (model, ref) not in preds:
                    continue
                yr, pr = preds[(model, ref)]
                assert np.array_equal(yt, yr), "les deux variantes doivent porter sur les memes essais"
                # paired_permutation renvoie a - b : euclidien moins la reference
                diff, pv, null_mean = pc.paired_permutation(yt, ea, pr, ses, n_perm=args.n_perm)
                pairs.append(dict(condition=condition, modele=model, configuration_a="euclidien",
                                  configuration_b=ref, difference=diff, p_apparie=pv,
                                  nulle_moy=null_mean))
        del reps

    t = pd.DataFrame(rows)
    t["p_holm"] = np.nan
    pt = pd.DataFrame(pairs)
    pt["p_holm"] = np.nan
    for condition in t.condition.unique():
        sel = t.condition == condition
        t.loc[sel, "p_holm"] = holm_bonferroni(t.loc[sel, "p_perm"].values)
        sel = pt.condition == condition
        pt.loc[sel, "p_holm"] = holm_bonferroni(pt.loc[sel, "p_apparie"].values)
    t["significatif_holm"] = t.p_holm < 0.05
    t.to_csv(out / "alignement.csv", index=False)
    pt.to_csv(out / "alignement_apparie.csv", index=False)
    with pd.option_context("display.width", 200):
        print("\n" + t[["condition", "modele", "alignement", "equilibree", "seuil_eq", "p_perm", "p_holm"]]
              .round(4).to_string(index=False))
        print("\n" + pt.round(4).to_string(index=False))
    print(f"\nsorties : alignement.csv, alignement_apparie.csv dans {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
