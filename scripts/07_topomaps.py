#!/usr/bin/env python3
"""Etape 7 : discriminabilite par electrode, pour les cartes topographiques.

Ce que mesure ce script. Pour chaque electrode prise SEULE, l'exactitude
equilibree d'un decodage des 15 mots a partir de sa seule puissance : quatre
bandes du banc du manuscrit par quatre sous-fenetres temporelles, soit seize
caracteristiques. Une electrode par une electrode, en validation croisee,
sur les deux conditions.

Pourquoi cette mesure et pas une autre. Les cartes topographiques publiees
montrent le plus souvent les **patrons spatiaux** d'un filtre ajuste sur toutes
les donnees. C'est une visualisation, pas une mesure : rien n'y est valide
croise, et la carte peut montrer une structure qu'aucun pli n'aurait reproduite.
Ici chaque valeur de la carte est une exactitude obtenue sur des essais non vus,
avec son propre seuil de detectabilite. La carte se lit donc comme le reste de
l'article.

Ce qu'elle doit montrer si le §4.3 dit vrai : en articule, les electrodes
mandibulaires et temporales ressortent ; en imagine, rien ne depasse le seuil
nulle part.

Le cout est faible — seize caracteristiques par electrode — mais il y a 64
electrodes par condition, d'ou quelques minutes.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from darija.cv import make_splits                                  # noqa: E402
from darija.features import MANUSCRIPT_BANK, band_power            # noqa: E402
from darija.preprocess import crop                                 # noqa: E402
from darija.stats import (balanced_accuracy, holm_bonferroni,      # noqa: E402
                          minimum_detectable_balanced_accuracy,
                          permutation_pvalue)
from darija.topo import resolve_positions                          # noqa: E402

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # noqa: E402
from sklearn.pipeline import Pipeline                                  # noqa: E402
from sklearn.preprocessing import StandardScaler                       # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", default="epochs.npz")
    parser.add_argument("--conditions", nargs="+",
                        default=["articulated", "imagined"])
    parser.add_argument("--scheme", default="loso")
    parser.add_argument("--windows", nargs="+",
                        default=["production:0.0:2.0"],
                        help="fenetres a evaluer, au format nom:tmin:tmax")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-perm", type=int, default=5000)
    parser.add_argument("--out", default="results/topo_electrodes.csv")
    args = parser.parse_args()

    archive = np.load(args.epochs, allow_pickle=True)
    X, times, sfreq = archive["X"], archive["times"], float(archive["sfreq"])
    y, cond = archive["y"], archive["cond"]
    session, trial_index = archive["session"], archive["trial_index"]
    sub_block = archive["sub_block"] if "sub_block" in archive.files else None
    ch_names = [str(c) for c in archive["ch_names"]]

    xy, garde, absents = resolve_positions(ch_names)
    if absents:
        print(f"{len(absents)} electrode(s) sans position connue, ignorees : "
              f"{', '.join(absents)}")
    if not garde:
        raise SystemExit("aucune electrode reconnue dans la table de positions")
    print(f"{len(garde)} electrodes positionnees sur {len(ch_names)}")

    # Le p d'une permutation ne peut pas descendre sous 1/(n_perm+1). Avec une
    # correction de Holm sur une electrode par test, ce plancher est multiplie
    # par leur nombre : a 300 permutations et 64 electrodes il vaut 0,21, donc
    # AUCUNE electrode ne peut survivre a la correction, si fort soit l'effet.
    # C'est la resolution des permutations qui limiterait, pas les donnees.
    plancher = len(garde) / (args.n_perm + 1)
    if plancher >= 0.05:
        besoin = int(np.ceil(len(garde) / 0.05))
        print(f"\n!!! {args.n_perm} permutations pour {len(garde)} electrodes : "
              f"le p minimal apres Holm vaut {plancher:.3f}.")
        print(f"    Aucune electrode ne pourra survivre a la correction. "
              f"Relancer avec --n-perm {besoin} au moins.\n")

    fenetres = []
    for spec in args.windows:
        nom, a, b = spec.split(":")
        fenetres.append((nom, float(a), float(b)))
    print(f"banc {MANUSCRIPT_BANK}")
    print("fenetres : " + ", ".join(f"{n} [{a:+.2f}, {b:+.2f}] s"
                                    for n, a, b in fenetres) + "\n")

    rows = []
    for fenetre, tmin, tmax in fenetres:
        Xf, _ = crop(X, times, tmin, tmax)
        for condition in args.conditions:
            mask = cond == condition
            Xc, yc = Xf[mask], y[mask]
            gc, tc = session[mask], trial_index[mask]
            sc = None if sub_block is None else sub_block[mask]
            classes = sorted(set(yc.tolist()))
            counts = [int(np.sum(yc == c)) for c in classes]
            seuil = minimum_detectable_balanced_accuracy(counts)
            splits = list(make_splits(args.scheme, gc, tc,
                                      n_splits=args.n_splits, sub_block=sc))
            print(f"{fenetre} / {condition} : {len(Xc)} essais | "
                  f"{len(classes)} classes | {len(splits)} plis | "
                  f"seuil {seuil:.4f}")

            # La puissance se calcule une fois pour tous les canaux.
            # band_power empile (bande, fenetre, canal) avec le CANAL comme
            # indice le plus rapide : les traits du canal i sont donc
            # P[:, i::n_canaux], et non une tranche contigue. Le rangement est
            # verrouille par test_band_power_layout_is_band_window_channel.
            P = band_power(Xc, sfreq, MANUSCRIPT_BANK)
            n_canaux = Xc.shape[1]
            attendu = len(MANUSCRIPT_BANK) * 4 * n_canaux
            assert P.shape[1] == attendu, (
                f"band_power renvoie {P.shape[1]} traits, {attendu} attendus : "
                f"le rangement a change et l'extraction par canal est fausse")

            t0 = time.time()
            for rang, i in enumerate(garde):
                traits = P[:, i::n_canaux]
                pred = np.empty(len(yc), dtype=object)
                vu = np.zeros(len(yc), dtype=bool)
                for train, test in splits:
                    if (len(np.unique(yc[train])) < 2
                            or len(train) <= len(classes)):
                        continue
                    modele = Pipeline([
                        ("scale", StandardScaler()),
                        ("clf", LinearDiscriminantAnalysis(solver="lsqr",
                                                           shrinkage="auto")),
                    ])
                    modele.fit(traits[train], yc[train])
                    pred[test] = modele.predict(traits[test])
                    vu[test] = True
                if not vu.any():
                    continue
                eq = balanced_accuracy(yc[vu], pred[vu], classes)
                pval, _, nulle = permutation_pvalue(
                    yc[vu], pred[vu], gc[vu], n_perm=args.n_perm, seed=0)
                rows.append(dict(
                    fenetre=fenetre, condition=condition,
                    electrode=ch_names[i], x=xy[rang, 0], y=xy[rang, 1],
                    equilibree=round(eq, 4),
                    hasard_eq=round(1.0 / len(classes), 4),
                    seuil_eq=round(seuil, 4), p_perm=round(pval, 4),
                    nulle_moy=round(nulle, 4), n=int(vu.sum())))
                if rang % 16 == 0:
                    print(f"  {rang + 1:3d}/{len(garde)}  {ch_names[i]:6s} "
                          f"{eq:.4f}  [{time.time() - t0:.0f}s]", flush=True)

    table = pd.DataFrame(rows)
    if not len(table):
        raise SystemExit("aucune electrode evaluee")
    # Holm s'applique DANS une carte : une carte = une fenetre x une condition.
    # Corriger sur l'ensemble des cartes punirait chaque carte d'avoir des
    # voisines, ce qui n'a pas de sens — ce sont des questions distinctes.
    table = table.sort_values(["fenetre", "condition"]).reset_index(drop=True)
    table["p_holm"] = np.concatenate(
        [holm_bonferroni(g.p_perm.values)
         for _, g in table.groupby(["fenetre", "condition"], sort=True)])
    table["significatif_holm"] = table.p_holm < 0.05

    print(f"\n{'=' * 78}\nPAR CONDITION\n{'=' * 78}")
    for (fenetre, condition), g in table.groupby(["fenetre", "condition"]):
        au_dessus = g[g.equilibree > g.seuil_eq]
        survivent = g[g.significatif_holm]
        print(f"\n{fenetre} / {condition} : meilleure {g.equilibree.max():.4f} "
              f"({g.loc[g.equilibree.idxmax()].electrode}), "
              f"seuil {g.seuil_eq.iloc[0]:.4f}")
        print(f"  au-dessus du seuil : {len(au_dessus)}/{len(g)} electrodes")
        print(f"  survivent a Holm   : {len(survivent)}/{len(g)}")
        if len(survivent):
            print("  " + ", ".join(
                f"{r.electrode} {r.equilibree:.3f}"
                for _, r in survivent.sort_values("equilibree",
                                                  ascending=False).head(12).iterrows()))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(f"\nresultats -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
