#!/usr/bin/env python3
"""Etape 4 : balayage du rang, pour trancher la question des 32 composantes.

    python scripts/04_rank_sweep.py --epochs epochs.npz --condition imagined

Le manuscrit affirme que la projection sur les 32 premieres composantes « n'est
pas un detail : a rang plein le vecteur de caracteristiques atteint 6 384
dimensions et l'effet disparait ». Telle quelle, cette phrase se lit comme un
aveu : l'effet depend d'un hyperparametre, et la valeur retenue est celle qui le
produit. Un relecteur s'en saisira.

Ce script produit la donnee qui repond a l'objection, dans un sens ou dans
l'autre. Deux issues possibles, et il faut savoir laquelle avant de soumettre :

**Optimum large.** L'exactitude monte, plafonne sur une plage de rangs, puis
redescend. La reduction de rang est alors une necessite d'estimation — a rang
plein la dimension excede le nombre d'essais d'entrainement — et 32 n'est qu'un
point quelconque du plateau. L'argument est solide et se redige comme tel.

**Pic isole.** L'effet n'existe qu'a 32 et retombe des 24 ou 40. C'est alors un
artefact de selection, et le chiffre ne doit pas etre publie comme un resultat.

Le script balaye aussi les deux lectures possibles de « mean session
covariance » : une base commune a toutes les sessions (``global``) ou une base
par session (``session``). La distinction n'est pas cosmetique. Les vecteurs
propres ont un signe arbitraire et leur ordre peut permuter entre valeurs
propres voisines : avec une base par session, la composante 7 d'une session ne
correspond pas a la composante 7 d'une autre, et les termes hors diagonale des
covariances changent de signe d'une session a l'autre. Cela degrade le transfert
inter-sessions plutot que de le fabriquer, mais il faut savoir laquelle des deux
variantes a produit le chiffre qu'on publie.

Le balayage reprend la ou il s'est arrete : Kaggle coupe a 12 h.
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from darija.evaluate import evaluate_many                     # noqa: E402
from darija.features import FILTER_BANK, MANUSCRIPT_BANK      # noqa: E402
from darija.pipelines import build_representations, representation_for  # noqa: E402
from darija.preprocess import crop                            # noqa: E402
from darija.stats import holm_bonferroni                       # noqa: E402

warnings.filterwarnings("ignore")

DEFAULT_RANKS = (8, 16, 24, 32, 40, 48, 56, 64)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", default="epochs.npz")
    parser.add_argument("--condition", default="imagined",
                        choices=("imagined", "articulated"))
    parser.add_argument("--tmin", type=float, default=0.0)
    parser.add_argument("--tmax", type=float, default=2.0)
    parser.add_argument("--model", default="bankcov_tangent")
    parser.add_argument("--schemes", nargs="+", default=["loso", "pooled_blocked"])
    parser.add_argument("--contrasts", nargs="+", default=["mots15"])
    parser.add_argument("--ranks", nargs="+", type=int, default=list(DEFAULT_RANKS))
    parser.add_argument("--scopes", nargs="+", default=["global", "session"],
                        choices=("global", "session"))
    parser.add_argument("--bank", default="manuscrit", choices=("manuscrit", "large"))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-perm", type=int, default=5000)
    parser.add_argument("--max-hours", type=float, default=11.0)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--out", default="results/rank_sweep.csv")
    args = parser.parse_args()

    t_start = time.time()
    archive = np.load(args.epochs, allow_pickle=True)
    mask = archive["cond"] == args.condition
    X, times = crop(archive["X"][mask], archive["times"], args.tmin, args.tmax)
    y = archive["y"][mask]
    session = archive["session"][mask]
    trial_index = archive["trial_index"][mask]
    sub_block = archive["sub_block"][mask] if "sub_block" in archive.files else None
    if sub_block is None and "subblock" in args.schemes:
        raise SystemExit("archive sans sub_block : reconstruire avec scripts/01_build_epochs.py")
    sfreq = float(archive["sfreq"])
    bands = MANUSCRIPT_BANK if args.bank == "manuscrit" else FILTER_BANK

    print(f"{len(X)} essais {args.condition} | {X.shape[1]} canaux | "
          f"fenetre [{times[0]:.2f}, {times[-1]:.2f}] s | banc {args.bank}\n")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    done: set[tuple] = set()
    if out_path.exists() and not args.restart:
        rows = pd.read_csv(out_path).to_dict("records")
        done = {(int(r["rang"]), r["portee"], r["schema"], r["contraste"]) for r in rows}
        print(f"reprise : {len(rows)} lignes deja presentes\n")

    representation = representation_for(args.model)
    deadline = t_start + args.max_hours * 3600
    remaining = []

    for rank, scope in itertools.product(sorted(args.ranks), args.scopes):
        todo = [(s, c) for s in args.schemes for c in args.contrasts
                if (rank, scope, s, c) not in done]
        if not todo:
            print(f"  rang {rank:3d} ({scope:7s}) : deja fait, saute")
            continue
        if time.time() > deadline:
            remaining.append((rank, scope))
            continue

        t0 = time.time()
        # Le rang plein est passe explicitement plutot que converti en None :
        # avec une base commune c'est une congruence appliquee uniformement, donc
        # un no-op pour une metrique affine-invariante, ce qui en fait un
        # controle de coherence ; avec une base par session c'est une rotation
        # differente par session, configuration bien reelle qu'il faut mesurer.
        representations = build_representations(
            X, sfreq, bands=bands, include=(representation,),
            n_components=rank, groups=session, rank_scope=scope,
        )
        dims = representations[representation].shape
        build_time = time.time() - t0

        for scheme in args.schemes:
            if all((rank, scope, scheme, c) in done for c in args.contrasts):
                continue
            block = evaluate_many(
                representations, y, session, trial_index,
                model=args.model, scheme=scheme, contrasts=tuple(args.contrasts),
                n_splits=args.n_splits, n_perm=args.n_perm, sub_block=sub_block,
            )
            for result in block:
                row = result.row()
                # dimension du vecteur tangent : n_bandes x c(c+1)/2. Une
                # representation a banc de bandes est en 4D (n, bandes, c, c),
                # une covariance simple en 3D (n, c, c) : une seule bande.
                n_bands = dims[1] if len(dims) == 4 else 1
                side = dims[-1]
                row.update(rang=rank, portee=scope,
                           dim_tangente=int(n_bands * side * (side + 1) / 2))
                rows.append(row)
                print(f"  rang {rank:3d} ({scope:7s}) {scheme:15s} "
                      f"{row['contraste']:12s} : equilibree {row['equilibree']:.4f} "
                      f"(hasard {row['hasard_eq']:.4f}, seuil {row['seuil_eq']:.4f})  "
                      f"p={row['p_perm']:.4f}  dim={row['dim_tangente']}", flush=True)
            pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"    [construction {build_time:.0f}s + evaluation "
              f"{time.time() - t0 - build_time:.0f}s]", flush=True)

    table = pd.DataFrame(rows)
    if not len(table):
        print("rien a rapporter")
        return 0

    for _, block in table.groupby(["contraste", "schema", "portee"]):
        table.loc[block.index, "p_holm"] = holm_bonferroni(block["p_perm"].values)

    table.to_csv(out_path, index=False)

    print("\n" + "=" * 96)
    print("COURBE RANG / EXACTITUDE")
    print("=" * 96)
    for (contrast, scheme), block in table.groupby(["contraste", "schema"]):
        print(f"\n  --- {contrast} / {scheme} ---")
        pivot = block.pivot_table(index="rang", columns="portee", values="equilibree")
        seuil = block.seuil_eq.iloc[0]
        hasard = block.hasard_eq.iloc[0]
        print(pivot.round(4).to_string())
        print(f"  hasard {hasard:.4f} | seuil de detectabilite {seuil:.4f}")

    print("\n" + "=" * 96)
    print("VERDICT SUR LA REDUCTION DE RANG")
    print("=" * 96)
    for (contrast, scheme, scope), block in table.groupby(["contraste", "schema", "portee"]):
        block = block.sort_values("rang")
        best = block.loc[block.equilibree.idxmax()]
        above = block[block.equilibree > block.seuil_eq]
        label = f"{contrast}/{scheme}/{scope}"
        if not len(above):
            print(f"  {label:38s} aucun rang n'atteint le seuil "
                  f"(meilleur {best.equilibree:.4f} au rang {int(best.rang)})")
            continue
        ranks_above = sorted(int(r) for r in above.rang)
        contiguous = ranks_above == list(range(ranks_above[0], ranks_above[0] + len(ranks_above))) \
            or len(ranks_above) > 2
        verdict = ("PLATEAU : necessite d'estimation, le rang exact n'est pas critique"
                   if len(ranks_above) >= 3 and contiguous
                   else "PIC ISOLE : a traiter comme un artefact de selection")
        print(f"  {label:38s} rangs au-dessus du seuil {ranks_above} -> {verdict}")

    if remaining:
        print(f"\n{'!' * 88}")
        print(f"BUDGET DE {args.max_hours} h ATTEINT — rangs non calcules : {remaining}")
        print("Relancer la MEME commande : ce qui est fait sera saute.")
        print("!" * 88)

    print(f"\nresultats -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
