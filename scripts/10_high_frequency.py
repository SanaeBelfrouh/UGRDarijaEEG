#!/usr/bin/env python3
"""Etape 10 : analyse de sensibilite en bande large, parole imaginee.

    python scripts/10_high_frequency.py --input /kaggle/input --work /kaggle/working

La question. Le pipeline principal coupe a 45 Hz, pour trois raisons : le
secteur a 50 Hz, l'activite musculaire qui domine l'EEG au-dessus de 30 a
40 Hz, et le fait que les rythmes corticaux classiques sont tous en dessous.
Un relecteur peut objecter que l'information de parole, en ECoG, est surtout
dans le gamma haut (70 a 150 Hz). Ce script repond a l'objection sur la parole
IMAGINEE, la seule ou le gamma haut n'est pas d'abord du muscle.

Ce qu'il fait.
1. Une seconde archive, ``epochs_hf.npz`` : meme rejet, meme fenetre, mais
   filtree de 0,5 a 120 Hz (notch a 50 et 100 Hz) et echantillonnee a 500 Hz.
   Les sessions a 500 Hz d'origine (S006, S007) le permettent : leur limite de
   Nyquist est 250 Hz.
2. Sur cette archive, les trois modeles de la reconciliation (ceux du 8,18 %),
   sous les deux schemas et avec et sans recentrage, sur les 15 mots et sur le
   nombre de syllabes, deux fois :
   - avec le banc du manuscrit (1-40 Hz), pour isoler l'effet du changement
     d'archive ;
   - avec ce banc prolonge par 55-95 et 105-120 Hz (``--bank haute``).
3. Un bilan qui met cote a cote les resultats a 45 Hz (si les CSV du run
   principal sont dans ``--work/results``) et ceux a 120 Hz.

Lecture. Si le gamma haut porte de l'information sur le mot imagine, la
colonne « banc haute » depasse la colonne « banc manuscrit » sur la meme
archive. Sinon, les deux se valent et la coupure a 45 Hz ne cache rien.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

MODELS = ("bankcov_tangent_lda", "cov_tangent_lda", "xdawn_tangent_lda")
SCHEMES = ("subblock", "loso")
CONTRASTS = ("mots15", "syllabes")
BANKS = ("manuscrit", "haute")


def run(script: str, *args) -> int:
    cmd = [sys.executable, str(HERE / script), *map(str, args)]
    print("\n$ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd).returncode


def best_rows(path: Path, label: str):
    """Meilleure configuration par contraste, avec son seuil et ses p."""
    import pandas as pd

    if not path.exists():
        return []
    t = pd.read_csv(path)
    out = []
    for contrast in CONTRASTS:
        g = t[t.contraste == contrast]
        if not len(g):
            continue
        r = g.loc[g.equilibree.idxmax()]
        out.append(dict(source=label, contraste=contrast, modele=r.modele,
                        schema=r.schema, recentre=bool(r.recentre), n=int(r.n),
                        equilibree=float(r.equilibree), seuil=float(r.seuil_eq),
                        p=float(r.p_perm),
                        p_holm=float(r.p_holm) if "p_holm" in r else float("nan")))
    return out


def summary(results: Path) -> None:
    rows = []
    rows += best_rows(results / "reconciliation_brut.csv", "45 Hz, run principal")
    rows += [r for r in best_rows(results / "imagine_grille.csv", "45 Hz, grille")
             if r["contraste"] == "syllabes"]
    for bank in BANKS:
        rows += best_rows(results / f"hf_{bank}.csv", f"120 Hz, banc {bank}")
    if not rows:
        print("aucun resultat a resumer")
        return
    print("\n" + "=" * 96)
    print("BILAN — parole imaginee, meilleure configuration par contraste et par source")
    print("=" * 96)
    print(f"{'source':24s} {'contraste':9s} {'modele':20s} {'schema':9s} {'rec':4s} "
          f"{'n':>5s} {'exact.':>7s} {'borne':>7s} {'p':>7s} {'p Holm':>7s}")
    for r in rows:
        print(f"{r['source']:24s} {r['contraste']:9s} {r['modele']:20s} {r['schema']:9s} "
              f"{'oui' if r['recentre'] else 'non':4s} {r['n']:5d} {r['equilibree']:7.4f} "
              f"{r['seuil']:7.4f} {r['p']:7.4f} {r['p_holm']:7.4f}")
    print("\nLecture : comparer d'abord les deux lignes « 120 Hz » d'un meme contraste "
          "(meme archive, memes essais) ; la ligne « 45 Hz » situe l'ensemble.")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default="/kaggle/input", help="racine des XDF")
    p.add_argument("--work", default="/kaggle/working")
    p.add_argument("--epochs", default=None,
                   help="archive bande large deja construite (sinon --work/epochs_hf.npz)")
    p.add_argument("--sfreq", type=float, default=500.0)
    p.add_argument("--high", type=float, default=120.0)
    p.add_argument("--n-perm", type=int, default=5000)
    p.add_argument("--max-hours", type=float, default=11.0)
    p.add_argument("--summary-only", action="store_true",
                   help="n'imprimer que le bilan des CSV deja produits")
    args = p.parse_args()

    work = Path(args.work)
    results = work / "results"
    results.mkdir(parents=True, exist_ok=True)
    if args.summary_only:
        summary(results)
        return 0

    archive = Path(args.epochs) if args.epochs else work / "epochs_hf.npz"
    if archive.exists():
        print(f"{archive.name} deja presente : construction sautee")
    else:
        code = run("01_build_epochs.py", "--input", args.input, "--out", archive,
                   "--sfreq", args.sfreq, "--high", args.high)
        if code or not archive.exists():
            print("archive bande large non produite : voir le rapport ci-dessus")
            return 1

    failed = []
    for bank in BANKS:
        code = run("02_benchmark.py",
                   "--epochs", archive, "--condition", "imagined", "--recenter",
                   "--models", *MODELS, "--schemes", *SCHEMES,
                   "--contrasts", *CONTRASTS,
                   "--n-components", 32, "--bank", bank,
                   "--n-perm", args.n_perm, "--max-hours", args.max_hours,
                   "--out", results / f"hf_{bank}.csv")
        if code:
            failed.append(bank)

    summary(results)
    if failed:
        print(f"\nechec du banc : {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
