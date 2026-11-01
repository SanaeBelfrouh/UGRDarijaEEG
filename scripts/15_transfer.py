#!/usr/bin/env python3
"""Etape 15 : un decodeur entraine sur la parole articulee lit-il la parole imaginee ?

    python scripts/15_transfer.py --epochs epochs.npz --out results

L'asymetrie articule / imagine (xDAWN, controle temporel) a deux lectures : une
autre source pendant l'articulation (peripherique), ou le meme signal a un
rapport signal sur bruit bien plus faible en imagine. Seul un test direct du
transfert les departage. Sur l'archive de reference, 15 mots, fenetre 0-2 s
apres le go, representations du manuscrit (rang 32, base commune), memes
modeles que les 15 mots imagines :

- ``articule -> imagine, LOSO`` : entrainement sur les essais articules des
  sept autres sessions, test sur les essais imagines de la session laissee de
  cote ;
- ``articule -> imagine, meme session`` : entrainement sur les essais articules
  d'une session, test sur ses essais imagines ;
- ``imagine -> articule, LOSO`` : le sens inverse ;
- ``imagine -> imagine, LOSO`` : la reference, le meme modele sur les memes
  essais de test ;
- ``imagine + articule -> imagine, LOSO`` : les essais articules des sept
  autres sessions ajoutes a l'entrainement de la reference. C'est la question
  pratique : la parole articulee sert-elle de donnees d'entrainement ?

Chaque schema est evalue sans et avec recentrage riemannien par session ET par
condition (adaptation de domaine non supervisee : chaque bloc est blanchi par
sa propre moyenne geometrique, sans etiquette).

La base de la reduction de rang (32 composantes) est calculee sur les seuls
essais imagines, comme dans l'etape ``mots15`` : la reference ``imagine ->
imagine`` reproduit alors exactement la configuration du manuscrit, et les
essais articules sont projetes sur la meme base. Une base calculee sur les deux
conditions serait dominee par la variance de l'articulation (muscle) et
degraderait le decodage imagine lui-meme.


Sorties : ``transfert.csv`` (exactitude equilibree, seuil de detectabilite,
p par permutation dans les sessions, Holm sur la famille) et
``transfert_apparie.csv`` (chaque schema vers l'imagine compare a la reference
sur les memes essais de test, permutation appariee, Holm).

Controle musculaire du transfert (``--montage``, ``--lowpass``, ``--tag``).
Un decodeur entraine sur l'articule apprend d'abord l'EMG des muscles
masticateurs, puissant sur l'anneau mandibulaire et au-dessus de 26 Hz. S'il
lit l'imagine, ce peut etre par une micro-activite musculaire residuelle et non
par une representation commune de la parole. Le meme test est donc rejoue :

- sans l'anneau mandibulaire ni l'occiput (``--montage sans_machoire``,
  46 canaux, reference moyenne recalculee sur ces canaux comme dans l'etape
  ``montages``). L'occiput part aussi : il porte la reponse visuelle a
  l'indice, identique dans les deux conditions, donc un autre chemin de
  transfert qui n'est pas la parole ;
- sous 26 Hz (``--lowpass 26`` : passe-bande 0,5-26 Hz, banc du manuscrit sans
  sa bande 26-40 Hz, comme ``controles_26hz``) ;
- les deux a la fois.

Sorties : ``transfert_<tag>.csv`` et ``transfert_<tag>_apparie.csv``.
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

from darija.evaluate import _recenter                                           # noqa: E402
from darija.features import MANUSCRIPT_BANK                                     # noqa: E402
from darija.montages import resolve                                             # noqa: E402
from darija.pipelines import (build_representations, make_classifier,          # noqa: E402
                              make_transform, representation_for)
from darija.preprocess import bandpass, common_average_reference, crop          # noqa: E402
from darija.riemann import covariances                                          # noqa: E402
from darija.stats import (balanced_accuracy, holm_bonferroni,                   # noqa: E402
                          minimum_detectable_balanced_accuracy, permutation_pvalue)

MODELS = ("bankcov_tangent_lda", "cov_tangent_lda", "puissance_lda")
COVARIANCE = ("bankcov", "cov")
REFERENCE = "imagine -> imagine, LOSO"


def folds(scheme, ses, cond):
    """(nom du schema, liste de (entrainement, test)) en masques booleens."""
    art, ima = cond == "articulated", cond == "imagined"
    out = []
    for s in np.unique(ses):
        here = ses == s
        if scheme == "articule -> imagine, LOSO":
            out.append((art & ~here, ima & here))
        elif scheme == "articule -> imagine, meme session":
            out.append((art & here, ima & here))
        elif scheme == "imagine -> articule, LOSO":
            out.append((ima & ~here, art & here))
        elif scheme == REFERENCE:
            out.append((ima & ~here, ima & here))
        elif scheme == "imagine + articule -> imagine, LOSO":
            out.append((~here, ima & here))
    return out


SCHEMES = ("articule -> imagine, LOSO", "articule -> imagine, meme session",
           "imagine -> articule, LOSO", REFERENCE, "imagine + articule -> imagine, LOSO")


def predict(R, y, splits, model):
    pred = np.empty(len(y), dtype=object)
    tested = np.zeros(len(y), bool)
    for train, test in splits:
        transform = make_transform(model)
        if transform is not None:
            transform.fit(R[train])
            A, B = transform.transform(R[train]), transform.transform(R[test])
        else:
            A, B = R[train], R[test]
        clf = make_classifier(model)
        clf.fit(A, y[train])
        pred[test] = clf.predict(B)
        tested |= test
    return pred, tested


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", required=True, help="archive de reference")
    p.add_argument("--out", default="results")
    p.add_argument("--n-perm", type=int, default=5000)
    p.add_argument("--montage", default="complet",
                   help="sous-montage (darija.montages), reference recalculee dessus")
    p.add_argument("--lowpass", type=float, default=None,
                   help="passe-bande 0,5-LOWPASS Hz et bandes du banc au-dessous")
    p.add_argument("--tag", default=None,
                   help="suffixe des sorties : transfert_<tag>.csv, transfert_<tag>_apparie.csv")
    p.add_argument("--max-hours", type=float, default=None, help="ignore, pour run_all")
    args = p.parse_args()
    stem = f"transfert_{args.tag}" if args.tag else "transfert"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    spec = importlib.util.spec_from_file_location("pc", HERE / "05_paired_compare.py")
    pc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pc)

    a = np.load(args.epochs, allow_pickle=True)
    y = a["y"].astype(str)
    ses = a["session"].astype(str)
    cond = a["cond"].astype(str)
    sf = float(a["sfreq"])
    X = a["X"].astype(np.float64)
    if args.montage != "complet":
        ch_names = [str(c) for c in a["ch_names"]]
        keep = resolve(ch_names, args.montage)
        # la reference moyenne de l'archive porte sur les 64 canaux : sans
        # re-reference, chaque canal retenu garderait -moyenne(64), donc l'EMG
        # mandibulaire que le montage est cense retirer
        X = common_average_reference(X[:, keep])
        print(f"montage {args.montage} : {len(keep)} canaux, reference recalculee "
              f"({', '.join(ch_names[i] for i in keep)})", flush=True)
    bank = MANUSCRIPT_BANK
    if args.lowpass is not None:
        # filtrer l'epoque entiere avant la decoupe, comme controles_26hz
        X = bandpass(X, sf, 0.5, args.lowpass)
        bank = tuple(b for b in MANUSCRIPT_BANK if b[1] <= args.lowpass)
        print(f"passe-bande 0,5-{args.lowpass:g} Hz, bandes {bank}", flush=True)
    X, _ = crop(X, a["times"], 0.0, 2.0)
    print(f"{int((cond == 'imagined').sum())} essais imagines, "
          f"{int((cond == 'articulated').sum())} articules, {len(np.unique(ses))} sessions", flush=True)
    t0 = time.time()
    # base de rang commune, calculee sur les essais imagines seuls (sans etiquette)
    _, V = np.linalg.eigh(covariances(X[cond == "imagined"]).mean(axis=0))
    X = np.einsum("kc,nct->nkt", V[:, -32:].T, X)
    reps = build_representations(X, sf, bands=bank,
                                 include=("bankcov", "cov", "puissance"), n_components=None)
    del X
    domain = np.char.add(np.char.add(ses, "|"), cond)
    print(f"representations pretes ({time.time() - t0:.0f} s)", flush=True)

    rows, preds = [], {}
    for model in MODELS:
        rep = representation_for(model)
        for recenter in ((False, True) if rep in COVARIANCE else (False,)):
            R = _recenter(reps[rep], domain) if recenter else reps[rep]
            for scheme in SCHEMES:
                t0 = time.time()
                pred, tested = predict(R, y, folds(scheme, ses, cond), model)
                yt, yp, g = y[tested], pred[tested], ses[tested]
                acc = balanced_accuracy(yt, yp, sorted(set(yt)))
                pv, _, null_mean = permutation_pvalue(yt, yp, g, n_perm=args.n_perm)
                rows.append(dict(schema=scheme, modele=model, recentre=recenter, base_rang="imagine",
                                 montage=args.montage,
                                 passe_bas=args.lowpass if args.lowpass is not None else np.nan,
                                 essais_test=int(tested.sum()), equilibree=acc, hasard=1 / len(set(yt)),
                                 seuil_eq=minimum_detectable_balanced_accuracy(Counter(yt).values()),
                                 p_perm=pv, nulle_moy=null_mean))
                preds[(scheme, model, recenter)] = (pred, tested)
                print(f"  {scheme:38s} {model:20s} rec={int(recenter)} {100 * acc:6.2f} % "
                      f"(p = {pv:.4f}) {time.time() - t0:.0f} s", flush=True)
    t = pd.DataFrame(rows)
    t["p_holm"] = holm_bonferroni(t.p_perm.values)
    t["significatif_holm"] = t.p_holm < 0.05
    t.to_csv(out / f"{stem}.csv", index=False)

    pairs = []
    for (scheme, model, recenter), (pred, tested) in preds.items():
        if scheme == REFERENCE or "-> imagine" not in scheme:
            continue
        ref, ref_tested = preds[(REFERENCE, model, recenter)]
        assert np.array_equal(tested, ref_tested), "les deux schemas doivent tester les memes essais"
        diff, pv, null_mean = pc.paired_permutation(y[tested], pred[tested], ref[tested], ses[tested],
                                                    n_perm=args.n_perm)
        pairs.append(dict(schema=scheme, reference=REFERENCE, modele=model, recentre=recenter,
                          montage=args.montage,
                          passe_bas=args.lowpass if args.lowpass is not None else np.nan,
                          difference=diff, p_apparie=pv, nulle_moy=null_mean))
    pt = pd.DataFrame(pairs)
    pt["p_holm"] = holm_bonferroni(pt.p_apparie.values)
    pt.to_csv(out / f"{stem}_apparie.csv", index=False)

    with pd.option_context("display.width", 200):
        print("\n" + t[["schema", "modele", "recentre", "equilibree", "seuil_eq", "p_perm", "p_holm"]]
              .round(4).to_string(index=False))
        print("\n" + pt.round(4).to_string(index=False))
    print(f"\nsorties : {stem}.csv, {stem}_apparie.csv dans {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
