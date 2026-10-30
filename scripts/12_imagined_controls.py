#!/usr/bin/env python3
"""Etape 12 : les resultats imagines resistent-ils aux contaminations ?

    python scripts/12_imagined_controls.py --epochs epochs.npz --flags results/essais.csv \
        --hf results/hf_55_95.npz --out results

Trois questions, avec les deux configurations du manuscrit (15 mots :
``bankcov_tangent_lda`` sans recentrage ; nombre de syllabes : ``cov_tangent_lda``
recentre), validation leave-one-session-out, sur les essais imagines de
l'archive de reference :

A. ``controles_exclusions.csv`` : retirer les essais avec un son (montee audio
   au-dessus de ``--sound-db`` dB), ceux avec un clignement pendant la
   production, ou les deux. Retirer des essais change a lui seul l'exactitude
   attendue : chaque exclusion est comparee a ``--n-random`` exclusions
   aleatoires du meme nombre d'essais dans chaque session.
B. ``controles_26hz.csv`` : les deux configurations sous 26 Hz (banc du
   manuscrit sans sa bande 26-40 Hz ; covariance du signal filtre 0,5-26 Hz).
C. ``controles_muscle.csv`` : la seule puissance 55-95 Hz (scripts/11) decode-
   t-elle le mot ? Les deux conditions, l'articule servant de controle positif ;
   ``controles_hf_hausse.csv`` : la hausse de cette puissance apres le go, par
   session et groupe d'electrodes, avec correction de Holm.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from darija.contrasts import apply_contrast                                     # noqa: E402
from darija.evaluate import evaluate_many                                       # noqa: E402
from darija.features import MANUSCRIPT_BANK                                     # noqa: E402
from darija.pipelines import build_representations                              # noqa: E402
from darija.preprocess import bandpass, crop                                    # noqa: E402
from darija.stats import (balanced_accuracy, holm_bonferroni,                   # noqa: E402
                          minimum_detectable_balanced_accuracy, permutation_pvalue)

CONFIGS = (("mots15", "bankcov_tangent_lda", False),
           ("syllabes", "cov_tangent_lda", True))
GROUPS = {
    "machoire": ("FT7", "FT8", "FT9", "FT10", "T7", "T8", "TP7", "TP8", "TP9", "TP10"),
    "central": ("Cz", "C1", "C2", "C3", "C4", "CP1", "CP2", "CP3", "CP4", "CPz"),
    "frontopolaire": ("Fp1", "Fp2", "AF7", "AF8"),
    "occipital": ("O1", "Oz", "O2", "PO3", "POz", "PO4", "PO7", "PO8"),
}


def load(epochs, flags, condition):
    a = np.load(epochs, allow_pickle=True)
    m = a["cond"].astype(str) == condition
    keys = pd.DataFrame({"session": a["session"][m].astype(str),
                         "trial_index": a["trial_index"][m].astype(int),
                         "word": a["y"][m].astype(str)})
    f = pd.read_csv(flags)
    f = f[f.cond == condition]
    merged = keys.merge(f, on=["session", "trial_index"], how="left", suffixes=("", "_f"))
    bad = merged.word_f.isna() | (merged.word != merged.word_f)
    if bad.any():
        raise SystemExit(f"{int(bad.sum())} essais de l'archive sans drapeau ou avec un autre "
                         f"mot : archive et essais.csv ne viennent pas des memes XDF")
    return a, m, merged


def run(reps, y, ses, tix, keep, model, contrast, recenter, n_perm):
    sub = {k: v[keep] for k, v in reps.items()}
    r = evaluate_many(sub, y[keep], ses[keep], tix[keep], model=model, scheme="loso",
                      contrasts=(contrast,), recenter=recenter, n_perm=n_perm)[0]
    return dict(essais=r.n_trials, equilibree=r.balanced_accuracy, p=r.pvalue,
                seuil_eq=r.mde_balanced)


def random_like(excluded, ses, seed):
    rng = np.random.default_rng(seed)
    keep = np.ones(len(excluded), bool)
    for s in np.unique(ses):
        idx = np.flatnonzero(ses == s)
        k = int(excluded[idx].sum())
        if k:
            keep[rng.choice(idx, k, replace=False)] = False
    return keep


def loso_lda(Z, labels, groups):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    pred = np.empty(len(labels), dtype=object)
    for s in np.unique(groups):
        te = groups == s
        clf = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
        clf.fit(Z[~te], labels[~te])
        pred[te] = clf.predict(Z[te])
    return pred


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", required=True, help="archive de reference")
    p.add_argument("--flags", required=True, help="essais.csv du script 11")
    p.add_argument("--hf", required=True, help="hf_55_95.npz du script 11")
    p.add_argument("--out", default="results")
    p.add_argument("--sound-db", type=float, default=10.0)
    p.add_argument("--n-perm", type=int, default=5000)
    p.add_argument("--n-random", type=int, default=5)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ A et B
    a, m, merged = load(args.epochs, args.flags, "imagined")
    X_full = a["X"][m].astype(np.float64)
    times, sf = a["times"], float(a["sfreq"])
    y = merged.word.values.astype(str)
    ses = merged.session.values.astype(str)
    tix = merged.trial_index.values.astype(int)
    sound = (merged.rise_db > args.sound_db).values
    blink = merged.blink_production.astype(bool).values
    print(f"{len(y)} essais imagines ; son > {args.sound_db:g} dB : {sound.sum()} ; "
          f"clignement en production : {blink.sum()} ; l'un ou l'autre : {(sound | blink).sum()}",
          flush=True)

    X, _ = crop(X_full, times, 0.0, 2.0)
    reps = build_representations(X, sf, bands=MANUSCRIPT_BANK, include=("bankcov", "cov"),
                                 n_components=32, groups=ses, rank_scope="global")
    del X
    exclusions = {"aucune": np.zeros(len(y), bool), "son": sound, "clignement": blink,
                  "son ou clignement": sound | blink}
    rows = []
    for contrast, model, recenter in CONFIGS:
        for name, ex in exclusions.items():
            t0 = time.time()
            r = run(reps, y, ses, tix, ~ex, model, contrast, recenter, args.n_perm)
            row = dict(contraste=contrast, modele=model, recentre=recenter, exclusion=name,
                       retires=int(ex.sum()), **r)
            if ex.any():
                accs = [run(reps, y, ses, tix, random_like(ex, ses, k), model, contrast,
                            recenter, 50)["equilibree"] for k in range(args.n_random)]
                row.update(aleatoire_moyenne=float(np.mean(accs)), aleatoire_min=float(np.min(accs)),
                           aleatoire_max=float(np.max(accs)))
            rows.append(row)
            print(f"  A {contrast:9s} {name:18s} {100 * r['equilibree']:.2f} % (p = {r['p']:.4f}) "
                  f"{time.time() - t0:.0f} s", flush=True)
    pd.DataFrame(rows).to_csv(out / "controles_exclusions.csv", index=False)

    low_bank = tuple(b for b in MANUSCRIPT_BANK if b[1] <= 26.0)
    X_low, _ = crop(bandpass(X_full, sf, 0.5, 26.0), times, 0.0, 2.0)
    reps_low = build_representations(X_low, sf, bands=low_bank, include=("bankcov", "cov"),
                                     n_components=32, groups=ses, rank_scope="global")
    del X_low, X_full
    rows = []
    for contrast, model, recenter in CONFIGS:
        for name, keep in (("tous", np.ones(len(y), bool)), ("sans son ni clignement", ~(sound | blink))):
            r = run(reps_low, y, ses, tix, keep, model, contrast, recenter, args.n_perm)
            rows.append(dict(contraste=contrast, modele=model, recentre=recenter, bandes="< 26 Hz",
                             essais_retenus=name, **r))
            print(f"  B {contrast:9s} < 26 Hz {name:24s} {100 * r['equilibree']:.2f} % (p = {r['p']:.4f})",
                  flush=True)
    pd.DataFrame(rows).to_csv(out / "controles_26hz.csv", index=False)

    # ------------------------------------------------------------ C
    hf = np.load(args.hf, allow_pickle=True)
    ch = [str(c) for c in hf["ch_names"]]
    hf_keys = pd.DataFrame({"session": hf["session"].astype(str), "trial_index": hf["trial_index"].astype(int)})
    hf_keys["pos"] = np.arange(len(hf_keys))
    jaw = [ch.index(c) for c in GROUPS["machoire"] if c in ch]
    sets = {"machoire, moyenne (4)": lambda F: F[:, jaw].mean(axis=1),
            "machoire, par electrode (40)": lambda F: F[:, jaw].reshape(len(F), -1),
            "64 electrodes (256)": lambda F: F.reshape(len(F), -1)}
    rows, rise_rows = [], []
    for condition in ("imagined", "articulated"):
        _, _, mc = load(args.epochs, args.flags, condition)
        mc = mc.merge(hf_keys, on=["session", "trial_index"])
        mc = mc[mc.hf_complete.astype(bool)]
        F = hf["F"][mc.pos.values].astype(np.float64)
        words, groups = mc.word.values.astype(str), mc.session.values.astype(str)
        for contrast in ("mots15", "syllabes"):
            labels, cmask, _ = apply_contrast(contrast, words)
            labels = np.asarray(labels, dtype=object)
            g = groups[cmask]
            for set_name, fn in sets.items():
                pred = loso_lda(fn(F[cmask]), labels, g)
                acc = balanced_accuracy(labels, pred, sorted(set(labels)))
                pv, _, _ = permutation_pvalue(labels, pred, g, n_perm=args.n_perm)
                rows.append(dict(condition=condition, contraste=contrast, caracteristiques=set_name,
                                 essais=len(labels), equilibree=acc, hasard=1 / len(set(labels)),
                                 seuil_eq=minimum_detectable_balanced_accuracy(Counter(labels).values()),
                                 p=pv))
                print(f"  C {condition:12s} {contrast:9s} {set_name:30s} {100 * acc:6.2f} % (p = {pv:.4f})",
                      flush=True)
        rise = F.mean(axis=2)                                   # 0 a 2 s apres le go
        for s in np.unique(groups):
            for gname, chans in GROUPS.items():
                idx = [ch.index(c) for c in chans if c in ch]
                v = rise[groups == s][:, idx].mean(axis=1)
                t, pv = sstats.ttest_1samp(v, 0.0)
                rise_rows.append(dict(condition=condition, session=s, groupe=gname, essais=len(v),
                                      hausse_db=float(v.mean()), t=float(t), p=float(pv)))
    pd.DataFrame(rows).to_csv(out / "controles_muscle.csv", index=False)
    rise = pd.DataFrame(rise_rows)
    rise["p_holm"] = np.nan
    for condition in rise.condition.unique():
        sel = rise.condition == condition
        rise.loc[sel, "p_holm"] = holm_bonferroni(rise.loc[sel, "p"].values)
    rise.to_csv(out / "controles_hf_hausse.csv", index=False)
    print("\n" + rise.pivot_table(index=["condition", "session"], columns="groupe",
                                  values="hausse_db").round(3).to_string())
    print(f"\nsorties : controles_exclusions.csv, controles_26hz.csv, controles_muscle.csv, "
          f"controles_hf_hausse.csv dans {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
