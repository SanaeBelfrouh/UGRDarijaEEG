#!/usr/bin/env python3
"""Etape 17 : desynchronisation et synchronisation (ERD/ERS) sur tout le spectre.

    python scripts/17_erd_ers.py --epochs epochs.npz --flags results/essais.csv --out results

Question d'un relecteur (Jose, p. 11) : l'analyse haute frequence (55-95 Hz) ne
dit rien des rythmes plus lents. Que montrent theta, mu, beta et gamma bas,
par rapport a l'avant-go, en articule et en imagine ?

Pour chaque essai sans clignement pendant la production (``essais.csv``), la
puissance de chaque electrode est estimee par multitaper (largeur de bande
4 Hz) dans deux fenetres : la ligne de base, les 500 ms qui precedent le go, et
la production a partir de 400 ms apres le go, apres la reponse visuelle et le
gros des clignements : 0,4-1,9 s, en trois segments de 500 ms. Ligne de base et
segments passent par le meme estimateur, et le changement est la moyenne des
trois log-rapports : sous l'hypothese nulle, chaque terme est la difference de
deux variables de meme loi, donc d'esperance nulle. Le changement est en dB, essai par essai :
la ligne de base etant celle du meme essai, la derive lente entre sessions
s'annule.

Bandes : theta 4-8 Hz, mu 8-13 Hz, beta 13-30 Hz, gamma bas 30-45 Hz (l'archive
est filtree a 45 Hz). Chaque carte (une condition x une bande) est testee par
permutation par clusters spatiaux (Maris et Oostenveld 2007) : test t a un
echantillon sur les essais, voisinage des electrodes par triangulation de leurs
positions 10-05, 2000 inversions de signe. Le p d'un cluster controle l'erreur
de famille dans sa carte. ``sessions_meme_signe`` dit dans combien des huit
sessions la moyenne de l'electrode a le signe de la moyenne globale.

Sorties : ``erders.csv`` (une ligne par condition x bande x electrode) et
``erders_clusters.csv`` (une ligne par cluster).
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from darija.preprocess import crop                                              # noqa: E402

BANDS = {"theta": (4.0, 8.0), "mu": (8.0, 13.0), "beta": (13.0, 30.0), "gamma bas": (30.0, 45.0)}
BASELINE = (-0.5, 0.0)
#: la production en trois segments de la longueur de la ligne de base : un
#: estimateur multitaper sur 0,5 s et un autre sur 1,6 s n'ont pas le meme biais
#: (sur du bruit blanc, +1 dB en theta), alors que deux segments de meme longueur
#: passent par le meme estimateur et le biais s'annule dans le rapport
PRODUCTION = ((0.4, 0.9), (0.9, 1.4), (1.4, 1.9))


def band_power(X, sfreq, bands, bandwidth=4.0):
    """(essais, bandes, canaux) : densite spectrale moyenne dans chaque bande."""
    from mne.time_frequency import psd_array_multitaper

    fmax = max(hi for _, hi in bands.values())
    psd, freqs = psd_array_multitaper(X, sfreq, fmin=1.0, fmax=min(fmax, sfreq / 2 - 1),
                                      bandwidth=bandwidth, n_jobs=1, verbose=False)
    out = np.empty((X.shape[0], len(bands), X.shape[1]))
    for b, (lo, hi) in enumerate(bands.values()):
        sel = (freqs >= lo) & (freqs < hi)
        out[:, b] = psd[..., sel].mean(axis=-1)
    return out


def adjacency_for(ch_names, sfreq):
    import mne

    info = mne.create_info(list(ch_names), sfreq, "eeg")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        info.set_montage("standard_1005", on_missing="ignore", match_case=False)
    adjacency, names = mne.channels.find_ch_adjacency(info, "eeg")
    assert list(names) == list(ch_names), "ordre des canaux modifie par MNE"
    return adjacency


def main() -> int:
    from mne.stats import permutation_cluster_1samp_test

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", required=True, help="archive de reference")
    p.add_argument("--flags", required=True, help="essais.csv du script 11")
    p.add_argument("--out", default="results")
    p.add_argument("--n-perm", type=int, default=2000)
    p.add_argument("--max-hours", type=float, default=None, help="ignore, pour run_all")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    a = np.load(args.epochs, allow_pickle=True)
    sf = float(a["sfreq"])
    ch = [str(c) for c in a["ch_names"]]
    adjacency = adjacency_for(ch, sf)
    flags = pd.read_csv(args.flags)

    rows, clusters = [], []
    for condition in ("articulated", "imagined"):
        m = np.flatnonzero(a["cond"].astype(str) == condition)
        keys = pd.DataFrame({"session": a["session"][m].astype(str),
                             "trial_index": a["trial_index"][m].astype(int), "pos": m})
        f = flags[flags.cond == condition][["session", "trial_index", "blink_production"]]
        keys = keys.merge(f, on=["session", "trial_index"], how="left")
        if keys.blink_production.isna().any():
            raise SystemExit(f"{int(keys.blink_production.isna().sum())} essais {condition} sans "
                             f"drapeau : archive et essais.csv ne viennent pas des memes XDF")
        keys = keys[~keys.blink_production.astype(bool)]
        ses = keys.session.values
        X = a["X"][keys.pos.values].astype(np.float64)
        t0 = time.time()
        # toutes les fenetres a la meme longueur, a l'echantillon pres : une
        # longueur differente change la grille de frequences, donc les raies
        # moyennees dans chaque bande, et le biais de l'estimateur
        windows = [crop(X, a["times"], *w)[0] for w in (BASELINE, *PRODUCTION)]
        n = min(w.shape[-1] for w in windows)
        base = band_power(windows[0][..., :n], sf, BANDS)
        # moyenne des log-rapports segment par segment, pas log du rapport des
        # moyennes : le log d'une moyenne de trois estimations a une autre
        # esperance que le log d'une seule (Jensen), et le biais passerait pour
        # une synchronisation
        D = np.mean([10 * np.log10(band_power(w[..., :n], sf, BANDS) / base)
                     for w in windows[1:]], axis=0)   # (essais, bandes, canaux)
        del X, windows
        print(f"\n{condition} : {len(keys)} essais sans clignement, puissance en "
              f"{time.time() - t0:.0f} s", flush=True)
        for b, band in enumerate(BANDS):
            d = D[:, b]
            T, cl, pv, _ = permutation_cluster_1samp_test(
                d, adjacency=adjacency, n_permutations=args.n_perm, tail=0, out_type="mask",
                seed=np.random.default_rng(0), verbose=False)
            mean = d.mean(axis=0)
            per_session = np.stack([d[ses == s].mean(axis=0) for s in np.unique(ses)])
            same_sign = (np.sign(per_session) == np.sign(mean)).sum(axis=0)
            p_elec = np.full(len(ch), np.nan)
            for mask, pc in zip(cl, pv):
                idx = np.flatnonzero(mask)
                p_elec[idx] = pc                          # les clusters sont disjoints
                sign = "ERS" if T[idx].sum() > 0 else "ERD"
                clusters.append(dict(condition=condition, bande=band, signe=sign,
                                     electrodes_n=len(idx), electrodes=" ".join(ch[i] for i in idx),
                                     t_somme=float(T[idx].sum()), db_moyen=float(mean[idx].mean()),
                                     p_cluster=float(pc), essais=len(d)))
            for i, name in enumerate(ch):
                rows.append(dict(condition=condition, bande=band, electrode=name, essais=len(d),
                                 db_moyen=float(mean[i]), t=float(T[i]), p_cluster=p_elec[i],
                                 dans_cluster=bool(p_elec[i] < 0.05) if np.isfinite(p_elec[i]) else False,
                                 sessions_meme_signe=int(same_sign[i])))
            sig = [c for c in clusters if c["condition"] == condition and c["bande"] == band
                   and c["p_cluster"] < 0.05]
            print(f"  {band:10s} moyenne {mean.mean():+.2f} dB (min {mean.min():+.2f}, max {mean.max():+.2f}) ; "
                  + (", ".join(f"{c['signe']} {c['electrodes_n']} el. p = {c['p_cluster']:.4f}" for c in sig)
                     or "aucun cluster significatif"), flush=True)

    pd.DataFrame(rows).to_csv(out / "erders.csv", index=False)
    cdf = pd.DataFrame(clusters, columns=["condition", "bande", "signe", "electrodes_n", "electrodes",
                                          "t_somme", "db_moyen", "p_cluster", "essais"])
    cdf.to_csv(out / "erders_clusters.csv", index=False)
    print(f"\nsorties : erders.csv, erders_clusters.csv dans {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
