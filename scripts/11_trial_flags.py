#!/usr/bin/env python3
"""Etape 11 : ce que chaque essai contient en dehors de l'EEG decode.

    python scripts/11_trial_flags.py --input /kaggle/input --out results

Lit chaque XDF une seule fois et ecrit, pour chaque essai, trois informations
dont les controles de la parole imaginee ont besoin (scripts/12) :

- ``essais.csv`` : une ligne par essai, avec la session, le numero d'essai
  (``trial_index``, le meme que dans les archives d'epoques), la condition, le
  mot, la montee du niveau audio pendant la production par rapport a la
  fixation du meme essai (``rise_db``), et la presence d'un clignement pendant
  la production (``blink_production``) ;
- ``hf_55_95.npz`` : la puissance 55-95 Hz de chaque electrode, en dB par
  rapport aux 500 ms precedant le go, dans quatre fenetres de 500 ms apres le
  go. A ces frequences l'EEG de scalp est domine par le muscle : ce sont des
  caracteristiques musculaires, pas cerebrales.

Le numero d'essai est calcule exactement comme dans ``darija.io_xdf`` : le rang
chronologique, dans la session, des marqueurs ``{condition}_start`` dont
l'epoque tient dans l'enregistrement. Le script 12 verifie, essai par essai, que
la condition et le mot coincident avec l'archive.

Clignements : detectes comme dans le paragraphe *Ocular activity* du manuscrit,
sur la moyenne de Fp1 et Fp2 moins la moyenne de toutes les electrodes (filtree
1-10 Hz, signal continu filtre 0,5-30 Hz), seuil mediane + 6 MAD et au moins
25 uV, pics espaces d'au moins 300 ms et larges d'au moins 40 ms.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import next_fast_len

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from darija.io_xdf import CONDITIONS, DEFAULT_TMAX, DEFAULT_TMIN, discover_xdf  # noqa: E402
from darija.preprocess import filter_continuous                                 # noqa: E402
from darija.validation import channel_names, find_streams, trials_from_markers  # noqa: E402

#: fenetres de puissance 55-95 Hz apres le go ; reference : les 500 ms avant le go
HF_WINDOWS = ((0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0))
HF_BASELINE = 0.5


def epoch_index(markers, stamps, sfreq) -> dict[float, tuple[int, str, str]]:
    """go (horloge LSL) -> (trial_index, condition, mot), comme darija.io_xdf.

    Seuls les essais dont l'epoque [-0,5 ; 2,0] s tient dans l'enregistrement
    recoivent un numero, exactement comme a la construction des archives.
    """
    n_samples = int(round((DEFAULT_TMAX - DEFAULT_TMIN) * sfreq))
    vals = [str(v[0] if isinstance(v, (list, np.ndarray)) else v)
            for v in markers["time_series"]]
    out, k = {}, 0
    for v, t in zip(vals, np.asarray(markers["time_stamps"], float)):
        stage, _, word = v.partition(":")
        for cond in CONDITIONS:
            if stage != f"{cond}_start":
                continue
            start = int(np.searchsorted(stamps, t + DEFAULT_TMIN))
            if start >= 0 and start + n_samples <= len(stamps):
                out[float(t)] = (k, cond, word or None)
                k += 1
            break
    return out


def detect_blinks(x, names, sfreq) -> np.ndarray:
    fp = [names.index(c) for c in ("Fp1", "Fp2") if c in names] or \
         [names.index(c) for c in ("AF7", "AF8") if c in names]
    proxy = x[fp].mean(axis=0) - x.mean(axis=0)
    proxy = signal.sosfiltfilt(signal.butter(4, [1.0, 10.0], btype="band", fs=sfreq,
                                             output="sos"), proxy)
    # le signe du clignement depend de la reference : on le prend tel qu'il domine
    if np.percentile(proxy, 99.9) < -np.percentile(proxy, 0.1):
        proxy = -proxy
    med = np.median(proxy)
    thr = max(med + 6 * 1.4826 * np.median(np.abs(proxy - med)), 25.0)
    peaks, _ = signal.find_peaks(proxy, height=thr, distance=int(0.3 * sfreq),
                                 width=int(0.04 * sfreq))
    return peaks


def hf_power(data, sfreq, stamps, gos) -> np.ndarray:
    """(essais, electrodes, fenetres) : puissance 55-95 Hz en dB, reference pre-go."""
    b50, a50 = signal.iirnotch(50.0, 30.0, sfreq)
    sos = signal.butter(4, [55.0, 95.0], btype="band", fs=sfreq, output="sos")
    hf = np.empty(data.shape, np.float32)
    for k in range(len(data)):
        x = data[k].astype(np.float64)
        hf[k] = signal.sosfiltfilt(sos, signal.filtfilt(b50, a50, x - x.mean()))
    hf -= hf.mean(axis=0, keepdims=True)                     # reference moyenne commune
    n, n_fft = hf.shape[1], next_fast_len(hf.shape[1])
    starts = np.searchsorted(stamps, np.asarray(gos) - HF_BASELINE)
    n_ep = int(round((HF_BASELINE + HF_WINDOWS[-1][1]) * sfreq))
    base = int(round(HF_BASELINE * sfreq))
    win = [(int(round((a + HF_BASELINE) * sfreq)), int(round((b + HF_BASELINE) * sfreq)))
           for a, b in HF_WINDOWS]
    F = np.full((len(gos), len(hf), len(HF_WINDOWS)), np.nan, np.float32)
    ok = (starts >= 0) & (starts + n_ep <= n)
    for k in range(len(hf)):
        env = np.abs(signal.hilbert(hf[k].astype(np.float64), N=n_fft)[:n]) ** 2
        for i in np.flatnonzero(ok):
            e = env[starts[i]:starts[i] + n_ep]
            ref = 10 * np.log10(e[:base].mean() + 1e-20)
            F[i, k] = [10 * np.log10(e[a:b].mean() + 1e-20) - ref for a, b in win]
    return F


def audio_rise(audio, trials) -> list[float]:
    """Montee du niveau RMS (dB) de la production par rapport a la fixation."""
    if audio is None:
        return [np.nan] * len(trials)
    x = np.asarray(audio["time_series"], dtype=np.float64)
    if x.ndim == 2:
        x = x.mean(axis=1)
    t = np.asarray(audio["time_stamps"], float)
    sfreq = float(audio["info"]["nominal_srate"][0])
    out = []
    for tr in trials:
        i0, i1, j0, j1 = np.searchsorted(t, [tr.fixation, tr.cue, tr.go, tr.end])
        fix, prod = x[i0:i1], x[j0:j1]
        if len(fix) < 0.2 * sfreq or len(prod) < 0.2 * sfreq:
            out.append(np.nan)
            continue
        out.append(20 * np.log10(max(np.sqrt(np.mean(prod ** 2)), 1e-12))
                   - 20 * np.log10(max(np.sqrt(np.mean(fix ** 2)), 1e-12)))
    return out


def main() -> int:
    import pyxdf

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="/kaggle/input", help="racine des XDF")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sessions = discover_xdf(args.input)
    if not sessions:
        raise SystemExit(f"aucun XDF sous {args.input}")
    rows, feats, ch_ref = [], [], None
    for ses, path in sorted(sessions.items()):
        t0 = time.time()
        streams, _ = pyxdf.load_xdf(str(path), dejitter_timestamps=True)
        eeg, markers, audio = find_streams(streams)
        names = channel_names(eeg)
        if ch_ref is None:
            ch_ref = names
        elif names != ch_ref:
            raise SystemExit(f"{ses} : montage different des sessions precedentes")
        sfreq = float(eeg["info"]["nominal_srate"][0])
        stamps = np.asarray(eeg["time_stamps"], float)
        data = np.asarray(eeg["time_series"], dtype=np.float32).T
        index = epoch_index(markers, stamps, sfreq)
        trials = [tr for tr in trials_from_markers(markers) if float(tr.go) in index]
        rises = audio_rise(audio, trials)
        del streams

        peaks_t = stamps[detect_blinks(filter_continuous(data, sfreq, low=0.5, high=30.0),
                                       names, sfreq)]
        F = hf_power(data, sfreq, stamps, [tr.go for tr in trials])
        del data

        for tr, rise, f in zip(trials, rises, F):
            k, cond, word = index[float(tr.go)]
            if cond != tr.cond or word != tr.word:
                raise SystemExit(f"{ses} essai {k} : marqueurs incoherents "
                                 f"({cond}/{word} contre {tr.cond}/{tr.word})")
            rows.append(dict(session=ses, trial_index=k, cond=cond, word=word, go_lsl=tr.go,
                             rise_db=rise,
                             blink_production=bool(np.any((peaks_t >= tr.go) & (peaks_t <= tr.end))),
                             hf_complete=bool(np.isfinite(f).all())))
        feats.append(F)
        n = len(trials)
        im = [r for r in rows[-n:] if r["cond"] == "imagined"]
        print(f"  {ses} : {n} essais, {len(peaks_t)} clignements, "
              f"{sum(r['blink_production'] for r in im)} imagines avec clignement en production, "
              f"{sum((r['rise_db'] or 0) > 10 for r in im)} imagines a plus de 10 dB "
              f"({time.time() - t0:.0f} s)", flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(out / "essais.csv", index=False)
    np.savez_compressed(out / "hf_55_95.npz", F=np.concatenate(feats),
                        session=table.session.values.astype(str),
                        trial_index=table.trial_index.values.astype(int),
                        cond=table.cond.values.astype(str), word=table.word.values.astype(str),
                        ch_names=np.asarray(ch_ref), windows=np.asarray(HF_WINDOWS))
    im = table[table.cond == "imagined"]
    print(f"\n{len(table)} essais ; imagines avec clignement en production : "
          f"{int(im.blink_production.sum())} ; a plus de 10 dB : {int((im.rise_db > 10).sum())}")
    print(f"sorties : {out / 'essais.csv'}, {out / 'hf_55_95.npz'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
