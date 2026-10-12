#!/usr/bin/env python3
"""Etape 9 : reecoute des essais imagines dont le micro a capte un son.

    python scripts/09_audio_review.py --input /kaggle/input --out results/audio_review

L'etape 8 repere les essais imagines dont le niveau sonore, pendant la fenetre de
production, monte au-dessus du 5e centile des essais articules. Le niveau seul ne
dit pas si c'est une vocalisation ou un autre bruit (toux, choc, bruit de la
salle). Ce script donne de quoi trancher :

- un fichier WAV par essai suspect, de 0,5 s avant le go a 0,5 s apres la fin de
  la fenetre de production, normalise pour l'ecoute ; et pour comparaison, dans la
  meme session, le meme mot articule et un essai imagine silencieux ;
- ``audio_review.csv`` : pour chaque essai suspect, sa position dans la session et
  ses traits acoustiques (duree du son, voisement, centroide spectral, instant du
  premier son apres le go), a comparer a ceux des essais articules ;
- ``audio_review.zip`` : les WAV et le CSV, a telecharger et a ecouter.

Seuls les flux audio et marqueurs sont lus : l'EEG n'est pas charge.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from darija.io_xdf import MARKER_STREAM, discover_xdf          # noqa: E402
from darija.validation import (                                # noqa: E402
    acoustic_features, audio_levels, trials_from_markers,
)


def audio_and_markers(streams):
    audio = markers = None
    for s in streams:
        info = s["info"]
        name = str(info["name"][0])
        kind = str(info.get("type", [""])[0]).lower()
        if name == MARKER_STREAM:
            markers = s
        elif kind == "audio" or "audio" in name.lower():
            audio = s
    return audio, markers


def segment(audio, a: float, b: float) -> np.ndarray:
    x = np.asarray(audio["time_series"], dtype=np.float64)
    if x.ndim == 2:
        x = x.mean(axis=1)
    t = np.asarray(audio["time_stamps"], float)
    i, j = np.searchsorted(t, [a, b])
    return x[i:j]


def session_table(streams, ses: str):
    """Niveau et traits acoustiques de chaque essai d'une session."""
    audio, markers = audio_and_markers(streams)
    if audio is None or markers is None:
        raise RuntimeError(f"{ses} : flux audio ou marqueurs introuvable")
    trials = trials_from_markers(markers)
    rows = audio_levels(audio, trials)
    sf = float(audio["info"]["nominal_srate"][0])
    for k, (tr, r) in enumerate(zip(trials, rows)):
        r.update(session=ses, trial=k, go=tr.go, end=tr.end)
        r.update(acoustic_features(segment(audio, tr.go, tr.end), sf, r["floor_db"]))
    return rows, audio, sf


def write_wav(path: Path, x: np.ndarray, sf: float) -> float:
    from scipy.io import wavfile

    peak = float(np.max(np.abs(x))) or 1.0
    wavfile.write(str(path), int(round(sf)), (0.9 * x / peak).astype(np.float32))
    return 0.9 / peak


def main() -> int:
    import pandas as pd
    import pyxdf

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default="/kaggle/input", help="racine des XDF")
    p.add_argument("--out", default="results/audio_review")
    p.add_argument("--percentile", type=float, default=5.0,
                   help="seuil : ce centile de la montee des essais articules")
    args = p.parse_args()
    out = Path(args.out)
    (out / "wav").mkdir(parents=True, exist_ok=True)
    sessions = sorted(discover_xdf(args.input).items())
    if not sessions:
        raise SystemExit(f"aucun XDF sous {args.input}")
    sel = [{"name": MARKER_STREAM}, {"type": "Audio"}]

    def load(path):
        """Audio et marqueurs seuls ; tout le fichier si le type audio differe."""
        streams, _ = pyxdf.load_xdf(str(path), select_streams=sel)
        if audio_and_markers(streams)[0] is None:
            streams, _ = pyxdf.load_xdf(str(path))
        return streams

    # passe 1 : niveaux et traits de tous les essais, pour fixer le seuil
    tables = {}
    for ses, path in sessions:
        print(f"  passe 1  {ses}", flush=True)
        streams = load(path)
        rows, _, _ = session_table(streams, ses)
        tables[ses] = rows
        del streams
    allr = pd.DataFrame([r for rows in tables.values() for r in rows])
    art = allr[allr.cond == "articulated"]
    seuil = float(np.percentile(art.rise_db, args.percentile))
    sus = allr[(allr.cond == "imagined") & (allr.rise_db > seuil)]
    print(f"\nseuil (centile {args.percentile:g} des articules) : {seuil:+.1f} dB ; "
          f"{len(sus)} essais imagines au-dessus\n")

    # passe 2 : extraction des WAV, session par session
    keep = []
    for ses, path in sessions:
        s_sus = sus[sus.session == ses]
        if not len(s_sus):
            continue
        streams = load(path)
        audio, _ = audio_and_markers(streams)
        sf = float(audio["info"]["nominal_srate"][0])
        here = allr[allr.session == ses]
        quiet = here[here.cond == "imagined"].iloc[
            (here[here.cond == "imagined"].rise_db.abs()).argsort().iloc[:1]]
        for _, q in quiet.iterrows():
            write_wav(out / "wav" / f"{ses}_REF-silent-imagined_t{q.trial:03d}_{q.word}.wav",
                      segment(audio, q.go - 0.5, q.end + 0.5), sf)
        for _, r in s_sus.iterrows():
            name = f"{ses}_SUSPECT_t{r.trial:03d}_{r.word}_{r.rise_db:+.0f}dB.wav"
            gain = write_wav(out / "wav" / name, segment(audio, r.go - 0.5, r.end + 0.5), sf)
            same = here[(here.cond == "articulated") & (here.word == r.word)]
            if len(same):
                ref = same.iloc[(same.rise_db - same.rise_db.median()).abs().argsort().iloc[:1]].iloc[0]
                write_wav(out / "wav" / f"{ses}_REF-articulated_t{ref.trial:03d}_{ref.word}.wav",
                          segment(audio, ref.go - 0.5, ref.end + 0.5), sf)
            keep.append(dict(r) | dict(wav=name, gain=gain))
        del streams

    if not keep:
        print("aucun essai imagine au-dessus du seuil : rien a reecouter")
        return 0
    cols = ["session", "trial", "word", "rise_db", "peak_rise_db", "loud_s",
            "first_event_s", "voiced_frac", "centroid_hz", "wav"]
    rev = pd.DataFrame(keep)[cols].sort_values(["session", "trial"])
    rev.to_csv(out / "audio_review.csv", index=False)

    def q(series):
        v = series.dropna()
        return f"mediane {v.median():.2f} [{v.quantile(.05):.2f} a {v.quantile(.95):.2f}]"
    print("traits acoustiques, articules (mediane [5e a 95e centile]) :")
    for c in ("loud_s", "first_event_s", "voiced_frac", "centroid_hz"):
        print(f"  {c:14s} {q(art[c])}")
    print("\nessais imagines suspects :")
    print(rev.drop(columns="wav").round(2).to_string(index=False))
    with zipfile.ZipFile(out.parent / "audio_review.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted((out / "wav").glob("*.wav")):
            z.write(f, f"audio_review/{f.name}")
        z.write(out / "audio_review.csv", "audio_review/audio_review.csv")
    print(f"\na ecouter : {out.parent / 'audio_review.zip'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
