#!/usr/bin/env python3
"""Etape 8 : validation technique du corpus, independante de tout decodage.

    python scripts/08_technical_validation.py --input /kaggle/input --out results

Lit chaque XDF une seule fois, une session a la fois, et produit :

- ``validation_sessions.csv`` : une ligne par session (frequence d'echantillonnage,
  pic alpha, bruit secteur, canaux hors norme, potentiel evoque visuel,
  niveaux audio) ;
- ``validation_audio.csv`` : une ligne par essai (plancher de bruit pendant la
  fixation, niveau pendant la production, montee, debut de parole) ;
- ``figures/fig8-technical-validation.pdf`` : spectres par session, potentiel
  evoque occipital par session, montee audio par condition, debut de parole.

Le bilan imprime en fin d'execution donne les chiffres a reporter dans la
section Technical Validation du manuscrit.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from darija.io_xdf import discover_xdf                       # noqa: E402
from darija.preprocess import filter_continuous               # noqa: E402
from darija.validation import (                               # noqa: E402
    audio_levels, channel_names, cue_erp, find_streams, spectrum,
    trials_from_markers,
)

COULEURS = {"articulated": "#2a78d6", "imagined": "#eb6834", "repere": "#8a8984"}


def analyse_session(streams, session: str) -> tuple[dict, list[dict], dict]:
    eeg, markers, audio = find_streams(streams)
    sfreq = float(eeg["info"]["nominal_srate"][0])
    names = channel_names(eeg)
    data = np.asarray(eeg["time_series"], dtype=np.float32).T
    stamps = np.asarray(eeg["time_stamps"], float)
    trials = trials_from_markers(markers)

    spec = spectrum(data, sfreq, names)
    filt = filter_continuous(data, sfreq, low=0.5, high=30.0)
    erp = cue_erp(filt, stamps, sfreq, names, [t.cue for t in trials])
    del filt

    row = dict(session=session, sfreq_hz=sfreq, n_channels=len(names),
               n_trials=len(trials), alpha_peak_hz=spec.alpha_peak_hz,
               line_noise_db=round(spec.line_noise_db, 2),
               n_outlier_channels=len(spec.outlier_channels),
               outlier_channels=" ".join(spec.outlier_channels),
               erp_peak_latency_ms=round(1000 * erp.peak_latency_s, 1),
               erp_p1_latency_ms=round(1000 * erp.p1_latency_s, 1),
               erp_n1_latency_ms=round(1000 * erp.n1_latency_s, 1),
               erp_peak_gfp_ratio=round(erp.peak_gfp_over_baseline, 2),
               erp_split_half_r=round(erp.split_half_r, 3),
               audio_stream=audio is not None)
    audio_rows = []
    if audio is not None:
        audio_rows = audio_levels(audio, trials)
        for r in audio_rows:
            r["session"] = session
        for cond in ("imagined", "articulated"):
            rise = [r["rise_db"] for r in audio_rows if r["cond"] == cond]
            row[f"{cond}_rise_db_median"] = round(float(np.median(rise)), 2) if rise else np.nan
    curves = dict(freqs=spec.freqs, psd=spec.median_psd, times=erp.times,
                  erp=erp.occipital_mean, sfreq=sfreq)
    return row, audio_rows, curves


def figure(curves: dict, audio, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    fig, ax = plt.subplots(1, 4, figsize=(10.5, 2.8))
    grey = plt.cm.Greys(np.linspace(0.45, 0.95, len(curves)))
    for (ses, c), g in zip(sorted(curves.items()), grey):
        m = (c["freqs"] >= 1) & (c["freqs"] <= 60)
        ax[0].semilogy(c["freqs"][m], c["psd"][m], color=g, lw=0.9, label=ses)
        ax[1].plot(1000 * c["times"], c["erp"], color=g, lw=0.9)
    ax[0].set(xlabel="frequency (Hz)", ylabel="median PSD ($\\mu$V$^2$/Hz)")
    ax[0].legend(frameon=False, fontsize=6, ncol=2)
    ax[1].axvline(0, color=COULEURS["repere"], lw=0.7)
    ax[1].axhline(0, color=COULEURS["repere"], lw=0.5)
    ax[1].set(xlabel="time from word onset (ms)", ylabel="occipital ERP ($\\mu$V)")
    if len(audio):
        bins = np.linspace(audio.rise_db.min(), audio.rise_db.max(), 50)
        for cond in ("imagined", "articulated"):
            v = audio.loc[audio.cond == cond, "rise_db"].values
            ax[2].hist(v, bins=bins, color=COULEURS[cond], alpha=0.75, label=cond)
        ax[2].set(xlabel="production minus fixation level (dB)", ylabel="trials")
        ax[2].legend(frameon=False)
        on = audio.loc[audio.cond == "articulated", "onset_s"].dropna().values
        ax[3].hist(1000 * on, bins=40, color=COULEURS["articulated"])
        ax[3].set(xlabel="speech onset after go (ms)", ylabel="articulated trials")
    for a, l in zip(ax, "abcd"):
        a.text(-0.02, 1.04, l, transform=a.transAxes, fontweight="bold", ha="right")
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "fig8-technical-validation.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    import pandas as pd
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
    rows, audio_rows, curves = [], [], {}
    for ses, path in sorted(sessions.items()):
        print(f"  {ses}  {path.name}", flush=True)
        streams, _ = pyxdf.load_xdf(str(path), dejitter_timestamps=True)
        row, arows, c = analyse_session(streams, ses)
        del streams
        rows.append(row); audio_rows += arows; curves[ses] = c
        print(f"     alpha {row['alpha_peak_hz']:.1f} Hz, secteur {row['line_noise_db']:+.1f} dB, "
              f"{row['n_outlier_channels']} canaux hors norme, "
              f"pic evoque {row['erp_peak_latency_ms']:.0f} ms "
              f"(x{row['erp_peak_gfp_ratio']:.1f} la ligne de base, r pair/impair "
              f"{row['erp_split_half_r']:.2f})", flush=True)

    t = pd.DataFrame(rows)
    t.to_csv(out / "validation_sessions.csv", index=False)
    a = pd.DataFrame(audio_rows)
    if len(a):
        a.to_csv(out / "validation_audio.csv", index=False)
    figure(curves, a, out / "figures")

    # bilan a reporter dans le manuscrit
    print("\n=== BILAN ===")
    print(f"sessions : {len(t)} ; essais reconstruits : {int(t.n_trials.sum())}")
    print(f"pic alpha : {t.alpha_peak_hz.min():.1f} a {t.alpha_peak_hz.max():.1f} Hz")
    print(f"bruit secteur (avant filtrage) : {t.line_noise_db.min():+.1f} a "
          f"{t.line_noise_db.max():+.1f} dB au-dessus du voisinage")
    print(f"canaux hors norme par session : {t.n_outlier_channels.min()} a "
          f"{t.n_outlier_channels.max()}")
    print(f"latence P1 occipitale : {t.erp_p1_latency_ms.min():.0f} a "
          f"{t.erp_p1_latency_ms.max():.0f} ms ; N1 : {t.erp_n1_latency_ms.min():.0f} a "
          f"{t.erp_n1_latency_ms.max():.0f} ms")
    print(f"pic du potentiel evoque : {t.erp_peak_latency_ms.min():.0f} a "
          f"{t.erp_peak_latency_ms.max():.0f} ms ; rapport a la ligne de base "
          f"{t.erp_peak_gfp_ratio.min():.1f} a {t.erp_peak_gfp_ratio.max():.1f} ; "
          f"r pair/impair {t.erp_split_half_r.min():.2f} a {t.erp_split_half_r.max():.2f}")
    sess = sorted(curves)
    erps = np.array([curves[s]["erp"] for s in sess if len(curves[s]["erp"]) ==
                     len(curves[sess[0]]["erp"])])
    if len(erps) > 1:
        r = np.corrcoef(erps)
        print(f"correlation moyenne entre sessions du potentiel evoque : "
              f"{r[np.triu_indices(len(erps), 1)].mean():.2f}")
    if len(a):
        ima = a[a.cond == "imagined"].rise_db
        art = a[a.cond == "articulated"].rise_db
        seuil = float(np.percentile(art, 5))
        print(f"audio, montee mediane : imagine {ima.median():+.1f} dB, "
              f"articule {art.median():+.1f} dB")
        print(f"essais imagines au-dessus du 5e centile des articules ({seuil:+.1f} dB) : "
              f"{int((ima > seuil).sum())} sur {len(ima)}")
        print(f"essais imagines a plus de 6 dB au-dessus de leur fixation : "
              f"{int((ima > 6).sum())} sur {len(ima)}")
        on = 1000 * a[a.cond == "articulated"].onset_s.dropna()
        print(f"debut de parole articulee : mediane {on.median():.0f} ms, "
              f"intervalle interquartile {on.quantile(.25):.0f} a {on.quantile(.75):.0f} ms, "
              f"detecte sur {len(on)} essais sur {int((a.cond == 'articulated').sum())}")
    else:
        print("aucun flux audio trouve : controles 3 et 4 non calcules")
    print(f"\nsorties : {out / 'validation_sessions.csv'}, "
          f"{out / 'validation_audio.csv'}, {out / 'figures/fig8-technical-validation.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
