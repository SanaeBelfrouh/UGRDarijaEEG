"""Tests de la validation technique, sur des flux XDF synthetiques.

Chaque controle doit retrouver ce qui a ete plante : un pic alpha a 10 Hz, un
pic secteur, une electrode morte, un potentiel evoque a 100 ms apres le mot, et
de la parole dans les seuls essais articules, a 300 ms apres le go.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from darija.validation import (                          # noqa: E402
    audio_levels, channel_names, cue_erp, find_streams, spectrum,
    trials_from_markers,
)

NAMES = ["Fp1", "Fz", "Cz", "Pz", "O1", "Oz", "O2", "PO7", "PO8", "T7"]
SF, AF = 250.0, 8000.0


def _streams(n_trials=40, seed=0):
    rng = np.random.default_rng(seed)
    marks, t = [], 1.0
    plan = []
    for k in range(n_trials):
        cond = "imagined" if k < n_trials // 2 else "articulated"
        fix, cue, isi = 1.2, 1.0, rng.uniform(0.8, 1.2)
        marks += [(f"{cond}_fixation_start", t), (f"{cond}_cue_start:dwa", t + fix),
                  (f"{cond}_cue_end:dwa", t + fix + cue),
                  (f"{cond}_start:dwa", t + fix + cue + isi),
                  (f"{cond}_end:dwa", t + fix + cue + isi + 2.0)]
        plan.append((cond, t + fix, t + fix + cue + isi))
        t += fix + cue + isi + 2.5
    dur = t + 1.0
    n = int(dur * SF)
    ts = np.arange(n) / SF
    x = rng.normal(0, 5, (len(NAMES), n))
    x += 10 * np.sin(2 * np.pi * 10 * ts)                  # alpha a 10 Hz
    x += 3 * np.sin(2 * np.pi * 50 * ts)                   # secteur
    x[NAMES.index("T7")] *= 0.01                           # electrode morte
    occ = [NAMES.index(c) for c in ("O1", "Oz", "O2", "PO7", "PO8")]
    for _, cue, _ in plan:                                 # evoque a 100 ms
        i = int((cue + 0.10) * SF)
        bump = 8 * np.hanning(int(0.06 * SF))
        x[occ, i:i + len(bump)] += bump
    na = int(dur * AF)
    ta = np.arange(na) / AF
    au = rng.normal(0, 1e-3, na)
    for cond, _, go in plan:                               # parole a 300 ms
        if cond == "articulated":
            i, j = int((go + 0.3) * AF), int((go + 1.0) * AF)
            au[i:j] += 0.2 * np.sin(2 * np.pi * 200 * ta[i:j])
    chans = [{"label": [c]} for c in NAMES]
    eeg = {"info": {"name": ["actiCHamp-123"], "type": ["EEG"],
                    "channel_count": [str(len(NAMES))], "nominal_srate": [str(SF)],
                    "desc": [{"channels": [{"channel": chans}]}]},
           "time_series": x.T.astype(np.float32), "time_stamps": ts}
    mk = {"info": {"name": ["DarijaSpeechMarkers"], "type": ["Markers"],
                   "channel_count": ["1"], "nominal_srate": ["0"]},
          "time_series": [[v] for v, _ in marks],
          "time_stamps": np.array([s for _, s in marks])}
    audio = {"info": {"name": ["AudioCaptureWin"], "type": ["Audio"],
                      "channel_count": ["1"], "nominal_srate": [str(AF)]},
             "time_series": au[:, None], "time_stamps": ta}
    return [eeg, mk, audio], n_trials


def test_streams_and_trials_are_found():
    streams, n = _streams()
    eeg, mk, audio = find_streams(streams)
    assert audio is not None and channel_names(eeg) == NAMES
    trials = trials_from_markers(mk)
    assert len(trials) == n
    assert all(tr.fixation < tr.cue < tr.go < tr.end for tr in trials)


def test_spectrum_finds_alpha_line_noise_and_dead_channel():
    streams, _ = _streams()
    eeg, _, _ = find_streams(streams)
    rep = spectrum(np.asarray(eeg["time_series"]).T, SF, NAMES)
    assert abs(rep.alpha_peak_hz - 10.0) <= 0.5, rep.alpha_peak_hz
    assert rep.line_noise_db > 10, rep.line_noise_db
    assert rep.outlier_channels == ["T7"], rep.outlier_channels


def test_cue_erp_peaks_where_it_was_planted():
    streams, _ = _streams()
    eeg, mk, _ = find_streams(streams)
    trials = trials_from_markers(mk)
    rep = cue_erp(np.asarray(eeg["time_series"], float).T, eeg["time_stamps"], SF, NAMES,
                  [t.cue for t in trials])
    assert 0.09 <= rep.peak_latency_s <= 0.16, rep.peak_latency_s
    assert rep.peak_gfp_over_baseline > 2, rep.peak_gfp_over_baseline
    assert rep.split_half_r > 0.8, rep.split_half_r


def test_audio_separates_conditions_and_finds_onset():
    streams, _ = _streams()
    _, mk, audio = find_streams(streams)
    rows = audio_levels(audio, trials_from_markers(mk))
    ima = [r["rise_db"] for r in rows if r["cond"] == "imagined"]
    art = [r for r in rows if r["cond"] == "articulated"]
    assert max(ima) < 3, max(ima)
    assert min(r["rise_db"] for r in art) > 20
    onsets = [r["onset_s"] for r in art]
    assert all(abs(o - 0.30) <= 0.02 for o in onsets), onsets[:5]


def test_script_runs_on_synthetic_session():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "tv", Path(__file__).resolve().parents[1] / "scripts" / "08_technical_validation.py")
    tv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tv)
    streams, n = _streams()
    row, arows, curves = tv.analyse_session(streams, "S001")
    assert row["n_trials"] == n and row["audio_stream"] and len(arows) == n
    assert row["imagined_rise_db_median"] < 3 < row["articulated_rise_db_median"]


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_"):
            continue
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL  {name}: {exc}")
    print("\nechecs :", failures)
    raise SystemExit(1 if failures else 0)
