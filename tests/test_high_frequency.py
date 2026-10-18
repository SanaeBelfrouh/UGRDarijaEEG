"""Tests de l'analyse en bande large (scripts/10_high_frequency.py).

Deux choses doivent tenir : le banc etendu tourne de bout en bout sur une
archive filtree jusqu'a 120 Hz, et il est refuse sur une archive filtree a
45 Hz, ou ses bandes hautes ne contiendraient que du bruit filtre.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from test_notebook import build_archive                    # noqa: E402


def _archive(path: Path, sfreq: float, high: float | None) -> Path:
    build_archive(path, n_per_word=2, sfreq=sfreq)
    if high is not None:
        data = dict(np.load(path, allow_pickle=True))
        data["filter_low"], data["filter_high"] = 0.5, high
        np.savez_compressed(path, **data)
    return path


def _run(*args) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True)


def test_high_bank_is_refused_on_a_45_hz_archive():
    with tempfile.TemporaryDirectory() as tmp:
        arch = _archive(Path(tmp) / "e.npz", sfreq=250.0, high=None)
        r = _run(ROOT / "scripts" / "02_benchmark.py", "--epochs", arch,
                 "--condition", "imagined", "--models", "bankcov_tangent_lda",
                 "--schemes", "loso", "--contrasts", "mots15", "--bank", "haute",
                 "--n-perm", 20, "--out", Path(tmp) / "r.csv")
        assert r.returncode != 0
        assert "reconstruire" in (r.stdout + r.stderr), r.stdout[-500:] + r.stderr[-500:]


def test_broadband_analysis_runs_end_to_end():
    import pandas as pd
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        arch = _archive(tmp / "epochs_hf.npz", sfreq=500.0, high=120.0)
        r = _run(ROOT / "scripts" / "10_high_frequency.py", "--work", tmp,
                 "--epochs", arch, "--n-perm", 20)
        assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
        for bank in ("manuscrit", "haute"):
            t = pd.read_csv(tmp / "results" / f"hf_{bank}.csv")
            assert set(t.contraste) == {"mots15", "syllabes"}, set(t.contraste)
            assert set(t.recentre) == {True, False}
        assert "BILAN" in r.stdout and "120 Hz, banc haute" in r.stdout


def test_followup_pairs_the_two_banks_and_runs_the_pre_go_window():
    import pandas as pd
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        arch = _archive(tmp / "epochs_hf.npz", sfreq=500.0, high=120.0)
        r = _run(ROOT / "scripts" / "10_high_frequency.py", "--work", tmp,
                 "--epochs", arch, "--n-perm", 20, "--followup-only")
        assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
        t = pd.read_csv(tmp / "results" / "hf_paired_banks.csv")
        # 2 schemas x 2 recentrages x 2 contrastes, chacun sur les memes essais
        assert len(t) == 8, t
        assert t.nulle_moy.abs().max() < 0.05, t.nulle_moy
        assert (t.p_holm >= t.p_apparie).all()
        for bank in ("manuscrit", "haute"):
            pre = pd.read_csv(tmp / "results" / f"hf_pre_{bank}.csv")
            assert set(pre.contraste) == {"mots15", "syllabes"}
        assert "TEST APPARIE" in r.stdout and "avant go, banc haute" in r.stdout


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print(f"  PASS  {name}")
            except AssertionError as exc:
                failures += 1
                print(f"  FAIL  {name}: {exc}")
    print("\nechecs :", failures)
    raise SystemExit(1 if failures else 0)
