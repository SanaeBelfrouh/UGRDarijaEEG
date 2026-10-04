#!/usr/bin/env python3
"""Etape 0 : attribution de la gigue, marqueurs seuls, aucun EEG charge.

    python scripts/00_timing_check.py --input /kaggle/input

Ce script existe pour corriger une conclusion. Le carnet de verification du
timing a mesure un ecart-type de 114 ms sur l'intervalle ``cue_end -> start`` et
en a conclu "REAL JITTER, les epoques a decalage fixe sont etalees, V3 est
justifie". Or cet intervalle EST l'ISI, et le protocole le tire lui-meme :

    isiDur = random.uniform(0.8, 1.2)        (EEG_Darija_Multimodal_FINAL_15.psyexp)

Un tirage uniforme sur 0,4 s a un ecart-type de 0.4/sqrt(12) = 115 ms. Les 114 ms
observes sont donc la gigue voulue, reproduite au milliseconde pres, et non une
imprecision d'horloge.

La consequence pratique est l'inverse de celle qui avait ete tiree. Les epoques
sont ancrees sur ``{cond}_start``, donc cette variabilite se situe entierement
AVANT t = 0 : elle ne peut pas etaler la fenetre de production. L'intervalle qui
determine reellement l'alignement des epoques est ``start -> end``, dont
l'ecart-type mesure vaut 1,1 ms, soit un quart d'echantillon a 250 Hz. Par la
regle de decision du carnet lui-meme (< 20 ms = timing logiciel), la conclusion
correcte est que le re-decoupage ne peut rien recuperer.

Ce que la gigue de l'ISI apporte, en revanche, est precieux : elle decorrele
l'indice visuel du debut de production, ce qui empeche un classifieur
d'exploiter le potentiel evoque par la lecture du mot. C'est un point fort du
protocole, pas un defaut.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from darija.io_xdf import CONDITIONS, discover_xdf, read_events   # noqa: E402

STAGES = ["fixation_start", "fixation_end", "cue_start", "cue_end",
          "start", "end", "isi_start", "isi_end", "iti_start", "iti_end"]

INTERVALS = {
    "fixation":     ("fixation_start", "fixation_end"),
    "cue":          ("cue_start", "cue_end"),
    "isi":          ("cue_end", "start"),        # == la fenetre d'ISI
    "production":   ("start", "end"),            # <- celui qui aligne les epoques
    "iti":          ("iti_start", "iti_end"),
}

#: ce que le protocole prescrit, pour comparer mesure et intention
EXPECTED = {
    "fixation":   ("uniform", 1.0, 1.5),
    "cue":        ("fixed", 1.0, 1.0),
    "isi":        ("uniform", 0.8, 1.2),
    "production": ("fixed", 2.0, 2.0),
    "iti":        ("fixed", 0.5, 0.5),
}


def reconstruct(events: list[dict], session: str) -> list[dict]:
    """Regroupe les marqueurs en essais, un essai par ``fixation_start``."""
    trials = []
    for cond in CONDITIONS:
        stream = [e for e in events if e["cond"] == cond]
        current: dict | None = None
        for event in stream:
            if event["stage"] == "fixation_start":
                if current:
                    trials.append(current)
                current = dict(session=session, condition=cond, word=None)
            if current is None:
                continue
            current.setdefault(event["stage"], event["t"])
            if event["word"]:
                current["word"] = event["word"]
        if current:
            trials.append(current)
    return trials


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="/kaggle/input")
    parser.add_argument("--out", default="results/timing.csv")
    args = parser.parse_args()

    files = discover_xdf(args.input)
    if not files:
        print(f"aucun .xdf sous {args.input}", file=sys.stderr)
        return 1

    rows = []
    for session, path in sorted(files.items()):
        events = read_events(path)
        trials = reconstruct(events, session)
        rows.extend(trials)
        print(f"[{session}] {len(events)} marqueurs -> {len(trials)} essais", flush=True)

    table = pd.DataFrame(rows)
    for name, (a, b) in INTERVALS.items():
        if a in table.columns and b in table.columns:
            table[name] = table[b] - table[a]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)

    print("\n" + "=" * 92)
    print("MESURE CONTRE INTENTION DU PROTOCOLE")
    print("=" * 92)
    print(f"{'intervalle':12s} {'median':>9s} {'ET (ms)':>9s} "
          f"{'min':>8s} {'max':>8s}   {'prescrit':>22s}   verdict")

    for name, (kind, lo, hi) in EXPECTED.items():
        if name not in table.columns:
            continue
        values = table[name].dropna()
        if not len(values):
            continue
        sd_ms = float(values.std() * 1000)

        if kind == "uniform":
            # ecart-type d'une loi uniforme sur [lo, hi]
            expected_sd_ms = (hi - lo) / np.sqrt(12) * 1000
            verdict = ("gigue VOULUE, conforme"
                       if abs(sd_ms - expected_sd_ms) < 0.15 * expected_sd_ms
                       else f"ATTENDU {expected_sd_ms:.0f} ms")
            prescrit = f"uniforme {lo}-{hi} s"
        else:
            verdict = "horloge logicielle" if sd_ms < 20 else "DERIVE ANORMALE"
            prescrit = f"fixe {lo} s"

        print(f"{name:12s} {values.median():9.4f} {sd_ms:9.2f} "
              f"{values.min():8.4f} {values.max():8.4f}   {prescrit:>22s}   {verdict}")

    print("\n" + "=" * 92)
    print("VERDICT SUR L'ALIGNEMENT DES EPOQUES")
    print("=" * 92)
    production_sd_ms = float(table["production"].dropna().std() * 1000)
    print(f"L'intervalle qui determine l'alignement est 'production' (start -> end),")
    print(f"puisque les epoques sont ancrees sur 'start'. Son ecart-type vaut "
          f"{production_sd_ms:.2f} ms,")
    print(f"soit {production_sd_ms * 250 / 1000:.2f} echantillon a 250 Hz.")
    print()
    if production_sd_ms < 20:
        print("  < 20 ms : le timing est pilote par logiciel. Re-decouper sur les")
        print("  marqueurs par essai ne peut rien recuperer, et les resultats a 6-9 %")
        print("  ne sont pas un artefact d'alignement.")
        print()
        print("  L'ecart-type de ~115 ms sur 'isi' est le tirage uniform(0.8, 1.2) du")
        print("  protocole, pas une imprecision. Il se situe AVANT t = 0 et ne peut")
        print("  donc pas etaler la fenetre de production ; il decorrele au contraire")
        print("  l'indice visuel du debut de production, ce qui protege le decodage")
        print("  d'une fuite par potentiel evoque.")
    else:
        print("  > 20 ms sur l'intervalle d'ancrage : la, il y aurait matiere a")
        print("  re-decouper.")

    print("\n" + "=" * 92)
    print("CE QUE LE TIMING CONFIRME PAR AILLEURS")
    print("=" * 92)
    if "word" in table.columns and "production" in table.columns:
        imagined = table[(table.condition == "imagined") & table.production.notna()]
        if len(imagined) and imagined.word.notna().any():
            spread = imagined.groupby("word").production.median()
            print(f"Duree de production imaginee par mot : etendue "
                  f"{spread.max() - spread.min():.4f} s.")
            print("Nulle par construction : la fenetre est fixee a 2 s pour tous les mots.")
            print("La condition imaginee ne porte donc AUCUN confondu de duree, ce qui")
            print("n'est pas le cas de l'articulee, ou la duree reelle varie avec le mot.")
    print(f"\nevenements par essai -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
