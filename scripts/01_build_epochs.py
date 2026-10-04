#!/usr/bin/env python3
"""Etape 1 : XDF bruts -> archive d'epoques verrouillees par essai.

A executer une fois, sur le noyau qui a les XDF attaches. Produit un seul .npz
que les etapes suivantes relisent en quelques secondes.

    python scripts/01_build_epochs.py --input /kaggle/input --out epochs.npz

Les epoques sont ancrees sur ``{condition}_start`` (le go), pas sur l'indice
visuel. La fenetre par defaut commence a -0.5 s, ce qui reste dans l'ISI meme
pour l'ISI le plus court (0.8 s) : aucune epoque ne contient le mot affiche.

Le filtrage a lieu sur le signal CONTINU, avant le decoupage. Filtrer epoque par
epoque laisse des transitoires de bord qui, sur un amplificateur couple en
continu comme l'actiCHamp, depassent l'amplitude de l'EEG lui-meme.

Le tableau de diagnostic imprime pour chaque session l'amplitude crete-a-crete
observee. Le lire avant de regler un seuil : si la mediane n'est pas de l'ordre
de quelques dizaines de microvolts, le flux n'est pas en microvolts et un seuil
absolu en uV n'a aucun sens (utiliser alors --scale).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from darija.io_xdf import DEFAULT_TMAX, DEFAULT_TMIN, discover_xdf, load_session  # noqa: E402
from darija.preprocess import (                                                   # noqa: E402
    looks_like_microvolts,
    prepare,
    reject_trials,
    remove_ocular,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="/kaggle/input", help="racine des XDF")
    parser.add_argument("--out", default="epochs.npz")
    parser.add_argument("--tmin", type=float, default=DEFAULT_TMIN)
    parser.add_argument("--tmax", type=float, default=DEFAULT_TMAX)
    parser.add_argument("--sfreq", type=float, default=250.0)
    parser.add_argument("--low", type=float, default=0.5)
    parser.add_argument("--high", type=float, default=45.0)
    parser.add_argument("--scale", type=float, default=1.0,
                        help="facteur applique au flux brut (1e6 si le flux est en volts)")
    parser.add_argument("--reject-uv", type=float, default=150.0,
                        help="amplitude crete-a-crete au-dela de laquelle un CANAL est bruite")
    parser.add_argument("--max-bad-fraction", type=float, default=0.15,
                        help="proportion de canaux bruites toleree avant de rejeter l'essai")
    parser.add_argument("--remove-ocular", action="store_true",
                        help="retire les composantes ICA a poids frontal (oculaires)")
    parser.add_argument("--ocular-ratio", type=float, default=2.5,
                        help="seuil en multiple de la part frontale d'une composante uniforme")
    parser.add_argument("--no-reject", action="store_true",
                        help="conserve tous les essais ; le rejet est alors laisse a l'analyse")
    args = parser.parse_args()

    files = discover_xdf(args.input)
    if not files:
        print(f"aucun .xdf sous {args.input}", file=sys.stderr)
        return 1

    all_X, all_y, all_cond, all_session, all_index, all_block = [], [], [], [], [], []
    ch_names = None
    reports = []

    for session, path in sorted(files.items()):
        epochs = load_session(path, tmin=args.tmin, tmax=args.tmax,
                              target_sfreq=args.sfreq, session=session,
                              low=args.low, high=args.high, scale=args.scale)
        X = prepare(epochs.X, epochs.sfreq)

        if args.remove_ocular:
            X, ica_report = remove_ocular(X, epochs.ch_names, ratio=args.ocular_ratio)
            print(f"  [{session}] ICA : composantes retirees {ica_report['dropped']} "
                  f"(part frontale {ica_report['dropped_share']}, "
                  f"seuil {ica_report['seuil']}, uniforme {ica_report['part_uniforme']})",
                  flush=True)
        keep, report = reject_trials(X, amplitude_uv=args.reject_uv,
                                     max_bad_fraction=args.max_bad_fraction)
        if args.no_reject:
            keep = np.ones(len(X), dtype=bool)

        report["session"] = session
        reports.append(report)

        if ch_names is None:
            ch_names = epochs.ch_names
        elif ch_names != epochs.ch_names:
            raise RuntimeError(f"{session}: montage different des sessions precedentes")

        all_X.append(X[keep].astype(np.float32))
        all_y.append(epochs.y[keep])
        all_cond.append(epochs.cond[keep])
        all_session.append(np.full(int(keep.sum()), session))
        all_index.append(epochs.trial_index[keep])
        all_block.append(epochs.sub_block[keep])

        n_blocks = len(np.unique(epochs.sub_block))
        print(f"[{session}] {keep.sum():4d}/{len(keep)} essais conserves | "
              f"{n_blocks} sous-blocs | "
              f"amplitude mediane {report['ptp_median_uv']:8.1f} "
              f"(p05 {report['ptp_p05_uv']:.1f}, p95 {report['ptp_p95_uv']:.1f}) | "
              f"{epochs.orig_sfreq:.0f} -> {epochs.sfreq:.0f} Hz", flush=True)

    # --- diagnostic d'unites, avant toute conclusion sur le taux de rejet ---
    medians = [r["ptp_median_uv"] for r in reports]
    overall = float(np.median(medians))
    if not looks_like_microvolts(overall):
        factor = 1e6 if overall < 1.0 else 1e-3
        print("\n" + "!" * 88)
        print(f"AMPLITUDE MEDIANE = {overall:.4g} apres passe-bande {args.low}-{args.high} Hz.")
        print("De l'EEG de scalp filtre se situe entre ~5 et ~150 uV crete-a-crete.")
        print("Cette valeur indique que le flux LSL n'est PAS en microvolts, et donc")
        print(f"que --reject-uv {args.reject_uv} ne veut rien dire pour ces donnees.")
        print(f"  -> relancer avec --scale {factor:g} (puis verifier la mediane)")
        print("!" * 88)

    total_kept = sum(len(x) for x in all_X)
    total_trials = sum(r["n_trials"] for r in reports)

    if total_kept == 0:
        print("\n" + "=" * 88, file=sys.stderr)
        print("AUCUNE EPOQUE CONSERVEE — aucune archive ecrite.", file=sys.stderr)
        print("=" * 88, file=sys.stderr)
        print("Le rejet a tout elimine. Les trois causes possibles, dans l'ordre :",
              file=sys.stderr)
        print(f"  1. unites : amplitude mediane {overall:.4g}, voir --scale ci-dessus",
              file=sys.stderr)
        print(f"  2. seuil trop strict : --reject-uv {args.reject_uv} avec "
              f"--max-bad-fraction {args.max_bad_fraction}", file=sys.stderr)
        print("  3. pour voir les donnees sans aucun filtre de qualite : --no-reject",
              file=sys.stderr)
        return 2

    X = np.concatenate(all_X)
    times = args.tmin + np.arange(X.shape[-1]) / args.sfreq

    np.savez_compressed(
        args.out,
        X=X,
        y=np.concatenate(all_y).astype(str),
        cond=np.concatenate(all_cond),
        session=np.concatenate(all_session),
        trial_index=np.concatenate(all_index),
        sub_block=np.concatenate(all_block),
        times=times,
        ch_names=np.asarray(ch_names),
        sfreq=args.sfreq,
    )

    size_gb = Path(args.out).stat().st_size / 1e9
    print(f"\n{total_kept}/{total_trials} epoques conservees "
          f"({100 * (1 - total_kept / total_trials):.1f} % rejetees), "
          f"{X.shape[1]} canaux, {X.shape[2]} echantillons")
    print(f"-> {args.out} ({size_gb:.2f} Go)")

    if total_kept < 0.5 * total_trials:
        print("\nATTENTION : plus de la moitie des essais a ete rejetee. Verifier le")
        print("tableau d'amplitudes ci-dessus avant de continuer : un taux pareil vient")
        print("presque toujours d'un seuil mal calibre, pas de donnees reellement mauvaises.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
