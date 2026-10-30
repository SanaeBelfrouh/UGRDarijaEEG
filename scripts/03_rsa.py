#!/usr/bin/env python3
"""Etape 3 : similarite representationnelle des 15 mots.

    python scripts/03_rsa.py --epochs epochs.npz --condition imagined

L'ASR pose une question differente de la classification et y repond avec plus de
sensibilite. Un classifieur a 15 classes doit, pour chaque essai, choisir la
bonne reponse parmi quinze ; l'ASR se contente de demander si les mots dont la
phonologie se ressemble produisent des motifs EEG qui se ressemblent. Elle
agrege les 105 paires au lieu de resumer le tout par une exactitude, et elle
peut sortir significative la ou le classifieur reste au hasard.

La dissimilarite est validee croisee (crossnobis) : d'esperance nulle sous
l'hypothese nulle, contrairement a une distance de Mahalanobis ordinaire qui est
toujours positive et donc ininterpretable seule.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from darija import WORDS                                          # noqa: E402
from darija.preprocess import crop                                # noqa: E402
from darija.riemann import covariances, geometric_mean, tangent_vectors  # noqa: E402
from darija.rsa import crossnobis_rdm, mantel_test, model_rdm     # noqa: E402
from darija.stats import holm_bonferroni                          # noqa: E402

warnings.filterwarnings("ignore")

#: Les deux derniers ne sont pas des hypotheses mais des CONTROLES : si la
#: geometrie neurale des mots imagines suit la forme ecrite de l'indice, ce qui
#: est decode est en partie orthographique. Un indice visuel engage
#: l'orthographe quel que soit l'alphabet, donc l'objection se pose pour
#: n'importe quelle transcription ; la mesurer vaut mieux que l'argumenter.
MODEL_KEYS = ("onset", "onset_voiced", "pharyngeal", "semantic", "orthographic",
              "cue_length")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", default="epochs.npz")
    parser.add_argument("--condition", default="imagined",
                        choices=("imagined", "articulated"))
    parser.add_argument("--tmin", type=float, default=0.0)
    parser.add_argument("--tmax", type=float, default=2.0)
    parser.add_argument("--n-perm", type=int, default=10000)
    parser.add_argument("--out", default="results/rsa.csv")
    args = parser.parse_args()

    archive = np.load(args.epochs, allow_pickle=True)
    mask = archive["cond"] == args.condition
    X, times = crop(archive["X"][mask], archive["times"], args.tmin, args.tmax)
    y = archive["y"][mask]
    session = archive["session"][mask]

    print(f"{len(X)} essais {args.condition}, fenetre [{times[0]:.2f}, {times[-1]:.2f}] s")

    # motifs = vecteurs tangents des covariances. La reference est la moyenne
    # geometrique de TOUS les essais : ici ce n'est pas une fuite, puisque l'ASR
    # n'entraine aucun classifieur et que la reference est independante des
    # etiquettes.
    covs = covariances(X)
    patterns = tangent_vectors(covs, geometric_mean(covs))
    print(f"motifs : {patterns.shape}")

    rdm, labels = crossnobis_rdm(patterns, y, session, labels=list(WORDS))
    np.savez_compressed(Path(args.out).with_suffix(".rdm.npz"),
                        rdm=rdm, labels=np.asarray(labels))

    rows = []
    for key in MODEL_KEYS:
        rho, pvalue = mantel_test(rdm, model_rdm(labels, key),
                                  n_perm=args.n_perm, seed=0)
        rows.append(dict(modele=key, rho_spearman=round(rho, 4), p_mantel=round(pvalue, 4)))
        print(f"  {key:14s} rho = {rho:+.4f}   p = {pvalue:.4f}")

    table = pd.DataFrame(rows)
    table["p_holm"] = holm_bonferroni(table.p_mantel.values)
    table["significatif"] = table.p_holm < 0.05

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)

    print("\n" + "=" * 70)
    print(table.to_string(index=False))
    print("=" * 70)

    off_diagonal = rdm[np.tril_indices(len(labels), k=-1)]
    print(f"\nDissimilarite moyenne entre mots : {off_diagonal.mean():+.6f}")
    print("Sous l'hypothese nulle cette moyenne vaut zero : l'estimateur crossnobis")
    print("est non biaise. Une valeur nettement positive indique que les mots se")
    print("distinguent, meme si aucun classifieur ne parvient a les nommer.")
    print(f"\nresultats -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
