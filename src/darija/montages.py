"""Sous-ensembles d'electrodes, pour les controles d'artefact musculaire.

Le raisonnement de ces controles est un raisonnement de contraste, pas de seuil.
On ne demande pas si un montage decode, mais si un montage *cense ne porter que
du muscle* decode aussi bien que le montage complet. S'il fait aussi bien, ce que
le montage complet lit est probablement du muscle. S'il reste derriere, il y a
autre chose.

Les listes suivent la nomenclature 10-20 du bonnet actiCAP 64 canaux. La
resolution est tolerante a la casse et ignore les electrodes absentes, ce qui
permet de reutiliser ces montages sur un autre bonnet sans les reecrire.

``machoire10`` est la liste de tous les resultats « jaw ring » du manuscrit
(Table 5, section 4.4), qui nomme ses dix electrodes : temporales et
mastoidiennes, au plus pres des muscles masticateurs et du cou
(Muthukumaraswamy, 2013). Sur le bonnet actiCAP 64 canaux, les dix et les huit
occipitales sont toutes presentes, d'ou 64 - 18 = 46 canaux pour
``sans_machoire``.
"""

from __future__ import annotations

import numpy as np

#: anneau mandibulaire : temporales et mastoidiennes, la ou l'EMG des muscles
#: masticateurs et du cou domine
JAW_RING = ("FT7", "FT8", "FT9", "FT10", "T7", "T8", "TP7", "TP8", "TP9", "TP10")

#: occipitales : ne devraient porter aucune information de parole, mais portent
#: la reponse visuelle a l'indice. Controle de ce que vaut un montage
#: « plausible mais hors sujet ».
OCCIPITAL = ("O1", "O2", "Oz", "PO3", "PO4", "PO7", "PO8", "POz")

#: les electrodes peripheriques recommandees pour detecter l'artefact musculaire
PERIPHERAL_14 = ("Fp1", "Fp2", "F7", "F8", "FT7", "FT8", "T7", "T8",
                 "TP7", "TP8", "P7", "P8", "PO7", "PO8")

#: centrales et frontales : ce qui reste quand on retire mâchoire et occiput
CENTRAL = ("Fz", "F1", "F2", "F3", "F4", "FC1", "FC2", "FC3", "FC4", "FCz",
           "Cz", "C1", "C2", "C3", "C4", "CP1", "CP2", "CP3", "CP4", "CPz")

MONTAGES: dict[str, dict] = {
    "complet":      dict(keep=None, drop=None,
                         note="tous les canaux survivants"),
    "machoire10":   dict(keep=JAW_RING, drop=None,
                         note="anneau mandibulaire, temoin EMG"),
    "occipital":    dict(keep=OCCIPITAL, drop=None,
                         note="occipitales, temoin hors sujet"),
    "peripherique14": dict(keep=PERIPHERAL_14, drop=None,
                           note="peripheriques de Muthukumaraswamy (2013)"),
    "central":      dict(keep=CENTRAL, drop=None,
                         note="centrales et frontales"),
    "sans_machoire": dict(keep=None, drop=JAW_RING + OCCIPITAL,
                          note="complet moins mâchoire et occiput"),
}


def resolve(ch_names, montage: str) -> np.ndarray:
    """Indices des canaux d'un montage nomme.

    Renvoie les indices dans l'ordre de ``ch_names``, jamais dans celui de la
    liste du montage : garder l'ordre du fichier evite qu'une permutation
    silencieuse des canaux ne change les covariances.
    """
    if montage not in MONTAGES:
        raise ValueError(f"montage inconnu : {montage} (disponibles : {sorted(MONTAGES)})")

    names = [str(c) for c in ch_names]
    lowered = [n.lower() for n in names]
    spec = MONTAGES[montage]

    if spec["keep"] is not None:
        wanted = {c.lower() for c in spec["keep"]}
        idx = [i for i, n in enumerate(lowered) if n in wanted]
        # Tolerant plutot que strict : l'electrode de reference en ligne (FCz
        # ici) n'existe pas comme canal de donnees, et un bonnet different
        # nomme parfois autrement. On signale ce qui manque via missing_from()
        # et on n'echoue que s'il ne reste pas de quoi estimer une covariance.
    else:
        excluded = {c.lower() for c in (spec["drop"] or ())}
        idx = [i for i, n in enumerate(lowered) if n not in excluded]

    if len(idx) < 4:
        raise ValueError(
            f"montage {montage} : {len(idx)} canaux retenus, trop peu pour une "
            f"covariance. Electrodes attendues et absentes : "
            f"{sorted(missing_from(ch_names, montage))}"
        )
    return np.asarray(idx, dtype=int)


def missing_from(ch_names, montage: str) -> list[str]:
    """Electrodes que le montage attend et que le fichier ne contient pas."""
    spec = MONTAGES[montage]
    if spec["keep"] is None:
        return []
    present = {str(c).lower() for c in ch_names}
    return sorted(c for c in spec["keep"] if c.lower() not in present)


def describe(ch_names) -> str:
    """Tableau des montages applicables a ce fichier, avec leur effectif."""
    lines = [f"{'montage':16s} {'canaux':>7s}  note"]
    for name, spec in MONTAGES.items():
        try:
            n = len(resolve(ch_names, name))
            absent = missing_from(ch_names, name)
            suffix = f"  [absentes : {', '.join(absent)}]" if absent else ""
            lines.append(f"{name:16s} {n:7d}  {spec['note']}{suffix}")
        except ValueError as exc:
            lines.append(f"{name:16s} {'-':>7s}  INDISPONIBLE : {exc}")
    return "\n".join(lines)
