"""Cartes topographiques : positions d'electrodes et trace.

Les positions sont FIGEES dans ce fichier plutot que lues depuis MNE. Le depot
n'a deliberement aucune dependance a MNE — tout y est reimplemente — et faire
dependre une figure d'un paquet de cinquante megaoctets pour y lire une table de
coordonnees serait un mauvais echange. Elles ont ete extraites une fois,
proprement, et leur provenance est ci-dessous.

Provenance
----------
``mne.channels.make_standard_montage("standard_1020")``, MNE 1.13.2, qui place
les electrodes sur la tete moyenne Colin27. Le centre de la sphere a ete ajuste
par moindres carres sur les 94 positions avant projection : sans cet ajustement
Cz tombe a 0,058 de l'origine et toute la carte est decentree.

Projection
----------
Azimutale equidistante, celle employee par l'ensemble des traces topographiques
EEG : l'angle polaire depuis le vertex devient le rayon, l'azimut devient
l'angle. Le rayon 1,0 correspond a l'equateur de la sphere, ou se trouvent T7 et
T8 (mesures a 1,043 et 1,045, symetriques a 0,002 pres).

Les electrodes mandibulaires et mastoidiennes — FT9, FT10, TP9, TP10 — tombent
a un rayon de 1,27 a 1,31, donc **a l'exterieur du cercle de tete**. Ce n'est pas
une erreur de projection : elles sont physiquement sous la ligne des oreilles.
Les laisser visibles hors du cercle est volontaire, puisque ce sont justement
celles que les controles musculaires interrogent, et leur position hors tete dit
d'elle-meme ce qu'elles sont.

Orientation : nez en haut, oreille gauche a gauche, convention habituelle.
"""

from __future__ import annotations

import numpy as np

#: nom d'electrode -> (x, y) apres projection. Voir la provenance ci-dessus.
POSITIONS: dict[str, tuple[float, float]] = {
    "A1": (-1.3984, -0.1362),
    "A2": (+1.4043, -0.1399),
    "AF1": (-0.1512, +0.7599),
    "AF10": (+0.6731, +1.0884),
    "AF2": (+0.1501, +0.7608),
    "AF3": (-0.2921, +0.7924),
    "AF4": (+0.2930, +0.7891),
    "AF5": (-0.4311, +0.8351),
    "AF6": (+0.4255, +0.8380),
    "AF7": (-0.5706, +0.8742),
    "AF8": (+0.5622, +0.8805),
    "AF9": (-0.6703, +1.0876),
    "AFz": (-0.0038, +0.7574),
    "C1": (-0.2392, +0.0425),
    "C2": (+0.2428, +0.0454),
    "C3": (-0.4907, +0.0364),
    "C4": (+0.4958, +0.0421),
    "C5": (-0.7552, +0.0259),
    "C6": (+0.7593, +0.0345),
    "CP1": (-0.2253, -0.1913),
    "CP2": (+0.2350, -0.1905),
    "CP3": (-0.4597, -0.2180),
    "CP4": (+0.4683, -0.2139),
    "CP5": (-0.7085, -0.2648),
    "CP6": (+0.7154, -0.2560),
    "CPz": (-0.0020, -0.1849),
    "Cz": (-0.0019, +0.0451),
    "F1": (-0.2028, +0.5282),
    "F10": (+1.0041, +0.8241),
    "F2": (+0.2075, +0.5341),
    "F3": (-0.4072, +0.5566),
    "F4": (+0.4101, +0.5682),
    "F5": (-0.6111, +0.6053),
    "F6": (+0.6173, +0.6096),
    "F7": (-0.8105, +0.6738),
    "F8": (+0.8080, +0.6809),
    "F9": (-1.0031, +0.8242),
    "FC1": (-0.2344, +0.2867),
    "FC2": (+0.2323, +0.2930),
    "FC3": (-0.4743, +0.3057),
    "FC4": (+0.4760, +0.3112),
    "FC5": (-0.7276, +0.3285),
    "FC6": (+0.7273, +0.3365),
    "FCz": (-0.0022, +0.2829),
    "FT10": (+1.2278, +0.4549),
    "FT7": (-0.9876, +0.3715),
    "FT8": (+0.9830, +0.3873),
    "FT9": (-1.2249, +0.4493),
    "Fp1": (-0.2935, +0.9777),
    "Fp2": (+0.2821, +0.9814),
    "Fpz": (-0.0057, +0.9887),
    "Fz": (-0.0028, +0.5220),
    "Iz": (-0.0078, -1.1203),
    "M1": (-1.3218, -0.4334),
    "M2": (+1.3244, -0.4433),
    "O1": (-0.2762, -0.8795),
    "O10": (+0.3295, -1.1093),
    "O2": (+0.2686, -0.8816),
    "O9": (-0.3437, -1.1036),
    "Oz": (-0.0054, -0.8834),
    "P1": (-0.1931, -0.4212),
    "P10": (+0.9760, -0.7716),
    "P2": (+0.2025, -0.4149),
    "P3": (-0.3928, -0.4552),
    "P4": (+0.3982, -0.4494),
    "P5": (-0.5889, -0.5176),
    "P6": (+0.5874, -0.5191),
    "P7": (-0.7834, -0.6096),
    "P8": (+0.7823, -0.6114),
    "P9": (-0.9817, -0.7621),
    "PO1": (-0.1505, -0.6517),
    "PO10": (+0.6653, -0.9997),
    "PO2": (+0.1468, -0.6531),
    "PO3": (-0.2969, -0.6725),
    "PO4": (+0.2904, -0.6789),
    "PO5": (-0.4273, -0.7200),
    "PO6": (+0.4263, -0.7198),
    "PO7": (-0.5424, -0.7907),
    "PO8": (+0.5381, -0.7940),
    "PO9": (-0.6759, -0.9904),
    "POz": (-0.0037, -0.6409),
    "Pz": (-0.0025, -0.4094),
    "T10": (+1.3087, +0.0027),
    "T3": (-1.0433, +0.0063),
    "T4": (+1.0445, +0.0187),
    "T5": (-0.7834, -0.6096),
    "T6": (+0.7823, -0.6114),
    "T7": (-1.0433, +0.0063),
    "T8": (+1.0445, +0.0187),
    "T9": (-1.3034, +0.0106),
    "TP10": (+1.2034, -0.4296),
    "TP7": (-0.9686, -0.3339),
    "TP8": (+0.9701, -0.3318),
    "TP9": (-1.2045, -0.4183),
}

#: rayon du cercle de tete, en unites de la projection
RAYON_TETE = 1.0


def resolve_positions(ch_names) -> tuple[np.ndarray, list[int], list[str]]:
    """(positions, indices retenus, noms introuvables).

    La resolution ignore la casse : un bonnet ecrit parfois ``FP1`` la ou la
    table dit ``Fp1``, et echouer la-dessus serait absurde.
    """
    table = {k.lower(): v for k, v in POSITIONS.items()}
    xy, garde, absents = [], [], []
    for i, nom in enumerate(ch_names):
        p = table.get(str(nom).lower())
        if p is None:
            absents.append(str(nom))
            continue
        xy.append(p)
        garde.append(i)
    return np.asarray(xy, dtype=float), garde, absents


def _masque_tete(xx, yy, rayon):
    return np.sqrt(xx ** 2 + yy ** 2) <= rayon


def draw_topomap(ax, valeurs, xy, *, cmap="Blues", vmin=None, vmax=None,
                 resolution=160, marge=1.35, noms=None, surligne=()):
    """Trace une carte topographique et renvoie l'objet image.

    L'interpolation est une ponderation par l'inverse du carre de la distance.
    Elle est simple, bornee par les valeurs mesurees — contrairement a une
    spline, qui peut inventer des extrema entre les electrodes — et c'est ce
    qu'on veut pour une grandeur qu'on va lire comme une mesure.

    ``surligne`` entoure les electrodes nommees, pour pointer un montage sans
    recourir a une couleur supplementaire.
    """
    valeurs = np.asarray(valeurs, dtype=float)
    etendue = max(marge, np.abs(xy).max() * 1.08)
    grille = np.linspace(-etendue, etendue, resolution)
    xx, yy = np.meshgrid(grille, grille)

    d2 = ((xx[..., None] - xy[:, 0]) ** 2 + (yy[..., None] - xy[:, 1]) ** 2)
    poids = 1.0 / np.maximum(d2, 1e-6) ** 1.0
    champ = (poids * valeurs).sum(axis=-1) / poids.sum(axis=-1)
    # hors du cercle de tete, l'interpolation n'a aucun appui : on n'affiche rien
    champ[~_masque_tete(xx, yy, RAYON_TETE)] = np.nan

    im = ax.imshow(champ, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                   extent=(-etendue, etendue, -etendue, etendue),
                   interpolation="bilinear", zorder=1)

    # contour de tete, nez et oreilles
    t = np.linspace(0, 2 * np.pi, 200)
    ax.plot(RAYON_TETE * np.cos(t), RAYON_TETE * np.sin(t),
            color="black", lw=1.1, zorder=4)
    ax.plot([-0.12, 0, 0.12], [RAYON_TETE * 0.99, RAYON_TETE * 1.13,
                               RAYON_TETE * 0.99], color="black", lw=1.1, zorder=4)
    for cote in (-1, 1):
        a = np.linspace(-0.42, 0.42, 40)
        # un arc accole au cercle : dessine comme une ellipse detachee, l'oreille
        # flotte a cote de la tete des que le rayon change
        rr = RAYON_TETE * (1.0 + 0.055 * np.cos(a * 3.6))
        ax.plot(cote * rr * np.cos(a), rr * np.sin(a),
                color="black", lw=1.1, zorder=4)

    ax.scatter(xy[:, 0], xy[:, 1], s=5, c="black", zorder=5, linewidths=0)
    for i in (surligne or ()):
        ax.scatter(xy[i, 0], xy[i, 1], s=42, facecolor="none",
                   edgecolor="black", lw=1.3, zorder=6)
    if noms is not None:
        for i, nom in enumerate(noms):
            if nom:
                ax.annotate(nom, xy[i], fontsize=5.5, ha="center", va="bottom",
                            xytext=(0, 3), textcoords="offset points", zorder=6)

    ax.set(xlim=(-etendue, etendue), ylim=(-etendue, etendue * 1.08))
    ax.set_aspect("equal")
    ax.axis("off")
    return im
