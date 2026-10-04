"""Decompositions du probleme a 15 classes en contrastes plus puissants.

Attaquer 15 classes de front est le pire usage possible de 1800 essais : 120 par
classe, un hasard a 6,67 %, et une exactitude minimale detectable autour de 8 %.
Les memes essais regroupes en un contraste binaire donnent 900 contre 900 et une
exactitude minimale detectable autour de 53 % : la sensibilite change d'ordre de
grandeur sans enregistrer un seul essai de plus.

Les regroupements ci-dessous sont des hypotheses, pas des verites. Ils encodent
ce qui est lisible depuis la transcription en arabizi et le sens des mots ; une
locutrice native doit relire ``WORD_FEATURES`` et corriger, en particulier le
nombre de syllabes de ``dwa``, ``lma``, ``lla`` et ``tbib``, dont la
syllabation depend du debit.
"""

from __future__ import annotations

import numpy as np

from . import WORDS

#: Une entree par mot :
#:   gloss        traduction
#:   syllables    nombre de syllabes en debit normal (A VERIFIER)
#:   onset        classe articulatoire du premier phoneme
#:   onset_voiced le premier phoneme est-il voise
#:   pharyngeal   le mot contient-il la pharyngale /ʕ/ (note 3 en arabizi)
#:   semantic     categorie d'usage
#:   polarity     'yes' / 'no' / None
WORD_FEATURES: dict[str, dict] = {
    "3afak":   dict(gloss="s'il te plait", syllables=2, onset="pharyngeal", onset_voiced=True,
                    pharyngeal=True,  semantic="politesse", polarity=None),
    "3awnni":  dict(gloss="aide-moi",      syllables=2, onset="pharyngeal", onset_voiced=True,
                    pharyngeal=True,  semantic="requete",  polarity=None),
    "3tini":   dict(gloss="donne-moi",     syllables=2, onset="pharyngeal", onset_voiced=True,
                    pharyngeal=True,  semantic="requete",  polarity=None),
    "3yan":    dict(gloss="fatigue",       syllables=2, onset="pharyngeal", onset_voiced=True,
                    pharyngeal=True,  semantic="etat",     polarity=None),
    "Ah":      dict(gloss="oui",           syllables=1, onset="glottal",    onset_voiced=False,
                    pharyngeal=False, semantic="polarite", polarity="yes"),
    "bghit":   dict(gloss="je veux",       syllables=1, onset="plosive",    onset_voiced=True,
                    pharyngeal=False, semantic="requete",  polarity=None),
    "dwa":     dict(gloss="medicament",    syllables=2, onset="plosive",    onset_voiced=True,
                    pharyngeal=False, semantic="objet",    polarity=None),
    "kul":     dict(gloss="mange",         syllables=1, onset="plosive",    onset_voiced=False,
                    pharyngeal=False, semantic="action",   polarity=None),
    "lla":     dict(gloss="non",           syllables=1, onset="lateral",    onset_voiced=True,
                    pharyngeal=False, semantic="polarite", polarity="no"),
    "lma":     dict(gloss="eau",           syllables=1, onset="lateral",    onset_voiced=True,
                    pharyngeal=False, semantic="objet",    polarity=None),
    "makla":   dict(gloss="nourriture",    syllables=2, onset="nasal",      onset_voiced=True,
                    pharyngeal=False, semantic="objet",    polarity=None),
    "n3es":    dict(gloss="dormir",        syllables=1, onset="nasal",      onset_voiced=True,
                    pharyngeal=True,  semantic="action",   polarity=None),
    "shukran": dict(gloss="merci",         syllables=2, onset="fricative",  onset_voiced=False,
                    pharyngeal=False, semantic="politesse", polarity=None),
    "tbib":    dict(gloss="medecin",       syllables=2, onset="plosive",    onset_voiced=False,
                    pharyngeal=False, semantic="objet",    polarity=None),
    "wqef":    dict(gloss="arrete",        syllables=1, onset="glide",      onset_voiced=True,
                    pharyngeal=False, semantic="action",   polarity=None),
}

assert set(WORD_FEATURES) == set(WORDS), "WORD_FEATURES doit couvrir exactement WORDS"


def _mapped(y, key):
    return np.asarray([WORD_FEATURES[w][key] for w in y], dtype=object)


def contrast_15(y):
    """Les 15 mots tels quels. Hasard 1/15."""
    return np.asarray(y, dtype=object), None


def contrast_yes_no(y):
    """``Ah`` contre ``lla``. Le contraste le plus utile cliniquement et le plus
    puissant : 120 essais par classe et par session, 960 contre 960 au total."""
    labels = _mapped(y, "polarity")
    mask = np.isin(labels, ["yes", "no"])
    return labels, mask


def contrast_syllables(y):
    """Une contre deux syllabes. Si l'imagerie porte une trace temporelle, c'est
    la dimension la plus susceptible de la reveler."""
    return _mapped(y, "syllables").astype(str), None


def contrast_onset(y):
    """Classe articulatoire du premier phoneme."""
    return _mapped(y, "onset"), None


def contrast_pharyngeal(y):
    """Presence de la pharyngale /ʕ/, propre a l'arabe et fortement marquee."""
    return _mapped(y, "pharyngeal").astype(str), None


def contrast_semantic(y):
    """Categorie d'usage : requete, objet, action, etat, polarite, politesse."""
    return _mapped(y, "semantic"), None


CONTRASTS = {
    "mots15":      (contrast_15, 1 / 15),
    "oui_non":     (contrast_yes_no, 1 / 2),
    "syllabes":    (contrast_syllables, None),
    "attaque":     (contrast_onset, None),
    "pharyngale":  (contrast_pharyngeal, None),
    "semantique":  (contrast_semantic, None),
}


def apply_contrast(name: str, y) -> tuple[np.ndarray, np.ndarray, float]:
    """Renvoie (etiquettes, masque des essais retenus, niveau du hasard).

    Le hasard est celui d'un classifieur qui repondrait toujours la classe
    majoritaire, pas 1/n_classes : avec des classes desequilibrees (six mots sur
    quinze contiennent la pharyngale) 1/n_classes sous-estime le hasard et rend
    significatif un decodeur qui n'a rien appris.
    """
    fn, declared = CONTRASTS[name]
    labels, mask = fn(y)
    if mask is None:
        mask = np.ones(len(labels), dtype=bool)
    kept = labels[mask]
    _, counts = np.unique(kept, return_counts=True)
    chance = float(counts.max() / counts.sum())
    if declared is not None:
        chance = max(chance, declared)
    return kept, mask, chance
