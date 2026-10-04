"""Decodage direct des 15 mots darija imagines a partir de l'EEG du scalp.

Le paquet est organise autour d'une seule idee : rendre impossible, par
construction, les fuites qui gonflent les resultats publies en parole imaginee.
Toute normalisation qui utilise les etiquettes est ajustee sur le pli
d'entrainement seul, les epoques ne contiennent jamais l'indice visuel, et les
plis respectent le regroupement par session.
"""

__version__ = "0.1.0"

WORDS = (
    "3afak", "3awnni", "3tini", "3yan", "Ah", "bghit", "dwa", "kul",
    "lla", "lma", "makla", "n3es", "shukran", "tbib", "wqef",
)
N_WORDS = len(WORDS)
CHANCE_15 = 1.0 / N_WORDS
