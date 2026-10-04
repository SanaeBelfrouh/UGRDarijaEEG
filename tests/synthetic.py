"""Generateur de donnees synthetiques : le temoin positif du code lui-meme.

Avant de conclure quoi que ce soit d'un resultat nul sur les vraies donnees, il
faut avoir montre que le pipeline detecte un effet quand il y en a un. Ce
generateur permet cela : ``effect`` regle la force du motif spatial propre a
chaque mot, et ``session_shift`` simule le deplacement de domaine entre sessions
qui rend la validation inter-sessions si difficile.
"""

from __future__ import annotations

import numpy as np


def make_dataset(
    *,
    n_sessions: int = 4,
    n_words: int = 15,
    n_per_word: int = 16,
    n_channels: int = 16,
    n_times: int = 250,
    sfreq: float = 250.0,
    effect: float = 0.0,
    session_shift: float = 0.0,
    seed: int = 0,
):
    """Renvoie (X, words, groups, trial_index, sfreq).

    Le signal, quand ``effect > 0``, est une oscillation a 10 Hz projetee sur un
    motif spatial propre au mot. Il apparait donc dans la covariance spatiale, ce
    que les methodes riemanniennes doivent voir et ce qu'une simple puissance par
    canal voit moins bien.
    """
    rng = np.random.default_rng(seed)
    words = np.asarray([f"w{i:02d}" for i in range(n_words)], dtype=object)
    patterns = rng.normal(size=(n_words, n_channels))
    patterns /= np.linalg.norm(patterns, axis=1, keepdims=True)

    t = np.arange(n_times) / sfreq
    X, y, groups, order = [], [], [], []

    for s in range(n_sessions):
        # deplacement de domaine propre a la session : gains par canal (derive
        # d'impedance) et gain global. C'est un changement de blanchiment, donc
        # exactement la classe de deplacement que le recentrage riemannien
        # annule. Un repositionnement du bonnet ajouterait une rotation, que le
        # recentrage seul ne corrige pas : il faudrait un alignement de
        # Procruste. La distinction est importante et le test la respecte.
        gains = np.exp(session_shift * rng.normal(size=n_channels))
        mixing = np.diag(gains)
        session_start = len(X)
        for wi in range(n_words):
            for _ in range(n_per_word):
                noise = rng.normal(size=(n_channels, n_times))
                # bruit colore : plus de puissance en basse frequence, comme l'EEG
                noise = np.cumsum(noise, axis=-1) / np.sqrt(n_times)
                signal = np.zeros_like(noise)
                if effect > 0:
                    phase = rng.uniform(0, 2 * np.pi)
                    osc = np.sin(2 * np.pi * 10.0 * t + phase)
                    signal = effect * np.outer(patterns[wi], osc)
                X.append(mixing @ (noise + signal))
                y.append(words[wi])
                groups.append(f"S{s:03d}")

        # PsychoPy tire l'ordre des mots au hasard dans chaque bloc : le rang
        # chronologique doit donc etre independant du mot. Le generer en ordre
        # de mot rendrait les plis chronologiques intra-session equivalents a des
        # plis par classe, et l'entrainement ne verrait jamais la classe testee.
        order.extend(rng.permutation(len(X) - session_start).tolist())

    X = np.stack(X).astype(np.float64)
    perm = rng.permutation(len(X))       # l'ordre des lignes ne porte rien
    return (
        X[perm],
        np.asarray(y, dtype=object)[perm],
        np.asarray(groups)[perm],
        np.asarray(order)[perm],
        sfreq,
    )
