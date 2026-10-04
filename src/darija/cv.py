"""Schemas de validation croisee.

Trois schemas, qui ne repondent pas a la meme question, et dont la confusion est
la premiere source de desaccord dans la litterature sur la parole imaginee.

``leave_one_session_out``
    Le modele est entraine sur 7 sessions et teste sur la 8e. C'est le test le
    plus severe possible : il exige que le decodeur survive au deplacement de
    domaine entre sessions (impedances, repositionnement du bonnet, etat du
    sujet). Un resultat au hasard ici ne prouve PAS l'absence de signal, il
    prouve l'absence d'un signal stable entre sessions.

``within_session_blocked``
    Un modele par session, plis chronologiques contigus. C'est le test qui
    repond a "y a-t-il du signal, tout court". Les plis ne sont jamais melanges :
    des essais voisins dans le temps partagent une derive lente, et un melange
    aleatoire la transforme en information apparemment decodable.

``pooled_blocked``
    Plis chronologiques appliques simultanement a toutes les sessions. Compromis
    entre les deux : le modele voit plusieurs sessions a l'entrainement mais
    n'est jamais teste sur un essai temporellement adjacent a un essai
    d'entrainement de la meme session.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np


def leave_one_session_out(groups: np.ndarray) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Un pli par session : test = cette session, entrainement = toutes les autres."""
    groups = np.asarray(groups)
    for g in np.unique(groups):
        test = np.flatnonzero(groups == g)
        train = np.flatnonzero(groups != g)
        yield train, test


def _contiguous_blocks(order: np.ndarray, n_splits: int) -> list[np.ndarray]:
    """Positions triees par ordre temporel, coupees en ``n_splits`` blocs contigus."""
    return np.array_split(np.argsort(order), n_splits)


def within_session_blocked(groups: np.ndarray, trial_index: np.ndarray,
                           n_splits: int = 5) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Plis chronologiques a l'interieur de chaque session, session par session.

    Chaque pli renvoye ne contient les essais que d'UNE session : entrainement et
    test viennent de la meme session. Le modele est donc ajuste autant de fois
    qu'il y a de sessions x plis.
    """
    groups = np.asarray(groups)
    trial_index = np.asarray(trial_index)
    for g in np.unique(groups):
        where = np.flatnonzero(groups == g)
        blocks = _contiguous_blocks(trial_index[where], n_splits)
        for k in range(n_splits):
            test = where[blocks[k]]
            train = where[np.concatenate([blocks[j] for j in range(n_splits) if j != k])]
            yield train, test


def pooled_blocked(groups: np.ndarray, trial_index: np.ndarray,
                   n_splits: int = 5) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Le bloc chronologique k de chaque session forme ensemble le pli de test."""
    groups = np.asarray(groups)
    trial_index = np.asarray(trial_index)
    per_session = {}
    for g in np.unique(groups):
        where = np.flatnonzero(groups == g)
        per_session[g] = [where[b] for b in _contiguous_blocks(trial_index[where], n_splits)]

    for k in range(n_splits):
        test = np.concatenate([per_session[g][k] for g in per_session])
        train = np.concatenate(
            [per_session[g][j] for g in per_session for j in range(n_splits) if j != k]
        )
        yield np.sort(train), np.sort(test)


def subblock_grouped(groups: np.ndarray, trial_index: np.ndarray,
                     n_splits: int = 8, sub_block: np.ndarray | None = None
                     ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Plis groupes par sous-bloc d'enregistrement, toutes sessions confondues.

    C'est le schema « block-grouped » du manuscrit. Il est nettement plus
    permissif que le leave-one-session-out : le modele voit d'autres sous-blocs
    de la MEME session a l'entrainement, donc l'information propre a la session
    et a l'etat du bonnet lui reste disponible. Il est en revanche plus strict
    qu'un melange aleatoire, puisqu'aucun essai de test n'est temporellement
    voisin d'un essai d'entrainement a l'interieur d'un sous-bloc.

    Le groupe est le couple (session, sous-bloc) : deux sous-blocs portant le
    meme index dans deux sessions differentes sont des groupes distincts.
    """
    if sub_block is None:
        raise ValueError("subblock_grouped exige sub_block")
    groups = np.asarray(groups)
    sub_block = np.asarray(sub_block)
    keys = np.asarray([f"{g}|{b}" for g, b in zip(groups, sub_block)])
    unique = np.unique(keys)
    folds = np.array_split(np.arange(len(unique)), min(n_splits, len(unique)))
    for fold in folds:
        held = set(unique[fold])
        mask = np.asarray([k in held for k in keys])
        yield np.flatnonzero(~mask), np.flatnonzero(mask)


SCHEMES = {
    "loso": leave_one_session_out,
    "within_session": within_session_blocked,
    "pooled_blocked": pooled_blocked,
    "subblock": subblock_grouped,
}


def make_splits(scheme: str, groups: np.ndarray, trial_index: np.ndarray,
                n_splits: int = 5, sub_block: np.ndarray | None = None
                ) -> list[tuple[np.ndarray, np.ndarray]]:
    """Materialise les plis d'un schema nomme."""
    if scheme == "loso":
        return list(leave_one_session_out(groups))
    if scheme == "subblock":
        return list(subblock_grouped(groups, trial_index, n_splits, sub_block=sub_block))
    if scheme not in SCHEMES:
        raise ValueError(f"schema inconnu : {scheme}")
    return list(SCHEMES[scheme](groups, trial_index, n_splits))
