"""Analyse de similarite representationnelle, avec distances validees croisees.

La classification repond par oui ou non a "peut-on lire le mot". L'ASR repond a
une question plus fine et nettement plus sensible : "la geometrie des 15 mots
dans l'espace EEG ressemble-t-elle a leur geometrie phonologique ou
semantique ?". Elle agrege l'information des 105 paires de mots au lieu de la
resumer en une exactitude, et elle peut detecter une structure la ou un
classifieur reste au hasard.

La distance employee est la distance de Mahalanobis validee croisee
(``crossnobis``) : le produit scalaire est pris entre deux plis independants,
donc son esperance est nulle en l'absence d'effet. Une distance de Mahalanobis
ordinaire est au contraire toujours positive, ce qui rend son interpretation
impossible sans distribution nulle.

Reference : Walther et al., "Reliability of dissimilarity measures for
multi-voxel pattern analysis", NeuroImage 2016.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from .contrasts import WORD_FEATURES


def _shrunk_precision(residuals: np.ndarray) -> np.ndarray:
    """Inverse de la covariance du bruit, avec retrecissement diagonal."""
    from sklearn.covariance import ledoit_wolf

    cov, _ = ledoit_wolf(residuals)
    return np.linalg.pinv(cov)


def crossnobis_rdm(X: np.ndarray, y: np.ndarray, folds: np.ndarray,
                   labels: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
    """Matrice de dissimilarite validee croisee entre conditions.

    ``X`` est de forme (n_trials, n_features) : les motifs, deja aplatis.
    ``folds`` attribue chaque essai a un pli (typiquement la session).
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    folds = np.asarray(folds)
    labels = labels or sorted(np.unique(y).tolist())
    n = len(labels)

    fold_ids = np.unique(folds)
    means = np.full((len(fold_ids), n, X.shape[1]), np.nan)
    residuals = []
    for fi, f in enumerate(fold_ids):
        for li, lab in enumerate(labels):
            mask = (folds == f) & (y == lab)
            if mask.sum() == 0:
                continue
            means[fi, li] = X[mask].mean(axis=0)
            residuals.append(X[mask] - means[fi, li])
    precision = _shrunk_precision(np.concatenate(residuals))

    rdm = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            products = []
            for a in range(len(fold_ids)):
                for b in range(len(fold_ids)):
                    if a == b:
                        continue
                    da = means[a, i] - means[a, j]
                    db = means[b, i] - means[b, j]
                    if np.isnan(da).any() or np.isnan(db).any():
                        continue
                    products.append(da @ precision @ db)
            value = float(np.mean(products)) if products else np.nan
            rdm[i, j] = rdm[j, i] = value
    return rdm, labels


def _levenshtein(a: str, b: str) -> int:
    """Distance d'edition, en programmation dynamique sur une seule ligne."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(previous[j] + 1,          # suppression
                               current[j - 1] + 1,       # insertion
                               previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


#: modeles derives de la seule chaine d'indice, sans annotation linguistique.
#: Ils servent de controle : si la geometrie neurale des mots imagines
#: correspond a la forme ECRITE de l'indice, ce qui est decode est en partie
#: orthographique et non phonologique. C'est l'objection qu'un relecteur posera
#: sur un indice visuel, quel que soit l'alphabet ; la tester vaut mieux que
#: l'argumenter.
ORTHOGRAPHIC_KEYS = ("orthographic", "cue_length")


def model_rdm(labels: list[str], key: str) -> np.ndarray:
    """RDM theorique : 0 si les deux mots partagent la modalite, 1 sinon.

    Deux exceptions, ordinales et non categorielles :

    - ``orthographic`` : distance d'edition entre les deux chaines d'indice,
      normalisee par la longueur de la plus longue, donc dans [0, 1] ;
    - ``cue_length`` : difference absolue du nombre de caracteres affiches.

    Les deux dernieres ne demandent aucune annotation : elles se lisent sur la
    transcription elle-meme. Elles sont la pour que la contribution
    orthographique d'un indice visuel soit mesuree et non supposee.
    """
    n = len(labels)
    rdm = np.zeros((n, n))

    if key in ORTHOGRAPHIC_KEYS:
        for i in range(n):
            for j in range(n):
                a, b = str(labels[i]), str(labels[j])
                if key == "cue_length":
                    rdm[i, j] = abs(len(a) - len(b))
                else:
                    longest = max(len(a), len(b))
                    rdm[i, j] = _levenshtein(a, b) / longest if longest else 0.0
        return rdm

    values = [WORD_FEATURES[w][key] for w in labels]
    for i in range(n):
        for j in range(n):
            rdm[i, j] = float(values[i] != values[j])
    return rdm


def _lower(rdm: np.ndarray) -> np.ndarray:
    idx = np.tril_indices(rdm.shape[0], k=-1)
    return rdm[idx]


def mantel_test(neural: np.ndarray, model: np.ndarray, n_perm: int = 10000,
                seed: int = 0) -> tuple[float, float]:
    """Correlation de Spearman entre deux RDM, testee par permutation des mots.

    Les entrees d'une RDM ne sont pas independantes : permuter directement les
    105 valeurs donnerait un p faux. La permutation porte donc sur l'ordre des
    mots, ce qui preserve la structure de dependance.
    """
    if np.isnan(neural).any():
        # une condition entierement absente d'un pli rendrait la permutation
        # incoherente : on retire la condition plutot que de masquer des paires
        bad = np.flatnonzero(np.isnan(neural).all(axis=1))
        keep = np.setdiff1d(np.arange(neural.shape[0]), bad)
        neural = neural[np.ix_(keep, keep)]
        model = model[np.ix_(keep, keep)]
    if np.isnan(neural).any():
        raise ValueError("RDM neurale incomplete apres retrait des conditions vides")

    a, b = _lower(neural), _lower(model)
    observed = stats.spearmanr(a, b).statistic

    rng = np.random.default_rng(seed)
    n = neural.shape[0]
    null = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(n)
        null[i] = stats.spearmanr(_lower(neural[np.ix_(perm, perm)]), b).statistic

    pvalue = (1.0 + np.sum(null >= observed)) / (n_perm + 1.0)
    return float(observed), float(pvalue)
