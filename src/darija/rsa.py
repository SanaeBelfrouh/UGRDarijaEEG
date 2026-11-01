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

    from scipy.linalg import LinAlgError, cho_factor, cho_solve

    cov, _ = ledoit_wolf(residuals)
    # la covariance retrecie est definie positive : Cholesky donne la meme
    # inverse que pinv (a 1e-14 pres) quarante fois plus vite, ce qui rend
    # possible de la re-estimer a chaque permutation
    try:
        return cho_solve(cho_factor(cov), np.eye(len(cov)))
    except LinAlgError:
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


def _fold_means(X, y, folds, labels, fold_ids):
    means = np.full((len(fold_ids), len(labels), X.shape[1]), np.nan)
    for fi, f in enumerate(fold_ids):
        in_fold = folds == f
        for li, lab in enumerate(labels):
            mask = in_fold & (y == lab)
            if mask.any():
                means[fi, li] = X[mask].mean(axis=0)
    return means


def _mean_crossnobis(means: np.ndarray, precision: np.ndarray) -> float:
    """Moyenne des dissimilarites crossnobis hors diagonale, par produits de Gram.

    Pour deux plis a != b, (m_ai - m_aj) P (m_bi - m_bj) = G_ii + G_jj - G_ij - G_ji
    avec G = M_a P M_b^T. Une paire de mots absente d'un pli est ignoree, comme
    dans :func:`crossnobis_rdm`.
    """
    n_folds, n = means.shape[:2]
    MP = means @ precision
    total = np.zeros((n, n))
    count = np.zeros((n, n))
    for a in range(n_folds):
        for b in range(n_folds):
            if a == b:
                continue
            G = means[a] @ MP[b].T
            d = np.diag(G)
            D = d[:, None] + d[None, :] - G - G.T
            ok = np.isfinite(D)
            total[ok] += D[ok]
            count[ok] += 1
    iu = np.triu_indices(n, k=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        pair = total[iu] / count[iu]
    return float(np.nanmean(pair))


def _precision_for(X, y, folds, labels, fold_ids, means):
    residuals = []
    for fi, f in enumerate(fold_ids):
        for li, lab in enumerate(labels):
            mask = (folds == f) & (y == lab)
            if mask.any():
                residuals.append(X[mask] - means[fi, li])
    return _shrunk_precision(np.concatenate(residuals))


def mean_dissimilarity_test(X: np.ndarray, y: np.ndarray, folds: np.ndarray,
                            labels: list[str] | None = None, n_perm: int = 200,
                            seed: int = 0) -> tuple[float, float, float]:
    """La dissimilarite crossnobis moyenne entre mots depasse-t-elle zero ?

    L'estimateur est d'esperance nulle sous l'hypothese nulle, mais une moyenne
    positive ne dit rien seule : il faut sa distribution sous le hasard. Les
    etiquettes sont permutees A L'INTERIEUR de chaque pli (session), ce qui
    garde la structure des sessions et le nombre d'essais par mot et par pli,
    puis TOUT est recalcule, matrice de precision comprise.

    Garder la precision des vraies etiquettes sous permutation serait plus
    rapide mais faux : avec plus de dimensions que d'essais, les moyennes
    permutees tombent en partie dans le sous-espace des residus sur lequel la
    precision a ete estimee, et la nulle n'est plus comparable a l'observe (sur
    du bruit pur : observe -15, nulle +0,08). Test unilateral.

    Renvoie (moyenne observee, p, moyenne de la distribution nulle).
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    folds = np.asarray(folds)
    labels = labels or sorted(np.unique(y).tolist())
    fold_ids = np.unique(folds)

    def statistic(lab):
        means = _fold_means(X, lab, folds, labels, fold_ids)
        return _mean_crossnobis(means, _precision_for(X, lab, folds, labels, fold_ids, means))

    observed = statistic(y)
    rng = np.random.default_rng(seed)
    idx_by_fold = [np.flatnonzero(folds == f) for f in fold_ids]
    null = np.empty(n_perm)
    for k in range(n_perm):
        yp = y.copy()
        for idx in idx_by_fold:
            yp[idx] = y[idx][rng.permutation(len(idx))]
        null[k] = statistic(yp)
    p = (np.sum(null >= observed) + 1) / (n_perm + 1)
    return observed, float(p), float(null.mean())


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
