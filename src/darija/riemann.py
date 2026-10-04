"""Geometrie riemannienne sur les matrices de covariance.

C'est la famille de methodes que la ligne de base actuelle n'a pas essayee : les
resultats publies reposent sur une LDA appliquee a des enveloppes de puissance
par bande, ce qui jette toute la structure de covariance spatiale. Sur les jeux
EEG de petite taille, la representation tangente aux matrices de covariance est
systematiquement la reference classique la plus solide, et elle est ici
implementee sans dependance a pyriemann pour rester executable sur un noyau
Kaggle hors ligne.

Reperes : Barachant et al., "Multiclass brain-computer interface classification
by Riemannian geometry", IEEE TBME 2012 ; Congedo et al., "Riemannian geometry
for EEG-based brain-computer interfaces", J. Neural Eng. 2017 ; Zanini et al.,
"Transfer learning: a Riemannian geometry framework", IEEE TBME 2018 (pour le
recentrage par session).
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


# --------------------------------------------------------------------------
# estimation des covariances
# --------------------------------------------------------------------------

def covariances(X: np.ndarray, estimator: str = "oas") -> np.ndarray:
    """Covariance spatiale par essai.

    ``X`` est de forme (n_trials, n_channels, n_times). Avec 64 canaux et des
    fenetres courtes, la covariance empirique est mal conditionnee : le
    retrecissement (OAS ou Ledoit-Wolf) n'est pas un raffinement optionnel, il
    conditionne la validite de tout ce qui suit.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 3:
        raise ValueError(f"attendu (n_trials, n_channels, n_times), recu {X.shape}")

    if estimator == "scm":
        Xc = X - X.mean(axis=-1, keepdims=True)
        covs = Xc @ Xc.transpose(0, 2, 1) / (X.shape[-1] - 1)
    elif estimator in ("oas", "lwf"):
        from sklearn.covariance import ledoit_wolf, oas

        fn = oas if estimator == "oas" else ledoit_wolf
        covs = np.stack([fn(x.T)[0] for x in X])
    else:
        raise ValueError(f"estimateur inconnu : {estimator}")

    return _regularize(covs)


def _regularize(covs: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Garantit la definie-positivite stricte, requise par le logarithme."""
    n = covs.shape[-1]
    trace = np.trace(covs, axis1=-2, axis2=-1)[..., None, None] / n
    return covs + eps * trace * np.eye(n)


# --------------------------------------------------------------------------
# fonctions matricielles sur les matrices symetriques definies positives
# --------------------------------------------------------------------------

def _apply_eig(C: np.ndarray, fn, positive: bool = True) -> np.ndarray:
    """Applique ``fn`` au spectre d'une matrice symetrique.

    ``positive`` plafonne les valeurs propres par le bas. C'est indispensable
    pour la racine, l'inverse et le logarithme, qui exigent une matrice definie
    positive et qu'un zero numerique ferait exploser. Mais c'est FAUX pour
    l'exponentielle : son argument est un vecteur tangent, symetrique mais
    INDEFINI, dont les valeurs propres negatives portent la moitie de
    l'information. Les ecraser a 1e-15 transforme exp(lambda) < 1 en 1 et
    detruit la direction de descente.
    """
    w, V = np.linalg.eigh(C)
    if positive:
        w = np.maximum(w, 1e-15)
    w = fn(w)
    return (V * w[..., None, :]) @ V.transpose(*range(V.ndim - 2), -1, -2)


def sqrtm(C: np.ndarray) -> np.ndarray:
    return _apply_eig(C, np.sqrt)


def invsqrtm(C: np.ndarray) -> np.ndarray:
    return _apply_eig(C, lambda w: 1.0 / np.sqrt(w))


def logm(C: np.ndarray) -> np.ndarray:
    return _apply_eig(C, np.log)


def expm(C: np.ndarray) -> np.ndarray:
    """Exponentielle matricielle d'une matrice symetrique QUELCONQUE.

    Sans ``positive=False`` la moyenne de Karcher ne converge jamais : sa
    direction de descente est un vecteur tangent indefini, et l'ecrasement des
    valeurs propres negatives la rend constante d'une iteration a l'autre.
    """
    return _apply_eig(C, np.exp, positive=False)


def distance_riemann(A: np.ndarray, B: np.ndarray) -> float:
    """Distance affine-invariante, la metrique naturelle du cone des SPD.

    Le produit A^-1 B a bien des valeurs propres reelles positives, mais il
    n'est PAS symetrique : lui appliquer ``eigvalsh``, qui ne lit qu'un triangle,
    donne un resultat faux. On passe donc par la congruence A^-1/2 B A^-1/2, qui
    est symetrique par construction et coherente avec l'espace tangent.
    """
    Am12 = invsqrtm(A)
    M = Am12 @ B @ Am12
    w = np.linalg.eigvalsh(0.5 * (M + M.T))
    return float(np.sqrt(np.sum(np.log(np.maximum(w, 1e-15)) ** 2)))


def geometric_mean(covs: np.ndarray, tol: float = 1e-8, max_iter: int = 60,
                   step: float = 1.0) -> np.ndarray:
    """Moyenne de Karcher au sens de la metrique affine-invariante.

    L'initialisation par la moyenne arithmetique puis la descente de gradient sur
    la variete converge quadratiquement : quatre a six iterations suffisent pour
    atteindre la precision machine sur des covariances EEG. Le nombre
    d'iterations reellement effectuees est expose par
    ``geometric_mean.last_iterations``, ce qui permet de le surveiller.
    """
    covs = np.asarray(covs, dtype=np.float64)
    C = covs.mean(axis=0)
    iterations = 0
    for _ in range(max_iter):
        iterations += 1
        Cm12, C12 = invsqrtm(C), sqrtm(C)
        T = logm(Cm12 @ covs @ Cm12).mean(axis=0)
        C = C12 @ expm(step * T) @ C12
        C = 0.5 * (C + C.T)
        if np.linalg.norm(T, ord="fro") < tol:
            break
    # la convergence est quadratique : quatre a six iterations suffisent pour
    # atteindre la precision machine. Atteindre max_iter signale un probleme.
    geometric_mean.last_iterations = iterations
    return C


# --------------------------------------------------------------------------
# espace tangent
# --------------------------------------------------------------------------

def tangent_vectors(covs: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Projette les covariances dans l'espace tangent au point ``reference``.

    Le resultat est un vecteur de dimension n(n+1)/2 par essai : le triangle
    superieur du logarithme, les termes hors diagonale ponderes par sqrt(2) pour
    que la norme euclidienne du vecteur egale la norme de Frobenius de la
    matrice. Une fois dans l'espace tangent, tout classifieur lineaire ordinaire
    devient applicable.
    """
    ref_m12 = invsqrtm(reference)
    S = logm(ref_m12 @ covs @ ref_m12)
    n = S.shape[-1]
    idx = np.triu_indices(n)
    weights = np.where(idx[0] == idx[1], 1.0, np.sqrt(2.0))
    return S[..., idx[0], idx[1]] * weights


class TangentSpace(BaseEstimator, TransformerMixin):
    """Transformateur sklearn : covariances -> vecteurs tangents.

    Le point de reference est la moyenne geometrique du pli d'ENTRAINEMENT seul.
    C'est le detail qui distingue une evaluation honnete d'une fuite : calculer
    la reference sur l'ensemble des donnees ferait passer de l'information du
    pli de test dans la representation.
    """

    def __init__(self, metric_tol: float = 1e-8, max_iter: int = 60):
        self.metric_tol = metric_tol
        self.max_iter = max_iter

    def fit(self, X, y=None):
        self.reference_ = geometric_mean(X, tol=self.metric_tol, max_iter=self.max_iter)
        return self

    def transform(self, X):
        return tangent_vectors(np.asarray(X, dtype=np.float64), self.reference_)


# --------------------------------------------------------------------------
# recentrage par session
# --------------------------------------------------------------------------

def recenter_by_group(covs: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Ramene chaque session a la meme origine sur la variete.

    Chaque session est transportee de sorte que sa propre moyenne geometrique
    devienne l'identite. C'est une adaptation de domaine NON SUPERVISEE : elle
    n'utilise aucune etiquette, seulement l'existence d'un lot de donnees par
    session. Pour une ICM reelle c'est legitime, puisqu'on dispose toujours de
    donnees de calibration non etiquetees ; mais comme elle touche aussi la
    session de test, tout resultat obtenu ainsi doit etre rapporte a part du
    resultat sans recentrage.
    """
    covs = np.asarray(covs, dtype=np.float64)
    out = np.empty_like(covs)
    for g in np.unique(groups):
        mask = groups == g
        Mm12 = invsqrtm(geometric_mean(covs[mask]))
        out[mask] = Mm12 @ covs[mask] @ Mm12
    return out


# --------------------------------------------------------------------------
# covariances augmentees (plongement par retards)
# --------------------------------------------------------------------------

def augmented_covariances(X: np.ndarray, order: int = 4, lag: int = 8,
                          estimator: str = "oas") -> np.ndarray:
    """Covariance d'un signal empile avec ses versions retardees.

    La covariance spatiale ordinaire est invariante par permutation temporelle :
    elle ignore entierement la dynamique. Empiler ``order`` copies decalees de
    ``lag`` echantillons avant de calculer la covariance reintroduit la structure
    autoregressive, au prix d'une matrice ``order`` fois plus grande.

    Reference : Carrara & Papadopoulo, "Classification of BCI-EEG based on
    augmented covariance matrix", 2023.
    """
    X = np.asarray(X, dtype=np.float64)
    n_trials, n_channels, n_times = X.shape
    span = lag * (order - 1)
    if span >= n_times:
        raise ValueError(f"order={order}, lag={lag} depasse {n_times} echantillons")

    stacked = np.concatenate(
        [X[..., k * lag: n_times - span + k * lag] for k in range(order)], axis=1
    )
    return covariances(stacked, estimator=estimator)
