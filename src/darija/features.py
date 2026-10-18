"""Representations : banc de filtres, covariances par bande, CSP, puissance.

Les fonctions qui produisent des covariances renvoient un dictionnaire bande ->
tableau de covariances. Les concatener dans l'espace tangent (une reference par
bande) est prefere a une grande matrice bloc-diagonale : le resultat est
identique en information et evite d'inverser une matrice 320 x 320.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import hilbert
from sklearn.base import BaseEstimator, TransformerMixin

from .preprocess import bandpass
from .riemann import augmented_covariances, covariances

#: banc par defaut. 30-45 Hz s'arrete avant le secteur ; au-dela, la
#: caracterisation haute frequence deja menee ne montre rien en imagine.
FILTER_BANK = (
    (0.5, 4.0),    # delta
    (4.0, 8.0),    # theta
    (8.0, 13.0),   # alpha
    (13.0, 20.0),  # beta bas
    (20.0, 30.0),  # beta haut
    (30.0, 45.0),  # gamma bas
)


def filter_bank(X: np.ndarray, sfreq: float, bands=FILTER_BANK) -> dict[tuple, np.ndarray]:
    """Applique chaque bande du banc et renvoie les signaux filtres."""
    return {band: bandpass(X, sfreq, band[0], band[1]) for band in bands}


def bank_covariances(X: np.ndarray, sfreq: float, bands=FILTER_BANK,
                     estimator: str = "oas") -> dict[tuple, np.ndarray]:
    """Une matrice de covariance par bande et par essai."""
    return {band: covariances(sig, estimator=estimator)
            for band, sig in filter_bank(X, sfreq, bands).items()}


def bank_augmented_covariances(X: np.ndarray, sfreq: float, bands=FILTER_BANK,
                               order: int = 3, lag: int = 8,
                               estimator: str = "oas") -> dict[tuple, np.ndarray]:
    """Covariances augmentees par retards, bande par bande."""
    return {band: augmented_covariances(sig, order=order, lag=lag, estimator=estimator)
            for band, sig in filter_bank(X, sfreq, bands).items()}


def band_power(X: np.ndarray, sfreq: float, bands=FILTER_BANK,
               n_windows: int = 4, log: bool = True) -> np.ndarray:
    """Puissance d'enveloppe par canal, bande et sous-fenetre temporelle.

    C'est la representation de la ligne de base actuelle, conservee ici comme
    point de comparaison : tout gain rapporte pour une methode riemannienne doit
    etre lu par rapport a elle, sur exactement les memes plis.
    """
    feats = []
    for band, sig in filter_bank(X, sfreq, bands).items():
        env = np.abs(hilbert(sig, axis=-1)) ** 2
        chunks = np.array_split(np.arange(env.shape[-1]), n_windows)
        feats.append(np.concatenate([env[..., c].mean(axis=-1) for c in chunks], axis=-1))
    out = np.concatenate(feats, axis=-1)
    return np.log(out + 1e-12) if log else out


# --------------------------------------------------------------------------
# CSP multiclasse un-contre-tous
# --------------------------------------------------------------------------

class MulticlassCSP(BaseEstimator, TransformerMixin):
    """CSP un-contre-tous, ``n_components`` filtres par classe.

    Les filtres sont appris par decomposition simultanee de la covariance de la
    classe et de celle du reste. Avec 15 classes et ``n_components=2`` cela fait
    30 filtres ; au-dela le nombre de parametres devient comparable au nombre
    d'essais par classe et le sur-apprentissage domine.
    """

    def __init__(self, n_components: int = 2, log: bool = True):
        self.n_components = n_components
        self.log = log

    def fit(self, X, y):
        covs = covariances(X)
        self.classes_ = np.unique(y)
        filters = []
        for cls in self.classes_:
            mask = np.asarray(y) == cls
            c1 = covs[mask].mean(axis=0)
            c2 = covs[~mask].mean(axis=0)
            w, V = np.linalg.eigh(np.linalg.solve(c1 + c2, c1))
            order = np.argsort(w)[::-1]
            V = V[:, order]
            # extremes du spectre : variance maximale pour la classe, puis minimale
            filters.append(np.concatenate(
                [V[:, :self.n_components], V[:, -self.n_components:]], axis=1))
        self.filters_ = np.concatenate(filters, axis=1).T   # (n_filters, n_channels)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        projected = self.filters_ @ X                       # (n_trials, n_filters, n_times)
        var = projected.var(axis=-1)
        var = var / (var.sum(axis=-1, keepdims=True) + 1e-12)
        return np.log(var + 1e-12) if self.log else var


class CovariancesTransformer(BaseEstimator, TransformerMixin):
    """Enveloppe sklearn autour de :func:`darija.riemann.covariances`."""

    def __init__(self, estimator: str = "oas"):
        self.estimator = estimator

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return covariances(X, estimator=self.estimator)


# --------------------------------------------------------------------------
# reduction de rang avant estimation des covariances
# --------------------------------------------------------------------------

#: banc de filtres du manuscrit, a preferer au banc par defaut pour reproduire
#: les chiffres publies.
MANUSCRIPT_BANK = ((1.0, 8.0), (8.0, 14.0), (14.0, 26.0), (26.0, 40.0))

#: banc du manuscrit prolonge par deux bandes gamma hautes, pour l'analyse de
#: sensibilite en bande large (scripts/10_high_frequency.py). Les bandes evitent
#: le secteur a 50 et 100 Hz, deja retire par le notch. Exige une archive filtree
#: au moins jusqu'a 120 Hz et echantillonnee a 300 Hz ou plus.
HIGH_BANK = MANUSCRIPT_BANK + ((55.0, 95.0), (105.0, 120.0))


def rank_reduce(X: np.ndarray, groups: np.ndarray, n_components: int = 32,
                scope: str = "global") -> np.ndarray:
    """Projette les epoques sur les ``n_components`` premiers vecteurs propres.

    Deux raisons de le faire, et la premiere est une necessite d'estimation. A
    rang plein, un banc de quatre bandes sur 64 canaux produit un vecteur
    tangent de 4 x 2080 = 8320 dimensions pour environ 1575 essais
    d'entrainement : le probleme est sous-determine, et la solution d'une LDA a
    retrecissement y est largement arbitraire. Reduire a 32 composantes ramene
    la dimension a 4 x 528 = 2112. La seconde raison est le cout : une
    decomposition propre 32x32 coute huit fois moins qu'une 64x64.

    ``scope`` decide de la base :

    ``"global"``
        une seule base, calculee sur la covariance moyenne de toutes les
        sessions. Les essais de toutes les sessions vivent alors dans le meme
        systeme de coordonnees, ce qui est la condition pour qu'un classifieur
        entraine sur sept sessions signifie quelque chose sur la huitieme.

    ``"session"``
        une base par session. Attention : les vecteurs propres ont un signe
        arbitraire et leur ordre peut permuter entre deux valeurs propres
        voisines, donc la composante 7 d'une session ne correspond pas a la
        composante 7 d'une autre. Les termes hors diagonale des covariances
        changent alors de signe d'une session a l'autre. Cela ne peut pas
        fabriquer un faux positif inter-sessions — cela degrade le transfert —
        mais il faut savoir laquelle des deux variantes a produit un chiffre
        avant de le publier.

    Aucune etiquette n'est utilisee. Comme le recentrage, c'est une operation
    non supervisee qui voit les donnees de la session de test : legitime pour
    une ICM, a rapporter comme telle.
    """
    X = np.asarray(X, dtype=np.float64)
    groups = np.asarray(groups)
    n_components = min(n_components, X.shape[1])

    def basis(block: np.ndarray) -> np.ndarray:
        mean_cov = covariances(block).mean(axis=0)
        _, V = np.linalg.eigh(mean_cov)
        # eigh trie par valeur propre croissante : les dernieres colonnes
        # portent le plus de variance
        return V[:, -n_components:]

    if scope == "global":
        P = basis(X)
        return np.einsum("kc,nct->nkt", P.T, X)

    if scope != "session":
        raise ValueError(f"scope inconnu : {scope} (attendu 'global' ou 'session')")

    out = np.empty((X.shape[0], n_components, X.shape[2]))
    for g in np.unique(groups):
        mask = groups == g
        out[mask] = np.einsum("kc,nct->nkt", basis(X[mask]).T, X[mask])
    return out


# --------------------------------------------------------------------------
# filtrage spatial supervise (xDAWN)
# --------------------------------------------------------------------------

class Xdawn(BaseEstimator, TransformerMixin):
    """Filtrage spatial supervise, un jeu de filtres par classe.

    xDAWN cherche les directions spatiales qui maximisent le rapport entre la
    puissance de la reponse EVOQUEE d'une classe et celle du bruit de fond. Pour
    chaque classe c on forme la reponse moyenne P_c, puis on resout le probleme
    aux valeurs propres generalise

        maximiser  w' (P_c P_c') w  /  w' Sigma_bruit w

    et on garde les ``n_filters`` premieres directions. Les filtres des classes
    sont ensuite concatenes, ce qui donne n_classes x n_filters canaux virtuels.

    ATTENTION, et c'est la raison pour laquelle ce transformateur vit dans
    l'etage CLASSIFIEUR et non dans le prefixe : ``fit`` utilise les etiquettes.
    L'ajuster hors du pli d'entrainement ferait passer de l'information du pli de
    test dans la representation, ce qui est exactement la fuite que ce paquet
    cherche a rendre impossible. Le manuscrit precise « in-fold » pour la meme
    raison.

    Reference : Rivet et al., « xDAWN algorithm to enhance evoked potentials »,
    IEEE TBME 2009.

    ``augment`` prepend aux canaux virtuels les PROTOTYPES de classe, c'est-a-dire
    les reponses evoquees moyennes passees par les memes filtres. La covariance du
    signal ainsi augmente contient alors, dans ses termes croises, la correlation
    entre l'essai et chaque prototype — et c'est de la que xDAWN tire l'essentiel
    de sa puissance pour les reponses evoquees. Sans cette augmentation le
    transformateur ne fait que reduire la dimension, ce qui peut degrader.

    Reference pour l'augmentation : Barachant & Congedo, « A plug-and-play P300
    BCI using information geometry », 2014.
    """

    def __init__(self, n_filters: int = 2, shrinkage: float = 1e-2,
                 augment: bool = True):
        self.n_filters = n_filters
        self.shrinkage = shrinkage
        self.augment = augment

    def fit(self, X, y):
        from scipy.linalg import eigh

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        n_channels = X.shape[1]
        self.classes_ = np.unique(y)

        # bruit de fond : covariance de tous les essais concatenes. Le
        # retrecissement vers l'identite est indispensable, sinon le probleme
        # generalise est mal conditionne des que les canaux sont correles — et
        # ils le sont fortement ici.
        flat = X.transpose(1, 0, 2).reshape(n_channels, -1)
        noise = flat @ flat.T / flat.shape[1]
        noise += self.shrinkage * np.trace(noise) / n_channels * np.eye(n_channels)

        filters = []
        for label in self.classes_:
            evoked = X[y == label].mean(axis=0)
            signal = evoked @ evoked.T / evoked.shape[1]
            signal = 0.5 * (signal + signal.T)
            _, vectors = eigh(signal, noise)
            # eigh trie par valeur propre croissante : les dernieres colonnes
            # maximisent le rapport signal / bruit
            filters.append(vectors[:, -self.n_filters:])

        self.filters_ = np.concatenate(filters, axis=1)      # (n_channels, n_virtuels)

        if self.augment:
            # Prototype de la classe c : sa reponse evoquee passee par SES PROPRES
            # filtres, et non par l'ensemble des filtres. La distinction est une
            # question de tractabilite : avec 15 classes et 2 filtres, passer
            # chaque evoquee par les 30 filtres donnerait 450 lignes, donc une
            # covariance augmentee de 480 canaux et 115 440 dimensions tangentes
            # pour 1575 essais d'entrainement. Un prototype par bloc de filtres
            # donne 30 lignes, soit 60 canaux augmentes et 1830 dimensions.
            blocks = []
            for i, label in enumerate(self.classes_):
                own = filters[i]                          # (n_channels, n_filters)
                blocks.append(own.T @ X[y == label].mean(axis=0))
            self.prototypes_ = np.concatenate(blocks, axis=0)   # (n_classes*n_filters, T)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        projected = np.einsum("ck,nct->nkt", self.filters_, X)
        if not self.augment:
            return projected
        proto = np.broadcast_to(self.prototypes_,
                                (len(X), *self.prototypes_.shape))
        return np.concatenate([proto, projected], axis=1)
