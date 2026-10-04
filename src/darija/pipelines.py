"""L'echelle de modeles, du plus simple au plus expressif.

L'ordre est volontaire. ``puissance_lda`` reproduit la ligne de base actuelle ;
tout ce qui suit doit etre lu comme un gain PAR RAPPORT a elle, sur exactement
les memes plis. Une methode qui ne bat pas la ligne de base sur les memes plis
n'apporte rien, quelle que soit sa sophistication.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import (
    FILTER_BANK,
    MANUSCRIPT_BANK,
    CovariancesTransformer,
    MulticlassCSP,
    Xdawn,
    band_power,
    bank_augmented_covariances,
    bank_covariances,
    rank_reduce,
)
from .riemann import (
    covariances,
    distance_riemann,
    geometric_mean,
    tangent_vectors,
    TangentSpace,
)


class BankTangentSpace(BaseEstimator, TransformerMixin):
    """Espace tangent bande par bande, puis concatenation.

    Une reference geometrique par bande, toutes ajustees sur le pli
    d'entrainement. Concatener dans l'espace tangent plutot que construire une
    grande matrice bloc-diagonale donne la meme information sans avoir a
    inverser une matrice de taille (n_bandes x n_canaux)."""

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)       # (n_trials, n_bands, c, c)
        self.references_ = [geometric_mean(X[:, b]) for b in range(X.shape[1])]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        return np.concatenate(
            [tangent_vectors(X[:, b], ref) for b, ref in enumerate(self.references_)], axis=1
        )


class MDM(BaseEstimator, ClassifierMixin):
    """Distance minimale a la moyenne riemannienne.

    Sans aucun hyperparametre a regler et avec un seul point par classe, ce
    classifieur ne peut pratiquement pas sur-apprendre. C'est pour cette raison
    qu'il vaut la peine d'etre rapporte meme s'il est moins performant : s'il
    est au hasard alors que la regression logistique tangente ne l'est pas, la
    difference vient probablement de la capacite du modele, pas du signal."""

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        self.classes_ = np.unique(y)
        self.means_ = np.stack([geometric_mean(X[np.asarray(y) == c]) for c in self.classes_])
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        distances = np.stack(
            [[distance_riemann(m, x) for m in self.means_] for x in X]
        )
        return self.classes_[np.argmin(distances, axis=1)]


def _logreg(C: float = 0.1) -> LogisticRegression:
    """Regression logistique multinomiale fortement regularisee.

    Avec 2080 dimensions tangentes pour 1575 essais d'entrainement, la
    regularisation n'est pas un reglage fin : sans elle le probleme est
    sous-determine et la solution est arbitraire."""
    # depuis scikit-learn 1.7 le multinomial est le comportement par defaut de
    # lbfgs et le parametre multi_class a ete retire
    return LogisticRegression(C=C, max_iter=3000, solver="lbfgs")


# --------------------------------------------------------------------------
# representations : calculees une fois, hors validation croisee
# --------------------------------------------------------------------------

def build_representations(X: np.ndarray, sfreq: float, *, bands=FILTER_BANK,
                          include: tuple[str, ...] | None = None,
                          n_components: int | None = None,
                          groups: np.ndarray | None = None,
                          rank_scope: str = "global") -> dict[str, np.ndarray]:
    """Calcule toutes les representations une seule fois.

    Le filtrage et l'estimation des covariances ne dependent d'aucune etiquette
    et sont identiques pour tous les essais : les sortir de la boucle de
    validation croisee ne cree pas de fuite et divise le temps de calcul par le
    nombre de plis.

    ``n_components`` applique une reduction de rang avant tout le reste (voir
    :func:`darija.features.rank_reduce`). C'est a la fois une necessite
    d'estimation pour les representations a banc de bandes et le levier de cout
    le plus efficace. ``None`` conserve le rang plein.
    """
    if n_components is not None:
        if groups is None:
            groups = np.zeros(len(X))
        X = rank_reduce(X, groups, n_components=n_components, scope=rank_scope)

    wanted = set(include) if include else None

    def want(name):
        return wanted is None or name in wanted

    out: dict[str, np.ndarray] = {}
    if want("brut"):
        out["brut"] = np.asarray(X, dtype=np.float64)
    if want("puissance"):
        out["puissance"] = band_power(X, sfreq, bands)
    if want("cov"):
        out["cov"] = covariances(X)
    if want("bankcov"):
        bank = bank_covariances(X, sfreq, bands)
        out["bankcov"] = np.stack([bank[b] for b in bands], axis=1)
    if want("augcov"):
        # plongement par retards : reintroduit la dynamique que la covariance
        # spatiale ordinaire ignore. Limite a 3 bandes basses, ou la dynamique
        # lente de la parole imaginee est attendue, pour rester tractable.
        low_bands = tuple(b for b in bands if b[1] <= 13.0)
        bank = bank_augmented_covariances(X, sfreq, low_bands, order=3, lag=8)
        out["augcov"] = np.stack([bank[b] for b in low_bands], axis=1)
    return out


#: nom du modele -> (representation, fabrique du PREFIXE, fabrique du CLASSIFIEUR)
#:
#: Le prefixe regroupe les etages qui n'utilisent PAS les etiquettes (espace
#: tangent, mise a l'echelle). Les separer du classifieur permet de l'ajuster une
#: seule fois par pli et de le reutiliser pour tous les contrastes, qui ne
#: different que par les etiquettes. Sur ``bankcov_tangent``, ou l'ajustement de
#: six moyennes geometriques 64x64 domine le cout, cela divise le temps par
#: presque autant qu'il y a de contrastes.
#:
#: ``None`` en prefixe signifie que le modele est indissociable de ses
#: etiquettes : le CSP apprend ses filtres sur les classes, MDM est directement
#: un classifieur. Ceux-la ne beneficient d'aucune mutualisation.
MODELS: dict[str, tuple[str, callable | None, callable]] = {
    # ligne de base : ce que fait le pipeline actuel
    "puissance_lda": (
        "puissance",
        lambda: StandardScaler(),
        lambda: LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
    ),
    "puissance_logreg": ("puissance", lambda: StandardScaler(), lambda: _logreg(0.1)),

    # spatial classique — filtres appris sur les classes, pas de mutualisation
    "csp_lda": (
        "brut",
        None,
        lambda: Pipeline([
            ("csp", MulticlassCSP(n_components=2)),
            ("scale", StandardScaler()),
            ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ]),
    ),

    # riemannien
    "cov_mdm": ("cov", None, lambda: MDM()),
    "cov_tangent": (
        "cov",
        lambda: Pipeline([("ts", TangentSpace()), ("scale", StandardScaler())]),
        lambda: _logreg(0.1),
    ),
    "bankcov_tangent": (
        "bankcov",
        lambda: Pipeline([("ts", BankTangentSpace()), ("scale", StandardScaler())]),
        lambda: _logreg(0.05),
    ),
    # variantes a LDA a retrecissement : c'est le classifieur du manuscrit
    # (« classified with shrinkage linear discriminant analysis »). En petite
    # taille et grande dimension, le retrecissement de Ledoit-Wolf sur la
    # covariance intra-classe se comporte souvent mieux qu'une penalite L2 sur
    # les coefficients, parce qu'il regularise la quantite reellement mal
    # estimee.
    "cov_tangent_lda": (
        "cov",
        lambda: Pipeline([("ts", TangentSpace()), ("scale", StandardScaler())]),
        lambda: LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
    ),
    "bankcov_tangent_lda": (
        "bankcov",
        lambda: Pipeline([("ts", BankTangentSpace()), ("scale", StandardScaler())]),
        lambda: LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
    ),
    # Filtrage spatial supervise, la configuration du manuscrit. Le prefixe est
    # None et TOUT vit dans l'etage classifieur : xDAWN utilise les etiquettes,
    # donc l'ajuster hors du pli d'entrainement serait une fuite. Le manuscrit
    # precise « in-fold » pour la meme raison.
    "xdawn_tangent_lda": (
        "brut",
        None,
        lambda: Pipeline([
            ("xdawn", Xdawn(n_filters=2, augment=True)),
            ("cov", CovariancesTransformer()),
            ("ts", TangentSpace()),
            ("scale", StandardScaler()),
            ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ]),
    ),
    "augcov_tangent": (
        "augcov",
        lambda: Pipeline([("ts", BankTangentSpace()), ("scale", StandardScaler())]),
        lambda: _logreg(0.02),
    ),
}


def make_transform(name: str):
    """Prefixe independant des etiquettes, ou ``None`` s'il n'y en a pas."""
    factory = MODELS[name][1]
    return factory() if factory is not None else None


def make_classifier(name: str):
    """Etage final, le seul a voir les etiquettes."""
    if name not in MODELS:
        raise ValueError(f"modele inconnu : {name} (disponibles : {sorted(MODELS)})")
    return MODELS[name][2]()


def representation_for(name: str) -> str:
    return MODELS[name][0]
