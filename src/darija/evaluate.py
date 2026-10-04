"""Moteur d'evaluation : un modele, un schema de plis, plusieurs contrastes.

Deux choix structurent ce module.

**La metrique primaire est l'exactitude equilibree, pas l'exactitude brute.**
Avec des classes desequilibrees, l'exactitude brute et sa loi nulle par
permutation ne repondent pas a la meme question. Un classifieur qui s'effondre
sur la classe la plus rare obtient une exactitude tres inferieure au taux de la
classe majoritaire tout en etant significativement au-dessus de sa propre loi
nulle, puisque celle-ci vaut somme_c P(pred=c) P(vrai=c) et s'effondre avec lui.
On obtient alors des lignes du type "0,083 (hasard 0,266) p = 0,010", qui se
lisent comme une aberration alors qu'elles sont arithmetiquement correctes.
L'exactitude equilibree supprime le probleme : son hasard vaut exactement
1/n_classes quel que soit le desequilibre.

**Les contrastes sont evalues ensemble, plis en boucle externe.** Les etages qui
n'utilisent pas les etiquettes (espace tangent, mise a l'echelle) sont ajustes
une fois par pli et reutilises pour tous les contrastes partageant le meme
masque d'essais. Seul le classifieur est reajuste.
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from .contrasts import apply_contrast
from .cv import make_splits
from .pipelines import make_classifier, make_transform, representation_for
from .riemann import recenter_by_group
from .stats import (
    accuracy,
    balanced_accuracy,
    binomial_ci,
    confusion_matrix,
    macro_f1,
    minimum_detectable_balanced_accuracy,
    permutation_pvalue,
)

#: representations sur lesquelles le recentrage riemannien a un sens
COVARIANCE_REPRESENTATIONS = ("cov", "bankcov", "augcov")


@dataclass
class Result:
    model: str
    scheme: str
    contrast: str
    recentered: bool
    n_trials: int
    n_classes: int
    chance_majority: float
    chance_balanced: float
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    ci_low: float
    ci_high: float
    pvalue: float
    null_mean: float
    mde_balanced: float
    unreachable: int = 0
    labels: list[str] = field(default_factory=list)
    confusion: np.ndarray | None = None
    y_true: np.ndarray | None = None
    y_pred: np.ndarray | None = None

    def row(self) -> dict:
        """Version plate, pour un tableau de resultats."""
        return {
            "modele": self.model,
            "schema": self.scheme,
            "contraste": self.contrast,
            "recentre": self.recentered,
            "n": self.n_trials,
            "classes": self.n_classes,
            # metrique primaire
            "equilibree": round(self.balanced_accuracy, 4),
            "hasard_eq": round(self.chance_balanced, 4),
            "seuil_eq": round(self.mde_balanced, 4),
            "p_perm": round(self.pvalue, 4),
            "nulle_moy": round(self.null_mean, 4),
            # descriptif, jamais teste directement
            "brute": round(self.accuracy, 4),
            "hasard_majo": round(self.chance_majority, 4),
            "f1_macro": round(self.macro_f1, 4),
            "ic95_brute": f"[{self.ci_low:.3f}, {self.ci_high:.3f}]",
            "classes_manquantes": self.unreachable,
            "significatif": self.pvalue < 0.05,
        }


def _recenter(representation: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Recentrage par session, en gerant les representations a banc de bandes."""
    if representation.ndim == 4:      # (n, n_bands, c, c)
        return np.stack(
            [recenter_by_group(representation[:, b], groups)
             for b in range(representation.shape[1])],
            axis=1,
        )
    return recenter_by_group(representation, groups)


def _score(model, scheme, contrast, recenter, labels, predictions, predicted,
           groups, chance_majority, unreachable, n_perm, seed) -> Result:
    y_true = labels[predicted]
    y_pred = predictions[predicted].astype(object)
    classes = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
    present = sorted(np.unique(y_true).tolist())
    counts = [int(np.sum(y_true == c)) for c in present]

    pvalue, _, null_mean = permutation_pvalue(
        y_true, y_pred, groups[predicted], n_perm=n_perm, seed=seed
    )
    acc = accuracy(y_true, y_pred)
    low, high = binomial_ci(int(round(acc * len(y_true))), len(y_true))

    return Result(
        model=model,
        scheme=scheme,
        contrast=contrast,
        recentered=recenter,
        n_trials=len(y_true),
        n_classes=len(present),
        chance_majority=chance_majority,
        chance_balanced=1.0 / len(present),
        accuracy=acc,
        balanced_accuracy=balanced_accuracy(y_true, y_pred, present),
        macro_f1=macro_f1(y_true, y_pred, classes),
        ci_low=low,
        ci_high=high,
        pvalue=pvalue,
        null_mean=null_mean,
        mde_balanced=minimum_detectable_balanced_accuracy(counts),
        unreachable=unreachable,
        labels=classes,
        confusion=confusion_matrix(y_true, y_pred, classes),
        y_true=y_true,
        y_pred=y_pred,
    )


def evaluate_many(
    representations: dict[str, np.ndarray],
    words: np.ndarray,
    groups: np.ndarray,
    trial_index: np.ndarray,
    *,
    model: str,
    scheme: str = "loso",
    contrasts: tuple[str, ...] = ("mots15",),
    recenter: bool = False,
    n_splits: int = 5,
    n_perm: int = 5000,
    seed: int = 0,
    sub_block: np.ndarray | None = None,
) -> list[Result]:
    """Evalue un modele sur plusieurs contrastes en mutualisant les plis.

    Les contrastes qui retiennent les memes essais partagent les memes plis et
    le meme prefixe ajuste : seul le classifieur final est reajuste pour chacun.
    """
    representation_name = representation_for(model)
    if recenter and representation_name not in COVARIANCE_REPRESENTATIONS:
        raise ValueError(f"recentrage inapplicable a la representation {representation_name}")

    specs = {name: apply_contrast(name, words) for name in contrasts}

    # les contrastes de meme masque partagent tout le travail independant des
    # etiquettes ; seul oui_non, qui ne retient que deux mots, a un masque propre
    by_mask: dict[bytes, list[str]] = defaultdict(list)
    for name, (_, mask, _) in specs.items():
        by_mask[mask.tobytes()].append(name)

    results: list[Result] = []
    for names in by_mask.values():
        _, mask, _ = specs[names[0]]

        X = representations[representation_name][mask]
        groups_c = np.asarray(groups)[mask]
        trial_index_c = np.asarray(trial_index)[mask]
        if recenter:
            X = _recenter(X, groups_c)

        sub_block_c = None if sub_block is None else np.asarray(sub_block)[mask]
        splits = make_splits(scheme, groups_c, trial_index_c, n_splits=n_splits,
                             sub_block=sub_block_c)

        predictions = {n: np.empty(int(mask.sum()), dtype=object) for n in names}
        predicted = {n: np.zeros(int(mask.sum()), dtype=bool) for n in names}
        unreachable = {n: 0 for n in names}
        degenerate = {n: 0 for n in names}

        for train, test in splits:
            transform = make_transform(model)
            if transform is not None:
                # le prefixe ne voit jamais les etiquettes : il est ajuste une
                # fois par pli et sert a tous les contrastes de ce masque
                transform.fit(X[train])
                X_train, X_test = transform.transform(X[train]), transform.transform(X[test])
            else:
                X_train, X_test = X[train], X[test]

            for name in names:
                labels = specs[name][0]
                train_classes = np.unique(labels[train])
                if len(train_classes) < 2:
                    continue
                # Une LDA exige strictement plus d'echantillons que de classes,
                # et un contraste qui ne retient que quelques essais (oui_non :
                # deux mots) peut produire un tel pli sous un schema fin. Sauter
                # le pli plutot que laisser lever : le score porte alors sur les
                # essais effectivement predits, et le compte est rapporte.
                if len(train) <= len(train_classes):
                    degenerate[name] += len(test)
                    continue
                unreachable[name] += int(np.sum(~np.isin(labels[test], train_classes)))

                estimator = make_classifier(model)
                estimator.fit(X_train, labels[train])
                predictions[name][test] = estimator.predict(X_test)
                predicted[name][test] = True

        for name in names:
            if degenerate[name]:
                warnings.warn(
                    f"{model}/{scheme}/{name} : {degenerate[name]} essais non "
                    f"evalues, leur pli d'entrainement etant trop petit pour le "
                    f"nombre de classes. Le score porte sur les autres.",
                    RuntimeWarning, stacklevel=2,
                )
            if not predicted[name].any():
                # Aucun essai evaluable : tous les plis etaient degeneres ou
                # mono-classe. Scorer cela leve sur une exactitude NaN. Le
                # contraste ne produit donc pas de ligne, ce qui fait qu'une
                # relance le reprendra au lieu de le croire deja fait.
                warnings.warn(
                    f"{model}/{scheme}/{name} : aucun essai evaluable, aucun "
                    f"resultat produit pour ce contraste.",
                    RuntimeWarning, stacklevel=2,
                )
                continue
            if unreachable[name]:
                warnings.warn(
                    f"{model}/{scheme}/{name} : {unreachable[name]} essais de test "
                    f"ont une classe absente de leur pli d'entrainement.",
                    RuntimeWarning, stacklevel=2,
                )
            results.append(_score(
                model, scheme, name, recenter, specs[name][0],
                predictions[name], predicted[name], groups_c,
                specs[name][2], unreachable[name], n_perm, seed,
            ))

    order = {name: i for i, name in enumerate(contrasts)}
    return sorted(results, key=lambda r: order[r.contrast])


def evaluate(representations, words, groups, trial_index, *, model, scheme="loso",
             contrast="mots15", **kwargs) -> Result:
    """Cas a un seul contraste, conserve pour les tests et l'usage interactif."""
    return evaluate_many(representations, words, groups, trial_index, model=model,
                         scheme=scheme, contrasts=(contrast,), **kwargs)[0]
