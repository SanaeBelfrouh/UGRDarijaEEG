"""Inference : permutations groupees, intervalles, puissance statistique.

La question de la puissance est ici plus importante que le choix du test. Avec
1800 essais imagines et un hasard a 6,67 %, la plus petite exactitude qu'on
puisse declarer differente du hasard tourne autour de 8 %. Autrement dit : meme
un effet reel de 1,5 point serait indetectable. Il faut le savoir AVANT
d'interpreter un resultat non significatif comme une absence d'effet.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def accuracy(y_true, y_pred) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def balanced_accuracy(y_true, y_pred, labels=None) -> float:
    """Moyenne des rappels par classe.

    C'est la metrique primaire de ce projet, et non l'exactitude brute. Avec des
    classes desequilibrees, l'exactitude brute et sa loi nulle par permutation ne
    repondent pas a la meme question, ce qui produit des lignes absurdes en
    apparence : un classifieur qui s'effondre sur la classe la plus RARE obtient
    une exactitude tres inferieure au taux de la classe majoritaire tout en etant
    significativement au-dessus de sa propre loi nulle, puisque celle-ci vaut
    somme_c P(pred=c) P(vrai=c) et s'effondre avec lui.

    L'exactitude equilibree n'a pas ce defaut : son niveau de hasard vaut
    exactement 1/n_classes quel que soit le desequilibre, elle est comparable
    d'un contraste a l'autre, et un effondrement sur une classe la ramene a
    1/n_classes au lieu de la faire varier arbitrairement.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(labels) if labels is not None else sorted(np.unique(y_true).tolist())

    recalls = []
    for label in labels:
        mask = y_true == label
        if mask.sum():
            recalls.append(float(np.mean(y_pred[mask] == label)))
    return float(np.mean(recalls)) if recalls else float("nan")


def permutation_pvalue(y_true, y_pred, groups, n_perm: int = 5000, seed: int = 0,
                       statistic=balanced_accuracy) -> tuple[float, np.ndarray, float]:
    """p par permutation des etiquettes A L'INTERIEUR de chaque session.

    Les predictions sont figees ; seules les etiquettes bougent. Permuter
    librement sur les 1800 essais supposerait qu'ils sont independants, alors
    qu'ils proviennent de 8 sessions dont chacune a sa propre derive : la
    distribution nulle serait trop etroite et le p trop optimiste. Permuter dans
    la session preserve la structure de groupe.

    La statistique testee est par defaut l'exactitude equilibree. Renvoie
    (p, distribution nulle, moyenne de la nulle) : la moyenne de la nulle doit
    etre rapportee, c'est elle qui dit par rapport a QUOI le p a ete calcule.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    groups = np.asarray(groups)
    labels = sorted(np.unique(y_true).tolist())
    observed = statistic(y_true, y_pred, labels)

    rng = np.random.default_rng(seed)
    blocks = [np.flatnonzero(groups == g) for g in np.unique(groups)]

    if statistic is balanced_accuracy:
        # Chemin rapide. Une permutation a l'interieur des sessions preserve le
        # multi-ensemble d'etiquettes, donc l'effectif de chaque classe est
        # constant d'une permutation a l'autre : l'exactitude equilibree se
        # ramene a un bincount des bonnes reponses divise par ces effectifs.
        index = {label: i for i, label in enumerate(labels)}
        true_codes = np.array([index[v] for v in y_true], dtype=np.int64)
        pred_codes = np.array([index.get(v, -1) for v in y_pred], dtype=np.int64)
        counts = np.bincount(true_codes, minlength=len(labels)).astype(float)

        permuted = true_codes.copy()
        null = np.empty(n_perm)
        for i in range(n_perm):
            for block in blocks:
                permuted[block] = rng.permutation(true_codes[block])
            hits = np.bincount(permuted[permuted == pred_codes], minlength=len(labels))
            null[i] = np.mean(hits / counts)
    else:
        permuted = y_true.copy()
        null = np.empty(n_perm)
        for i in range(n_perm):
            for block in blocks:
                permuted[block] = rng.permutation(y_true[block])
            null[i] = statistic(permuted, y_pred, labels)

    pvalue = (1.0 + np.sum(null >= observed)) / (n_perm + 1.0)
    return float(pvalue), null, float(null.mean())


def binomial_ci(n_correct: int, n_total: int, alpha: float = 0.05) -> tuple[float, float]:
    """Intervalle de Wilson, preferable a Wald pres du hasard et aux petits n."""
    if n_total == 0:
        return (float("nan"), float("nan"))
    z = stats.norm.ppf(1 - alpha / 2)
    p = n_correct / n_total
    denom = 1 + z**2 / n_total
    centre = (p + z**2 / (2 * n_total)) / denom
    half = z * np.sqrt(p * (1 - p) / n_total + z**2 / (4 * n_total**2)) / denom
    return float(centre - half), float(centre + half)


def minimum_detectable_accuracy(n: int, chance: float, power: float = 0.80,
                                alpha: float = 0.05) -> float:
    """Plus petite exactitude vraie detectable avec la puissance demandee.

    Test binomial unilateral, approximation normale. Sert a repondre a la
    question qui precede toute interpretation d'un resultat nul : cette
    experience avait-elle seulement les moyens de voir l'effet recherche ?
    """
    z_alpha = stats.norm.ppf(1 - alpha)
    z_beta = stats.norm.ppf(power)
    se0 = np.sqrt(chance * (1 - chance) / n)

    # resolution par point fixe : l'ecart-type sous H1 depend de la cible
    p = chance + z_alpha * se0
    for _ in range(100):
        se1 = np.sqrt(p * (1 - p) / n)
        new = chance + (z_alpha * se0 + z_beta * se1)
        if abs(new - p) < 1e-10:
            break
        p = new
    return float(min(p, 1.0))


def minimum_detectable_balanced_accuracy(class_counts, power: float = 0.80,
                                         alpha: float = 0.05) -> float:
    """Seuil de detectabilite de l'exactitude equilibree.

    Sous l'hypothese nulle, le rappel de chaque classe vaut 1/k et sa variance
    (1/k)(1-1/k)/n_c. L'exactitude equilibree etant leur moyenne, sa variance
    vaut (1/k^2) (1/k)(1-1/k) somme_c 1/n_c. Les classes rares pesent donc
    lourd : une classe a 120 essais degrade la sensibilite de tout le contraste,
    ce qui est exactement la raison de preferer des contrastes equilibres.
    """
    counts = np.asarray(list(class_counts), dtype=float)
    counts = counts[counts > 0]
    k = len(counts)
    if k < 2:
        return float("nan")

    chance = 1.0 / k
    se = np.sqrt((1.0 / k**2) * chance * (1 - chance) * np.sum(1.0 / counts))
    z = stats.norm.ppf(1 - alpha) + stats.norm.ppf(power)
    return float(min(chance + z * se, 1.0))


def required_trials(effect: float, chance: float, power: float = 0.80,
                    alpha: float = 0.05) -> int:
    """Nombre d'essais necessaires pour detecter une exactitude vraie ``effect``."""
    if effect <= chance:
        return -1
    z_alpha = stats.norm.ppf(1 - alpha)
    z_beta = stats.norm.ppf(power)
    num = z_alpha * np.sqrt(chance * (1 - chance)) + z_beta * np.sqrt(effect * (1 - effect))
    return int(np.ceil((num / (effect - chance)) ** 2))


def holm_bonferroni(pvalues) -> np.ndarray:
    """Correction de Holm : uniformement plus puissante que Bonferroni simple."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted


def confusion_matrix(y_true, y_pred, labels) -> np.ndarray:
    """Matrice de confusion brute, lignes = verite, colonnes = prediction."""
    labels = list(labels)
    index = {lab: i for i, lab in enumerate(labels)}
    M = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        M[index[t], index[p]] += 1
    return M


def macro_f1(y_true, y_pred, labels) -> float:
    """F1 moyenne par classe : moins trompeuse que l'exactitude si le
    classifieur s'effondre sur quelques classes majoritaires."""
    M = confusion_matrix(y_true, y_pred, labels)
    tp = np.diag(M).astype(float)
    precision = np.divide(tp, M.sum(axis=0), out=np.zeros_like(tp), where=M.sum(axis=0) > 0)
    recall = np.divide(tp, M.sum(axis=1), out=np.zeros_like(tp), where=M.sum(axis=1) > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return float(f1.mean())
