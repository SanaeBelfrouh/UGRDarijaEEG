"""Tests du pipeline : il doit voir un effet planté, et ne pas en inventer."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from darija import cv, riemann, stats                      # noqa: E402
from darija.preprocess import (                            # noqa: E402
    bandpass,
    filter_continuous,
    looks_like_microvolts,
    reject_trials,
)
from darija.evaluate import evaluate, evaluate_many        # noqa: E402
from darija.pipelines import build_representations         # noqa: E402
from synthetic import make_dataset                         # noqa: E402


def test_riemann_identities():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(8, 40))
    C = A @ A.T / 40 + np.eye(8)

    assert np.allclose(riemann.sqrtm(C) @ riemann.sqrtm(C), C, atol=1e-8)
    assert np.allclose(riemann.invsqrtm(C) @ riemann.sqrtm(C), np.eye(8), atol=1e-8)
    assert np.allclose(riemann.expm(riemann.logm(C)), C, atol=1e-8)
    assert riemann.distance_riemann(C, C) < 1e-8

    # la moyenne geometrique d'une matrice repetee est cette matrice
    assert np.allclose(riemann.geometric_mean(np.stack([C] * 5)), C, atol=1e-6)


def test_expm_accepts_indefinite_matrices():
    """Non-regression : ``expm`` ecrasait les valeurs propres negatives.

    Son argument est un vecteur tangent, symetrique mais INDEFINI. Le plafond
    par le bas a 1e-15, indispensable pour la racine et le logarithme qui
    exigent une matrice definie positive, transformait exp(lambda) < 1 en 1 et
    detruisait la moitie de l'information."""
    from scipy.linalg import expm as scipy_expm

    rng = np.random.default_rng(0)
    A = rng.normal(size=(8, 8))
    T = 0.5 * (A + A.T)                       # symetrique, valeurs propres des deux signes
    assert (np.linalg.eigvalsh(T) < 0).any(), "le scenario doit etre indefini"

    assert np.allclose(riemann.expm(T), scipy_expm(T), atol=1e-9)


def test_geometric_mean_actually_converges():
    """Non-regression : la moyenne de Karcher n'atteignait jamais son critere.

    Avec l'exponentielle corrompue, la direction de descente etait constante
    d'une iteration a l'autre : la boucle consommait ses 60 iterations sans
    jamais bouger du point d'initialisation, et renvoyait donc a peu pres la
    moyenne ARITHMETIQUE. Tout l'espace tangent etait construit sur un mauvais
    point de reference, et chaque appel coutait quinze fois son prix."""
    rng = np.random.default_rng(1)
    covs = riemann.covariances(rng.normal(size=(30, 8, 200)))

    mean = riemann.geometric_mean(covs, tol=1e-10, max_iter=60)
    assert riemann.geometric_mean.last_iterations < 10, \
        f"{riemann.geometric_mean.last_iterations} iterations"

    # au point de Karcher, la somme des logarithmes tangents est nulle
    Mm12 = riemann.invsqrtm(mean)
    gradient = riemann.logm(Mm12 @ covs @ Mm12).mean(axis=0)
    assert np.linalg.norm(gradient) < 1e-9, np.linalg.norm(gradient)

    # et il est distinct de la moyenne arithmetique, sinon le test ne prouve rien
    arithmetic = covs.mean(axis=0)
    assert riemann.distance_riemann(mean, arithmetic) > 1e-6


def test_tangent_space_is_isometric_at_reference():
    """La norme du vecteur tangent doit egaler la distance riemannienne a la
    reference : c'est ce qui rend un classifieur lineaire tangent sense."""
    rng = np.random.default_rng(1)
    covs = riemann.covariances(rng.normal(size=(6, 8, 200)))
    ref = riemann.geometric_mean(covs)
    vectors = riemann.tangent_vectors(covs, ref)
    for v, c in zip(vectors, covs):
        assert abs(np.linalg.norm(v) - riemann.distance_riemann(ref, c)) < 1e-6


def test_recentering_makes_session_means_identity():
    rng = np.random.default_rng(2)
    covs = riemann.covariances(rng.normal(size=(40, 8, 200)))
    groups = np.repeat(["a", "b"], 20)
    covs[groups == "b"] *= 7.0            # deplacement de domaine grossier

    out = riemann.recenter_by_group(covs, groups)
    for g in ("a", "b"):
        mean = riemann.geometric_mean(out[groups == g])
        assert np.allclose(mean, np.eye(8), atol=1e-4)


def test_cv_splits_are_disjoint_and_grouped():
    groups = np.repeat(["S1", "S2", "S3"], 20)
    order = np.tile(np.arange(20), 3)

    for train, test in cv.leave_one_session_out(groups):
        assert not set(train) & set(test)
        assert len(np.unique(groups[test])) == 1

    for train, test in cv.within_session_blocked(groups, order, n_splits=4):
        assert not set(train) & set(test)
        # entrainement et test viennent de la meme session
        assert set(groups[train]) == set(groups[test])
        # les blocs sont contigus dans le temps : le test ne chevauche pas
        assert np.ptp(order[test]) < 20

    for train, test in cv.pooled_blocked(groups, order, n_splits=4):
        assert not set(train) & set(test)
        assert len(np.unique(groups[test])) == 3


def test_power_helpers_are_consistent():
    n = stats.required_trials(0.10, 1 / 15)
    mde = stats.minimum_detectable_accuracy(n, 1 / 15)
    assert abs(mde - 0.10) < 0.005, mde
    # plus d'essais, seuil plus bas
    assert stats.minimum_detectable_accuracy(4000, 1 / 15) < \
        stats.minimum_detectable_accuracy(1000, 1 / 15)


def test_no_signal_stays_at_chance():
    """Le controle negatif : sans effet plante, le pipeline ne doit rien trouver.

    Le test porte sur la MOYENNE de plusieurs graines, pas sur une seule : un
    test a alpha = 5 % ressort une fois sur vingt par construction, et faire
    echouer la suite la-dessus serait exiger du pipeline qu'il soit anti-
    conservateur. Ce qui doit tenir, c'est que l'exactitude moyenne reste au
    niveau du hasard."""
    accuracies = []
    for seed in range(5):
        X, y, groups, order, sfreq = make_dataset(
            n_words=4, n_per_word=20, n_channels=8, n_times=200, effect=0.0, seed=seed
        )
        reps = build_representations(X, sfreq, bands=((8.0, 13.0),), include=("cov",))
        result = evaluate(reps, y, groups, order, model="cov_tangent", scheme="loso",
                          contrast="mots15", n_perm=200, seed=0)
        accuracies.append(result.accuracy)

    chance = 0.25
    n = 4 * 20 * 4
    tolerance = 3.0 * np.sqrt(chance * (1 - chance) / n) / np.sqrt(len(accuracies))
    assert abs(np.mean(accuracies) - chance) < tolerance, \
        f"exactitude moyenne {np.mean(accuracies):.4f} loin du hasard {chance}"


def test_planted_signal_is_recovered():
    """Le controle positif : avec un effet, le pipeline doit le voir."""
    X, y, groups, order, sfreq = make_dataset(
        n_words=4, n_per_word=25, n_channels=8, n_times=200, effect=0.9, seed=4
    )
    reps = build_representations(X, sfreq, bands=((8.0, 13.0),), include=("cov",))
    result = evaluate(reps, y, groups, order, model="cov_tangent",
                      scheme="loso", contrast="mots15", n_perm=400, seed=0)
    assert result.balanced_accuracy > result.chance_balanced + 0.10, result.balanced_accuracy
    assert result.pvalue < 0.01, result.pvalue


def test_recentering_rescues_a_session_shift():
    """Un deplacement de domaine peut ramener au hasard un signal bien present ;
    le recentrage doit le recuperer. C'est l'argument central du rapport."""
    X, y, groups, order, sfreq = make_dataset(
        n_words=4, n_per_word=25, n_channels=8, n_times=200,
        effect=0.35, session_shift=1.2, seed=5
    )
    reps = build_representations(X, sfreq, bands=((8.0, 13.0),), include=("cov",))
    plain = evaluate(reps, y, groups, order, model="cov_tangent", scheme="loso",
                     n_perm=300, seed=0)
    centred = evaluate(reps, y, groups, order, model="cov_tangent", scheme="loso",
                       recenter=True, n_perm=300, seed=0)
    gain = centred.accuracy - plain.accuracy
    assert gain > 0.03, f"gain du recentrage trop faible : {gain:.4f}"


def _realistic_64ch(rng, n_trials=200, n_channels=64, n_times=625):
    """EEG de scalp plausible : ~35 uV, quelques electrodes bruyantes, clignements."""
    X = rng.normal(0, 6.0, size=(n_trials, n_channels, n_times))
    X[:, :4] *= 6.0                                    # electrodes a haute impedance
    t = np.arange(n_times)
    for i in np.flatnonzero(rng.random(n_trials) < 0.35):
        peak = np.exp(-((t - rng.integers(100, 500)) ** 2) / (2 * 25.0 ** 2))
        X[i, :6] += peak * rng.uniform(150, 400)       # clignement sur les frontales
    return X


def test_rejection_keeps_usable_64_channel_trials():
    """Non-regression : un critere portant sur le PIRE des 64 canaux rejette tout.

    Sur un montage 64 canaux avec fixation visuelle, quelques derivations
    frontales depassent 200 uV crete-a-crete des qu'il y a un clignement, et une
    ou deux electrodes sont toujours plus bruyantes que les autres. Exiger que
    les 64 restent sous le seuil revient a exiger un enregistrement parfait
    pendant 2,5 s : le taux de rejet atteint 100 % sur des donnees parfaitement
    exploitables. Le critere doit porter sur la PROPORTION de canaux atteints."""
    rng = np.random.default_rng(0)
    X = _realistic_64ch(rng)

    worst_channel = np.ptp(X, axis=-1).max(axis=1)
    assert (worst_channel < 200.0).sum() == 0, "le scenario ne reproduit pas le bug"

    keep, report = reject_trials(X, amplitude_uv=150.0, max_bad_fraction=0.15)
    assert keep.mean() > 0.9, f"{keep.mean():.2%} conserves, {report}"
    assert 5.0 < report["ptp_median_uv"] < 150.0, report


def test_rejection_catches_widespread_artifacts():
    """Le critere doit rester capable de rejeter ce qui est vraiment mauvais."""
    rng = np.random.default_rng(1)
    X = rng.normal(0, 6.0, size=(200, 64, 625))
    bad = rng.choice(200, 20, replace=False)
    X[bad] += rng.normal(0, 90.0, size=(20, 64, 1)) * np.linspace(0, 1, 625)

    keep, _ = reject_trials(X)
    assert not keep[bad].any(), "artefacts generalises non detectes"
    assert keep.sum() == 180, f"faux rejets : {180 - keep.sum()}"


def test_unit_check_flags_non_microvolt_streams():
    assert looks_like_microvolts(35.0)
    assert not looks_like_microvolts(3.5e-5)     # flux en volts
    assert not looks_like_microvolts(4.0e4)      # decalage continu non retire


def test_continuous_filtering_beats_per_epoch():
    """Le filtrage appartient au continu, pas a l'epoque.

    Un amplificateur couple en continu comme l'actiCHamp presente des decalages
    de ligne de base de plusieurs millivolts. Filtrer chaque epoque separement
    laisse des transitoires de bord qui gonflent l'amplitude crete-a-crete."""
    rng = np.random.default_rng(2)
    sfreq, n = 1000.0, 60000
    continuous = rng.normal(0, 6.0, size=(4, n)) + 40000.0     # decalage de 40 mV

    filtered = filter_continuous(continuous, sfreq, low=0.5, high=45.0, do_notch=False)
    starts = range(2000, 20000, 2500)
    first = np.stack([filtered[:, i:i + 2500] for i in starts])

    raw = np.stack([continuous[:, i:i + 2500] for i in starts])
    second = bandpass(raw, sfreq, 0.5, 45.0)

    assert np.median(np.ptp(first, axis=-1)) < np.median(np.ptp(second, axis=-1))


def test_balanced_accuracy_neutralises_class_collapse():
    """Non-regression : l'exactitude brute rendait significatif un effondrement.

    Avec des classes desequilibrees, un classifieur qui repond toujours la classe
    la plus RARE obtient une exactitude tres inferieure au taux de la classe
    majoritaire, tout en etant au-dessus de sa propre loi nulle par permutation
    — celle-ci valant somme_c P(pred=c) P(vrai=c), elle s'effondre avec lui. Le
    banc produisait alors des lignes du type "0,083 (hasard 0,266) p = 0,010".

    L'exactitude equilibree ramene un tel effondrement a exactement 1/k."""
    y_true = np.repeat(["a", "b", "c"], [100, 50, 10])
    collapsed = np.full(len(y_true), "c")

    raw = float(np.mean(collapsed == y_true))
    assert raw < 0.10                                   # tres en dessous de 100/160
    assert abs(stats.balanced_accuracy(y_true, collapsed) - 1 / 3) < 1e-9

    perfect = y_true.copy()
    assert stats.balanced_accuracy(y_true, perfect) == 1.0


def test_balanced_threshold_penalises_rare_classes():
    """Le seuil de detectabilite doit se degrader quand une classe est rare."""
    balanced = stats.minimum_detectable_balanced_accuracy([300] * 6)
    skewed = stats.minimum_detectable_balanced_accuracy([479, 359, 359, 240, 240, 120])
    assert skewed > balanced, (balanced, skewed)
    # a effectif egal, plus de classes => hasard plus bas
    assert stats.minimum_detectable_balanced_accuracy([120] * 15) < balanced


def test_shared_folds_match_independent_evaluation():
    """Mutualiser les plis entre contrastes ne doit rien changer aux resultats.

    Le prefixe independant des etiquettes est desormais ajuste une fois par pli
    et reutilise pour tous les contrastes. Ce test verifie que cette
    optimisation est exactement neutre."""
    X, y, groups, order, sfreq = make_dataset(
        n_words=15, n_per_word=6, n_channels=8, n_times=200, effect=0.5, seed=8
    )
    from darija import WORDS

    y = np.asarray(WORDS, dtype=object)[[int(w[1:]) for w in y]]
    reps = build_representations(X, sfreq, bands=((8.0, 13.0),), include=("cov",))

    together = evaluate_many(reps, y, groups, order, model="cov_tangent",
                             scheme="loso", contrasts=("mots15", "pharyngale"), n_perm=100)
    separate = [
        evaluate(reps, y, groups, order, model="cov_tangent", scheme="loso",
                 contrast=name, n_perm=100)
        for name in ("mots15", "pharyngale")
    ]
    for a, b in zip(together, separate):
        assert a.contrast == b.contrast
        assert np.array_equal(a.y_pred, b.y_pred), a.contrast


def test_rank_reduce_global_full_rank_preserves_riemannian_geometry():
    """Une rotation globale a rang plein doit etre invisible a la metrique.

    La distance affine-invariante est inchangee par congruence : si toutes les
    epoques subissent la MEME rotation orthogonale, les distances entre
    covariances sont identiques. Cela vaut controle de coherence du balayage de
    rang, et confirme que la portee 'global' a rang plein est un no-op."""
    from darija.features import rank_reduce

    rng = np.random.default_rng(0)
    X = rng.normal(size=(12, 10, 300))
    groups = np.repeat(["a", "b"], 6)

    rotated = rank_reduce(X, groups, n_components=10, scope="global")
    assert rotated.shape == X.shape

    before = riemann.covariances(X)
    after = riemann.covariances(rotated)
    for i in range(len(X) - 1):
        d0 = riemann.distance_riemann(before[i], before[i + 1])
        d1 = riemann.distance_riemann(after[i], after[i + 1])
        assert abs(d0 - d1) < 1e-6, (d0, d1)


def test_rank_reduce_keeps_leading_variance():
    """La projection doit retenir les composantes de plus forte variance."""
    from darija.features import rank_reduce

    rng = np.random.default_rng(1)
    X = rng.normal(size=(20, 12, 400))
    X[:, 0] *= 8.0                      # un canal domine largement
    groups = np.zeros(20)

    reduced = rank_reduce(X, groups, n_components=3, scope="global")
    assert reduced.shape == (20, 3, 400)
    # la variance totale conservee doit etre l'essentiel de la variance d'origine
    kept = reduced.var(axis=-1).sum()
    total = X.var(axis=-1).sum()
    assert kept / total > 0.8, kept / total


ACTICAP_64 = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FT9", "FC5", "FC1", "FC2", "FC6",
    "FT10", "T7", "C3", "Cz", "C4", "T8", "TP9", "CP5", "CP1", "CP2", "CP6", "TP10",
    "P7", "P3", "Pz", "P4", "P8", "PO9", "O1", "Oz", "O2", "PO10", "AF7", "AF3",
    "AF4", "AF8", "F5", "F1", "F2", "F6", "FT7", "FC3", "FC4", "FT8", "C5", "C1",
    "C2", "C6", "TP7", "CP3", "CPz", "CP4", "TP8", "P5", "P1", "P2", "P6", "PO7",
    "PO3", "POz", "PO4", "PO8",
]


def test_montages_resolve_on_an_acticap_64():
    """Les montages des controles d'artefact doivent tous se resoudre.

    L'anneau mandibulaire doit faire exactement dix electrodes : c'est le chiffre
    du manuscrit, et la comparabilite du controle en depend."""
    from darija.montages import MONTAGES, missing_from, resolve

    assert len(resolve(ACTICAP_64, "machoire10")) == 10
    assert len(resolve(ACTICAP_64, "complet")) == 64
    assert len(resolve(ACTICAP_64, "peripherique14")) == 14

    jaw = set(resolve(ACTICAP_64, "machoire10"))
    occ = set(resolve(ACTICAP_64, "occipital"))
    rest = set(resolve(ACTICAP_64, "sans_machoire"))
    assert not (jaw | occ) & rest, "sans_machoire doit exclure mâchoire et occiput"
    assert len(jaw | occ | rest) == 64, "la partition doit couvrir tous les canaux"

    # FCz est la reference en ligne : absente des donnees, donc signalee et non
    # fatale — un montage qui la contient doit rester utilisable
    assert missing_from(ACTICAP_64, "central") == ["FCz"]
    assert len(resolve(ACTICAP_64, "central")) == 19

    for name in MONTAGES:
        assert len(resolve(ACTICAP_64, name)) >= 4, name


def test_montage_resolution_is_case_insensitive_and_ordered():
    """Les indices doivent suivre l'ordre du fichier, pas celui de la liste.

    Garder l'ordre du fichier evite qu'une permutation silencieuse des canaux ne
    change les covariances d'un run a l'autre."""
    from darija.montages import resolve

    lowered = [c.lower() for c in ACTICAP_64]
    assert np.array_equal(resolve(ACTICAP_64, "machoire10"),
                          resolve(lowered, "machoire10"))

    idx = resolve(ACTICAP_64, "machoire10")
    assert list(idx) == sorted(idx), "les indices doivent etre croissants"


def test_subblock_grouping_is_looser_than_loso_but_not_random():
    """Le schema « block-grouped » du manuscrit, et ce qu'il autorise.

    Il est plus permissif que le leave-one-session-out — le modele voit d'autres
    sous-blocs de la MEME session a l'entrainement — et plus strict qu'un
    melange aleatoire, puisqu'un sous-bloc entier part d'un coup."""
    sessions = np.repeat(["S1", "S2"], 240)
    sub_block = np.tile(np.repeat(np.arange(8), 30), 2)
    order = np.tile(np.arange(240), 2)

    folds = list(cv.subblock_grouped(sessions, order, n_splits=8, sub_block=sub_block))
    assert len(folds) == 8

    for train, test in folds:
        assert not set(train) & set(test)
        # un sous-bloc part en entier : aucun de ses essais ne reste a l'entrainement
        held = {(sessions[i], sub_block[i]) for i in test}
        assert not held & {(sessions[i], sub_block[i]) for i in train}
        # mais la session du test est presente a l'entrainement, contrairement au loso
        assert set(sessions[test]) <= set(sessions[train])

    covered = np.zeros(len(sessions), dtype=bool)
    for _, test in folds:
        covered[test] = True
    assert covered.all(), "chaque essai doit etre teste une fois"


def test_ocular_removal_leaves_untouched_channels_exact():
    """Le retrait oculaire ne doit toucher que ce qu'il vise.

    Trois proprietes, et les trois ont ete cassees par une implementation naive :
    l'artefact part, les canaux qui ne le portent pas restent EXACTEMENT
    identiques, et rien n'est retire quand il n'y a pas d'artefact."""
    from darija.preprocess import bandpass, remove_ocular

    names = list(ACTICAP_64)
    frontal = [names.index(c) for c in ("Fp1", "Fp2", "AF7", "AF3", "AF4", "AF8")]
    posterior = [names.index(c) for c in ("O1", "Oz", "O2", "PO3", "PO4")]

    rng = np.random.default_rng(0)
    X = rng.normal(0, 5.0, size=(60, len(names), 500))
    topography = np.zeros(len(names))
    topography[frontal] = [9, 9, 6, 5, 5, 6]
    t = np.arange(500)
    for i in range(0, 60, 2):
        bump = np.exp(-((t - 250) ** 2) / (2 * 30.0 ** 2)) * rng.uniform(60, 110)
        X[i] += np.outer(topography, bump)
    X = bandpass(X, 250.0, 0.5, 45.0)

    cleaned, report = remove_ocular(X, names, ratio=2.5)
    assert len(report["dropped"]) == 1, report
    assert report["dropped_share"][0] > 0.9, report

    before = np.abs(X[0::2][:, frontal]).mean()
    after = np.abs(cleaned[0::2][:, frontal]).mean()
    assert after < 0.2 * before, f"artefact mal retire : {before:.2f} -> {after:.2f}"

    # Les posterieures ne sont PAS inchangees a la precision machine, et il ne
    # faut pas l'exiger : la composante retiree a une part frontale de 0,996,
    # donc 0,4 % de son poids se repartit sur les 58 autres canaux. Ce qui doit
    # tenir est que le retrait soit spatialement cible.
    def rms(a):
        return float(np.sqrt((a ** 2).mean()))

    removed_frontal = rms(X[:, frontal] - cleaned[:, frontal])
    removed_posterior = rms(X[:, posterior] - cleaned[:, posterior])
    assert removed_posterior < 0.01 * removed_frontal, \
        f"retrait mal cible : {removed_posterior:.4f} contre {removed_frontal:.4f} uV RMS"
    assert abs(rms(cleaned[:, posterior]) / rms(X[:, posterior]) - 1) < 0.005

    # sans artefact, rien ne doit bouger du tout
    Y = bandpass(rng.normal(0, 5.0, size=(40, len(names), 400)), 250.0, 0.5, 45.0)
    clean_y, report_y = remove_ocular(Y, names, ratio=2.5)
    assert report_y["dropped"] == [], report_y
    assert np.array_equal(clean_y, Y)


def test_shrinkage_lda_models_are_available():
    """Les variantes a LDA a retrecissement, classifieur du manuscrit."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    from darija.pipelines import make_classifier, make_transform

    for name in ("cov_tangent_lda", "bankcov_tangent_lda"):
        assert isinstance(make_classifier(name), LinearDiscriminantAnalysis)
        assert make_transform(name) is not None


def test_subset_of_a_referenced_montage_inherits_the_global_mean():
    """Non-regression : pourquoi un sous-montage doit etre re-reference.

    La reference moyenne commune soustrait la moyenne des 64 canaux a chacun.
    Extraire ensuite huit electrodes occipitales laisse -moyenne(64) dans chacune,
    c'est-a-dire une projection du signal GLOBAL — EMG mandibulaire compris. Le
    sous-montage herite alors d'information qu'il ne capte pas lui-meme, et la
    comparaison entre sous-montages ne compare plus rien.

    Ce test plante un signal UNIQUEMENT sur les electrodes mandibulaires et
    verifie qu'il apparait dans les occipitales apres reference globale, puis
    qu'il disparait apres re-reference dans le sous-ensemble."""
    from darija.montages import resolve
    from darija.preprocess import common_average_reference

    names = list(ACTICAP_64)
    jaw = resolve(names, "machoire10")
    occ = resolve(names, "occipital")

    rng = np.random.default_rng(0)
    X = rng.normal(0, 1.0, size=(40, len(names), 300))
    planted = rng.normal(0, 30.0, size=(40, 1, 300))     # fort, mandibulaire seul
    X[:, jaw] += planted

    referenced = common_average_reference(X)

    # sans re-reference : le signal plante fuit dans les occipitales
    leaked = referenced[:, occ]
    correlation = np.corrcoef(leaked.reshape(-1), np.repeat(
        planted, len(occ), axis=1).reshape(-1))[0, 1]
    assert abs(correlation) > 0.5, f"le scenario ne reproduit pas la fuite ({correlation:.3f})"

    # avec re-reference dans le sous-ensemble : la fuite disparait
    fixed = common_average_reference(referenced[:, occ])
    correlation_fixed = np.corrcoef(fixed.reshape(-1), np.repeat(
        planted, len(occ), axis=1).reshape(-1))[0, 1]
    assert abs(correlation_fixed) < 0.05, \
        f"la re-reference ne supprime pas la fuite ({correlation_fixed:.3f})"


def test_xdawn_prototypes_are_what_make_it_work():
    """xDAWN sans augmentation par prototypes ne sert quasiment a rien.

    Le filtrage seul ne fait que reduire la dimension. Ce qui donne sa puissance
    a xDAWN, c'est la covariance du signal AUGMENTE des prototypes de classe :
    ses termes croises portent la correlation entre l'essai et chaque prototype.
    Mesure sur un effet evoque faible plante : 0,25 sans, 0,27 avec la projection
    seule, 0,53 avec les prototypes."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import GroupKFold, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from darija.features import CovariancesTransformer, Xdawn

    rng = np.random.default_rng(0)
    n_classes, n_channels, n_times = 5, 16, 200
    patterns = rng.normal(size=(n_classes, n_channels))
    shape = np.exp(-((np.arange(n_times) - 100) ** 2) / (2 * 20.0 ** 2))

    X, y, groups = [], [], []
    for session in range(4):
        for cls in range(n_classes):
            for _ in range(40):
                noise = np.cumsum(rng.normal(0, 1.0, size=(n_channels, n_times)),
                                  axis=-1) / np.sqrt(n_times)
                X.append(noise + 0.12 * np.outer(patterns[cls], shape))
                y.append(cls)
                groups.append(session)
    X, y, groups = np.stack(X), np.asarray(y), np.asarray(groups)

    def tail():
        return [("cov", CovariancesTransformer()), ("ts", riemann.TangentSpace()),
                ("sc", StandardScaler()),
                ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))]

    def score(steps):
        return cross_val_score(Pipeline(steps), X, y, groups=groups,
                               cv=GroupKFold(n_splits=4)).mean()

    plain = score(tail())
    augmented = score([("xd", Xdawn(2, augment=True))] + tail())
    assert augmented > plain + 0.15, f"{plain:.4f} -> {augmented:.4f}"

    # et la dimension doit rester tractable : un prototype par bloc de filtres,
    # pas chaque evoquee passee par tous les filtres
    fitted = Xdawn(2, augment=True).fit(X, y)
    assert fitted.prototypes_.shape[0] == n_classes * 2, fitted.prototypes_.shape
    assert fitted.transform(X[:2]).shape[1] == 2 * n_classes * 2


def test_xdawn_model_cannot_be_fitted_outside_the_fold():
    """Le modele xDAWN doit avoir un prefixe None.

    xDAWN utilise les etiquettes. S'il vivait dans le prefixe independant des
    etiquettes, il serait ajuste une fois par pli et reutilise pour tous les
    contrastes — donc ajuste sur des etiquettes du pli de test. Le prefixe None
    force tout l'enchainement dans l'etage classifieur."""
    from darija.pipelines import make_classifier, make_transform, representation_for

    assert make_transform("xdawn_tangent_lda") is None
    assert representation_for("xdawn_tangent_lda") == "brut"
    assert [name for name, _ in make_classifier("xdawn_tangent_lda").steps][0] == "xdawn"


def test_trial_reconstruction_from_markers():
    """Reconstruction des essais a partir du seul flux de marqueurs.

    Reproduit la sequence exacte du protocole, avec l'ISI tire dans 0.8-1.2 s,
    et verifie que l'intervalle mesure comme 'isi' retrouve bien ce tirage tandis
    que 'production' reste exact. C'est la verification qui corrige le verdict du
    carnet de timing."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "timing_check",
        Path(__file__).resolve().parents[1] / "scripts" / "00_timing_check.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    rng = np.random.default_rng(0)
    events, t = [], 100.0
    words = ["3afak", "lla", "Ah"]
    for i in range(60):
        word = words[i % len(words)]
        fixation = rng.uniform(1.0, 1.5)
        isi = rng.uniform(0.8, 1.2)
        for stage, duration in (
            ("fixation_start", 0.0), ("fixation_end", fixation),
            ("cue_start", 0.0), ("cue_end", 1.0),
            ("isi_start", 0.0), ("isi_end", isi),
            ("start", 0.0), ("end", 2.0),
            ("iti_start", 0.0), ("iti_end", 0.5),
        ):
            t += duration
            events.append(dict(cond="imagined", stage=stage,
                               word=word if "cue" in stage or stage in ("start", "end") else None,
                               t=t))

    trials = module.reconstruct(events, "S001")
    assert len(trials) == 60, len(trials)
    assert all(tr["word"] == w for tr, w in zip(trials, [words[i % 3] for i in range(60)]))

    isi_measured = np.array([tr["start"] - tr["cue_end"] for tr in trials])
    production = np.array([tr["end"] - tr["start"] for tr in trials])

    # l'ISI reproduit le tirage uniforme : ~115 ms d'ecart-type
    assert 0.8 <= isi_measured.min() and isi_measured.max() <= 1.2
    assert abs(isi_measured.std() - 0.4 / np.sqrt(12)) < 0.03, isi_measured.std()
    # la production est exacte : c'est elle qui aligne les epoques
    assert production.std() < 1e-9, production.std()



def test_degenerate_fold_is_skipped_not_raised():
    """Un pli d'entrainement plus petit que le nombre de classes est saute.

    Le cas se produit reellement : oui_non ne retient que deux mots, et un
    schema fin sur peu d'essais peut produire un pli ou la LDA ne peut pas etre
    ajustee. Avant, la grille entiere levait. Le score doit desormais porter sur
    les essais effectivement predits, et le compte des autres etre rapporte.
    """
    import warnings as _w

    from darija.evaluate import evaluate_many
    from darija.pipelines import build_representations

    # deux mots, deux essais par mot et par session : sous within_session les
    # plis contiennent un ou deux essais, donc au plus autant que de classes
    X, y, groups, trial_index, sfreq = make_dataset(
        n_sessions=3, n_words=2, n_per_word=2, n_channels=8, n_times=128,
        sfreq=128.0, effect=0.4, seed=0,
    )
    reps = build_representations(X, sfreq, include=("cov",))
    with _w.catch_warnings(record=True) as captured:
        _w.simplefilter("always")
        results = evaluate_many(
            reps, y, groups, trial_index, model="cov_tangent_lda",
            scheme="within_session", contrasts=("mots15",), n_splits=2, n_perm=50,
        )
    # ici TOUS les plis sont degeneres : le contraste ne doit produire aucune
    # ligne, et surtout aucune exception. Avant, la grille entiere levait.
    assert results == [], (
        f"aucun essai n'etait evaluable, donc aucun resultat ne doit sortir, "
        f"vu {len(results)}"
    )
    messages = [str(c.message) for c in captured]
    assert any("trop petit" in m for m in messages), (
        f"le saut d'un pli degenere doit etre signale, pas silencieux : {messages}"
    )
    assert any("aucun essai evaluable" in m for m in messages), (
        f"un contraste sans essai evaluable doit etre signale : {messages}"
    )


def test_one_failing_block_does_not_kill_the_grid():
    """02_benchmark.py doit survivre a un bloc qui leve.

    Sur une nuit de calcul, un seul couple (modele, schema) qui leve emportait
    tous les blocs suivants. Le script doit signaler le bloc, n'ecrire aucune
    ligne pour lui — pour qu'une relance le reprenne — et continuer.
    """
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts" / "02_benchmark.py").read_text(encoding="utf-8")
    assert "failed.append" in source, "aucune capture par bloc dans 02_benchmark.py"
    body = source[source.index("        try:"):source.index("        elapsed =")]
    assert "evaluate_many" in body, "l'appel evalue n'est pas sous try"
    assert "continue" in body, "un bloc en echec doit passer au suivant"
    assert "rows.append" not in body, (
        "un bloc en echec ne doit rien ecrire, sinon une relance le sauterait"
    )


def test_resume_recomputes_a_block_whose_predictions_are_missing():
    """Les predictions par essai font partie de la sortie, pas un extra.

    Scenario reel : un premier run sans --save-predictions ecrit le CSV. On
    relance en ajoutant --save-predictions et un modele. La reprise sautait les
    blocs deja dans le CSV, le magasin ne recevait que le nouveau modele, et la
    comparaison appariee repondait « aucune paire comparable » sans dire
    pourquoi — alors que les deux modeles etaient bien dans le tableau.
    """
    import subprocess
    import tempfile

    root = Path(__file__).resolve().parents[1]
    X, y, groups, trial_index, sfreq = make_dataset(
        n_sessions=3, n_words=3, n_per_word=6, n_channels=8, n_times=128,
        sfreq=128.0, effect=0.3, seed=1,
    )
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        archive = tmp / "epochs.npz"
        np.savez_compressed(
            archive, X=X.astype(np.float32),
            times=np.arange(X.shape[-1]) / sfreq, sfreq=sfreq,
            y=y, cond=np.full(len(X), "imagined", dtype=object),
            session=groups, trial_index=trial_index,
            sub_block=(trial_index // 6).astype(int),
            ch_names=np.asarray([f"C{i}" for i in range(8)], dtype=object),
        )
        out = tmp / "b.csv"
        base = [sys.executable, str(root / "scripts" / "02_benchmark.py"),
                "--epochs", str(archive), "--condition", "imagined",
                "--schemes", "loso", "--contrasts", "mots15",
                "--n-components", "8", "--n-perm", "50", "--out", str(out)]

        # premier run : un modele, SANS predictions
        first = subprocess.run(base + ["--models", "cov_tangent"],
                               capture_output=True, text=True, cwd=root)
        assert first.returncode == 0, first.stderr[-600:]
        assert not out.with_suffix(".predictions.npz").exists()

        # second run : on ajoute un modele ET --save-predictions
        second = subprocess.run(
            base + ["--models", "cov_tangent", "cov_tangent_lda", "--save-predictions"],
            capture_output=True, text=True, cwd=root,
        )
        assert second.returncode == 0, second.stderr[-600:]
        assert "deja fait, saute" not in second.stdout, (
            "cov_tangent a ete saute alors que ses predictions manquaient :\n"
            + second.stdout[-800:]
        )

        with np.load(out.with_suffix(".predictions.npz"), allow_pickle=True) as store:
            modeles = {k.split("|")[0] for k in store.files}
        assert modeles == {"cov_tangent", "cov_tangent_lda"}, (
            f"le magasin ne contient que {sorted(modeles)} : le test apparie "
            f"n'aurait rien a comparer"
        )

        # et le CSV ne doit pas avoir double la ligne recalculee
        import pandas as pd
        table = pd.read_csv(out)
        cle = ["modele", "schema", "contraste", "recentre"]
        assert not table.duplicated(cle).any(), (
            f"lignes dupliquees apres recalcul :\n{table[cle]}"
        )


def test_recovery_ignores_other_datasets_files():
    """La recherche est recursive sur /kaggle/input, qui porte TOUS les datasets.

    Sans filtre, un submission.csv ou un train.csv venu d'un autre dataset
    attache au notebook serait copie dans results/. Et s'il portait par hasard
    le nom d'une de nos sorties, il serait repris comme un resultat deja
    calcule — donc une etape serait sautee sur la foi d'un fichier etranger.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "run_all", Path(__file__).resolve().parents[1] / "scripts" / "run_all.py")
    run_all = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_all)

    for name in ("imagine_mots15.csv", "imagine_mots15.predictions.npz",
                 "emg_montages.confusions.npz", "rsa_imagined.csv",
                 "imagine.csv"):          # ancien nom, doit rester reutilisable
        assert run_all.is_our_output(name), f"{name} devrait etre recupere"
    for name in ("submission.csv", "train.csv", "sample_data.npz",
                 "epochs.npz", "epochs_ica.npz"):
        assert not run_all.is_our_output(name), (
            f"{name} ne doit pas etre copie dans results/"
        )


def test_flattened_kaggle_download_is_recovered():
    """`kaggle kernels output -p /kaggle/working` peut aplatir l'arborescence.

    Les CSV du run precedent arrivent alors a la racine du dossier de travail au
    lieu de results/, et la reprise ne les voyait pas : elle recalculait tout en
    silence. Les archives, elles, doivent RESTER a la racine.
    """
    import tempfile

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "run_all", Path(__file__).resolve().parents[1] / "scripts" / "run_all.py")
    run_all = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_all)

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "working"
        work.mkdir()
        (work / "imagine_mots15.csv").write_text("modele\n")
        (work / "imagine_mots15.predictions.npz").write_bytes(b"")
        (work / "epochs.npz").write_bytes(b"")          # doit rester en place

        r = run_all.Runner(work, Path(tmp), max_hours=1.0)
        run_all.flatten_into_results(r)

        assert (r.results / "imagine_mots15.csv").exists(), (
            "le CSV aplati n'a pas ete rapatrie dans results/"
        )
        assert (r.results / "imagine_mots15.predictions.npz").exists(), (
            "les predictions aplaties n'ont pas ete rapatriees"
        )
        assert (work / "epochs.npz").exists(), (
            "l'archive a ete deplacee : run_all la cherche a la racine du "
            "dossier de travail"
        )


def test_orthographic_rdm_is_a_separable_control():
    """Les controles orthographiques doivent etre mesurables ET separables.

    Un indice visuel engage l'orthographe quel que soit l'alphabet : c'est
    l'objection qu'un relecteur posera sur un indice ecrit, arabizi ou non. Pour
    y repondre par une mesure et non par un argument, deux RDM sont derivees de
    la seule chaine affichee — distance d'edition et difference de longueur.

    Encore faut-il qu'elles soient separables des modeles linguistiques : si
    elles leur etaient fortement correlees, un resultat phonologique et un
    resultat orthographique seraient indiscernables, et le controle ne
    controlerait rien.
    """
    from scipy import stats

    from darija import WORDS
    from darija.rsa import ORTHOGRAPHIC_KEYS, model_rdm

    labels = list(WORDS)
    idx = np.tril_indices(len(labels), -1)

    for key in ORTHOGRAPHIC_KEYS:
        rdm = model_rdm(labels, key)
        assert rdm.shape == (len(labels), len(labels))
        assert np.allclose(rdm, rdm.T), f"{key} n'est pas symetrique"
        assert np.allclose(np.diag(rdm), 0), f"{key} a une diagonale non nulle"
        assert rdm[idx].std() > 0, f"{key} est constante : elle ne distingue rien"

    # la distance d'edition doit se lire sur les chaines, pas sur une annotation
    assert model_rdm(["kul", "kul"], "orthographic")[0, 1] == 0.0
    assert model_rdm(["lla", "lma"], "orthographic")[0, 1] > 0.0
    assert model_rdm(["Ah", "shukran"], "cue_length")[0, 1] == 5.0

    for key in ORTHOGRAPHIC_KEYS:
        ortho = model_rdm(labels, key)[idx]
        for autre in ("onset", "pharyngeal", "semantic"):
            rho = stats.spearmanr(ortho, model_rdm(labels, autre)[idx]).statistic
            assert abs(rho) < 0.6, (
                f"{key} correle a {autre} a rho = {rho:+.3f} : un resultat "
                f"orthographique et un resultat linguistique seraient "
                f"indiscernables, donc le controle ne controle rien"
            )


def test_rsa_script_runs_the_orthographic_controls():
    """Les controles doivent etre dans la liste du script, pas seulement codes."""
    from darija.rsa import ORTHOGRAPHIC_KEYS

    source = (Path(__file__).resolve().parents[1] / "scripts" / "03_rsa.py"
              ).read_text(encoding="utf-8")
    for key in ORTHOGRAPHIC_KEYS:
        assert f'"{key}"' in source, (
            f"{key} n'est pas dans MODEL_KEYS : il ne serait jamais teste"
        )


def test_figure_names_match_the_manuscript():
    """Les noms de figures produites doivent etre ceux que le .tex inclut.

    Le lien entre scripts/06_figures.py et manuscript/tex/darija_eeg.tex est un
    nom de fichier, donc rien ne signale une divergence : LaTeX se contente de
    laisser un cadre vide. Ce test transforme l'oubli silencieux en echec.

    Il verifie aussi l'absence de souligne dans ces noms : un souligne dans
    l'argument de \\includegraphics casse la compilation sur les distributions
    qui n'embarquent pas grffile.
    """
    import re

    root = Path(__file__).resolve().parents[1]
    figures = (root / "scripts" / "06_figures.py").read_text(encoding="utf-8")
    tex = root / "manuscript" / "tex" / "darija_eeg.tex"
    if not tex.exists():
        return                                  # le manuscrit est optionnel

    produites = set(re.findall(r'out / "(fig[^"]+)\.pdf"', figures))
    incluses = set(re.findall(r'\\includegraphics\[[^\]]*\]\{([^}]*)\}',
                            tex.read_text(encoding="utf-8")))
    assert produites, "aucune figure produite par 06_figures.py"
    assert produites == incluses, (
        f"le script produit {sorted(produites)}\n"
        f"le manuscrit inclut {sorted(incluses)}\n"
        f"produites et non incluses : {sorted(produites - incluses)}\n"
        f"incluses et non produites : {sorted(incluses - produites)}"
    )
    for nom in incluses:
        assert "_" not in nom, (
            f"'{nom}' contient un souligne : \\includegraphics le refuse sur "
            f"les distributions sans grffile"
        )


def test_band_power_layout_is_band_window_channel():
    """Le canal est l'indice le plus RAPIDE dans la sortie de band_power.

    scripts/07_topomaps.py isole les traits d'une electrode par P[:, i::n_canaux].
    Si le rangement devenait (canal, bande, fenetre), cette tranche melangerait
    des canaux differents et la carte topographique serait fausse sans qu'aucune
    erreur ne soit levee : elle resterait une carte plausible. Ce test fixe le
    rangement pour que le changement casse ici.
    """
    from darija.features import MANUSCRIPT_BANK, band_power

    rng = np.random.default_rng(0)
    n_trials, n_canaux, n_times = 6, 8, 250
    X = rng.normal(size=(n_trials, n_canaux, n_times))
    P = band_power(X, 250.0, MANUSCRIPT_BANK)

    n_fenetres = 4
    assert P.shape == (n_trials, len(MANUSCRIPT_BANK) * n_fenetres * n_canaux)

    # on amplifie UN canal et on regarde quels traits bougent
    for canal in (0, 3, 7):
        Y = X.copy()
        Y[:, canal] *= 50.0
        Q = band_power(Y, 250.0, MANUSCRIPT_BANK)
        bouge = np.flatnonzero(np.abs(Q - P).mean(axis=0) > 1e-6)
        assert set(bouge % n_canaux) == {canal}, (
            f"amplifier le canal {canal} modifie des traits d'autres canaux : "
            f"le rangement n'est pas (bande, fenetre, canal)"
        )
        assert len(bouge) == len(MANUSCRIPT_BANK) * n_fenetres, (
            f"{len(bouge)} traits touches, "
            f"{len(MANUSCRIPT_BANK) * n_fenetres} attendus pour un canal"
        )


def test_electrode_positions_are_a_sane_head():
    """Les positions figees doivent former une tete plausible.

    Elles ont ete extraites une fois de MNE puis figees : rien ne les revalide
    au fil du temps, et une erreur de projection donnerait une carte d'allure
    normale mais fausse.
    """
    from darija.topo import POSITIONS, RAYON_TETE, resolve_positions

    assert len(POSITIONS) >= 60

    xy, garde, absents = resolve_positions(["Fp1", "Fz", "Cz", "Oz", "T7", "T8",
                                            "FT9", "FT10", "O1", "O2",
                                            "PASUNEELECTRODE"])
    assert absents == ["PASUNEELECTRODE"]
    p = dict(zip(["Fp1", "Fz", "Cz", "Oz", "T7", "T8", "FT9", "FT10", "O1", "O2"],
                 xy))

    assert np.linalg.norm(p["Cz"]) < 0.08, "Cz doit tomber au centre"
    assert p["Fp1"][1] > 0.8, "Fp1 doit etre a l'avant"
    assert p["Oz"][1] < -0.8, "Oz doit etre a l'arriere"
    assert p["T7"][0] < -0.9 and p["T8"][0] > 0.9, "T7 a gauche, T8 a droite"
    assert abs(p["T7"][0] + p["T8"][0]) < 0.05, "T7 et T8 doivent etre symetriques"
    assert abs(p["O1"][0] + p["O2"][0]) < 0.05, "O1 et O2 doivent etre symetriques"
    for nom in ("T7", "T8"):
        assert abs(np.linalg.norm(p[nom]) - RAYON_TETE) < 0.08, (
            f"{nom} est sur l'equateur, donc au rayon du cercle de tete"
        )
    # les mandibulaires sont SOUS la ligne des oreilles, donc hors du cercle
    for nom in ("FT9", "FT10"):
        assert np.linalg.norm(p[nom]) > RAYON_TETE, (
            f"{nom} devrait tomber hors du cercle de tete"
        )

if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_"):
            continue
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL  {name}: {exc}")
    print("\nechecs :", failures)
    raise SystemExit(1 if failures else 0)
