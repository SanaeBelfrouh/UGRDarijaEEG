"""Filtrage, reference et rejet d'essais.

Aucune de ces operations n'utilise les etiquettes, elles peuvent donc etre
appliquees avant le decoupage en plis sans creer de fuite. Les operations qui
en utiliseraient (mise a l'echelle par classe, selection de canaux guidee par la
performance) n'ont pas leur place ici : elles appartiennent a un ``Pipeline``
sklearn ajuste sur le pli d'entrainement.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt

LINE_FREQ = 50.0


def notch(X: np.ndarray, sfreq: float, freq: float = LINE_FREQ, quality: float = 30.0,
          n_harmonics: int = 4) -> np.ndarray:
    """Retire le secteur et ses harmoniques sous la frequence de Nyquist."""
    X = np.asarray(X, dtype=np.float64)
    for k in range(1, n_harmonics + 1):
        f0 = freq * k
        if f0 >= 0.45 * sfreq:
            break
        b, a = iirnotch(f0, quality, sfreq)
        X = filtfilt(b, a, X, axis=-1)
    return X


def bandpass(X: np.ndarray, sfreq: float, low: float, high: float, order: int = 4) -> np.ndarray:
    """Passe-bande Butterworth a phase nulle."""
    nyq = 0.5 * sfreq
    high = min(high, 0.98 * nyq)
    sos = butter(order, [low / nyq, high / nyq], btype="band", output="sos")
    return sosfiltfilt(sos, np.asarray(X, dtype=np.float64), axis=-1)


def common_average_reference(X: np.ndarray) -> np.ndarray:
    """Reference moyenne commune, calculee canal par canal sur chaque essai."""
    X = np.asarray(X, dtype=np.float64)
    return X - X.mean(axis=-2, keepdims=True)


def crop(X: np.ndarray, times: np.ndarray, tmin: float, tmax: float
         ) -> tuple[np.ndarray, np.ndarray]:
    """Restreint les epoques a [tmin, tmax)."""
    mask = (times >= tmin) & (times < tmax)
    return X[..., mask], times[mask]


def reject_trials(X: np.ndarray, *, amplitude_uv: float = 150.0,
                  max_bad_fraction: float = 0.15,
                  mad_threshold: float = 6.0) -> tuple[np.ndarray, dict]:
    """Masque booleen des essais a conserver, plus un rapport de diagnostic.

    Le critere porte sur la PROPORTION de canaux hors bornes, pas sur le pire
    canal. Exiger que les 64 canaux restent sous un seuil revient a exiger
    qu'aucun clignement, aucune electrode bruyante et aucun mouvement n'ait
    touche une seule derivation pendant 2,5 s : sur un montage 64 canaux avec
    fixation visuelle, cela rejette la quasi-totalite des essais. Un essai est
    exploitable si l'immense majorite de ses canaux est propre.

    Le second critere est robuste et relatif a la session : mediane + k * MAD de
    l'amplitude mediane par essai. Il attrape les derives qui restent sous le
    seuil absolu tout en etant aberrantes pour cette session, et il ne depend
    d'aucune hypothese sur les unites du flux.
    """
    X = np.asarray(X, dtype=np.float64)
    ptp = np.ptp(X, axis=-1)                        # (n_trials, n_channels)

    bad_fraction = (ptp > amplitude_uv).mean(axis=1)
    keep = bad_fraction <= max_bad_fraction

    trial_level = np.median(ptp, axis=1)            # amplitude typique de l'essai
    median = np.median(trial_level)
    mad = np.median(np.abs(trial_level - median)) * 1.4826
    if mad > 0:
        keep &= trial_level < median + mad_threshold * mad

    report = {
        "ptp_median_uv": float(median),
        "ptp_p05_uv": float(np.percentile(trial_level, 5)),
        "ptp_p95_uv": float(np.percentile(trial_level, 95)),
        "ptp_worst_channel_median_uv": float(np.median(ptp.max(axis=1))),
        "rejected_by_amplitude": int(np.sum(bad_fraction > max_bad_fraction)),
        "rejected_total": int(np.sum(~keep)),
        "n_trials": int(len(keep)),
    }
    return keep, report


def looks_like_microvolts(ptp_median: float) -> bool:
    """L'amplitude mediane est-elle compatible avec de l'EEG exprime en uV ?

    Apres un passe-bande 0,5-45 Hz, un essai d'EEG de scalp a typiquement une
    amplitude crete-a-crete de quelques dizaines de uV. Une mediane a 1e-5 ou a
    1e6 signale que le flux n'est pas en microvolts, et rend tout seuil absolu
    exprime en uV denue de sens.
    """
    return 1.0 <= ptp_median <= 500.0


def filter_continuous(data: np.ndarray, sfreq: float, *, low: float = 0.5,
                     high: float = 45.0, do_notch: bool = True,
                     chunk: int = 8) -> np.ndarray:
    """Secteur et passe-bande sur l'enregistrement CONTINU, avant decoupage.

    C'est l'ordre correct, et il n'est pas interchangeable avec l'autre. Filtrer
    chaque epoque separement fait apparaitre des transitoires de bord : un
    passe-haut a 0,5 Hz a une constante de temps de l'ordre de 0,3 s, et un
    amplificateur couple en continu comme l'actiCHamp presente des decalages de
    ligne de base de plusieurs millivolts. Sur une epoque de 2,5 s, ces
    transitoires dominent l'amplitude crete-a-crete et font rejeter des essais
    parfaitement propres.

    Le filtrage se fait par paquets de canaux pour borner la memoire : 64 canaux
    x 1,5 million d'echantillons en double precision font 768 Mo.
    """
    data = np.asarray(data)
    out = np.empty_like(data, dtype=np.float32)
    for start in range(0, data.shape[0], chunk):
        block = np.asarray(data[start:start + chunk], dtype=np.float64)
        if do_notch:
            block = notch(block, sfreq)
        block = bandpass(block, sfreq, low, high)
        out[start:start + chunk] = block.astype(np.float32)
    return out


def detrend_epochs(X: np.ndarray) -> np.ndarray:
    """Retire une tendance lineaire par epoque et par canal.

    Garde-fou bon marche contre ce qui subsiste de derive lente apres le
    passe-bande, notamment les rampes de sueur et les derives d'impedance.
    """
    from scipy.signal import detrend

    return detrend(np.asarray(X, dtype=np.float64), axis=-1, type="linear")


def prepare(X: np.ndarray, sfreq: float, *, car: bool = True,
            detrend: bool = True) -> np.ndarray:
    """Finition au niveau de l'epoque : detendance puis reference moyenne.

    Le filtrage n'est deliberement PAS fait ici : il appartient au signal
    continu (voir :func:`filter_continuous`). La reference moyenne commune est
    en revanche une operation point de temps par point de temps, donc identique
    qu'on l'applique avant ou apres le decoupage.
    """
    X = np.asarray(X, dtype=np.float64)
    if detrend:
        X = detrend_epochs(X)
    if car:
        X = common_average_reference(X)
    return np.ascontiguousarray(X, dtype=np.float64)


# --------------------------------------------------------------------------
# retrait des composantes oculaires
# --------------------------------------------------------------------------

#: electrodes frontales du 10-20, sur lesquelles se concentre le poids des
#: composantes oculaires (clignements et saccades)
FRONTAL = ("Fp1", "Fp2", "AF7", "AF3", "AF4", "AF8",
           "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8")


def frontal_weight_share(mixing: np.ndarray, ch_names, frontal=FRONTAL) -> np.ndarray:
    """Part du poids absolu de chaque composante portee par les frontales.

    ``mixing`` est de forme (n_channels, n_components). La part vaut 1 pour une
    composante entierement frontale, et environ n_frontales / n_canaux pour une
    composante uniformement repartie.
    """
    lowered = [str(c).lower() for c in ch_names]
    wanted = {c.lower() for c in frontal}
    idx = np.asarray([i for i, n in enumerate(lowered) if n in wanted], dtype=int)
    if not len(idx):
        raise ValueError("aucune electrode frontale reconnue dans ch_names")

    weight = np.abs(np.asarray(mixing, dtype=np.float64))
    total = weight.sum(axis=0)
    return weight[idx].sum(axis=0) / np.where(total > 0, total, 1.0)


def remove_ocular(X: np.ndarray, ch_names, *, ratio: float = 2.5, max_drop: int = 3,
                  subsample: int = 4, seed: int = 0) -> tuple[np.ndarray, dict]:
    """Projette hors des donnees les composantes independantes a poids frontal.

    Aucune etiquette n'est utilisee : la selection ne regarde que la topographie
    des composantes, exactement comme la procedure du manuscrit (« projecting out
    independent components whose weight mass fell mainly on frontal
    electrodes »). Le manuscrit note que les etiquettes stockees avec les
    matrices ICA sous-comptent les composantes oculaires, parce que la regle de
    labellisation exigeait une dominance basse frequence que les composantes
    frontales melant clignement et activite du frontalis n'atteignent pas ;
    d'ou une selection sur la part de poids frontal plutot que sur un label.

    ``ratio`` est le seuil, exprime en MULTIPLE de ce qu'une composante
    uniformement repartie porterait, soit n_frontales / n_canaux. Un seuil
    absolu serait trompeur : sur un montage ou la moitie des electrodes sont
    frontales, une composante uniforme a deja une part de 0,5, et un seuil a 0,5
    retirerait tout. Exprime en multiple, le meme reglage vaut pour tous les
    montages.

    ``max_drop`` borne le nombre de composantes retirees. Sans cette borne, un
    seuil mal choisi vide le signal : sur 64 canaux, retirer dix composantes
    supprime un sixieme du rang.

    L'ICA est ajustee a RANG PLEIN, autant de composantes que de canaux. Ce
    n'est pas un choix de commodite. Avec moins de composantes que de canaux,
    FastICA projette sur un sous-espace et la decomposition n'est plus
    inversible : la reconstruction complete s'ecarte des donnees (mesure a 16,8
    uV sur un cas de test a 10 composantes pour 20 canaux). Soustraire une
    contribution estimee dans ce sous-espace a des donnees de rang plein est
    alors incoherent, et cela injecte du signal dans les essais depourvus
    d'artefact au lieu d'en retirer.

    Renvoie les donnees nettoyees et un rapport indiquant ce qui a ete retire.
    """
    from sklearn.decomposition import FastICA

    X = np.asarray(X, dtype=np.float64)
    n_trials, n_channels, n_times = X.shape
    n_components = n_channels          # rang plein : voir la note du docstring

    lowered = [str(c).lower() for c in ch_names]
    n_frontal = sum(1 for n in lowered if n in {c.lower() for c in FRONTAL})
    if not n_frontal:
        raise ValueError("aucune electrode frontale reconnue dans ch_names")
    uniform = n_frontal / n_channels
    threshold = min(0.95, ratio * uniform)

    # l'ICA est ajustee sur les echantillons concatenes, sous-echantillonnes en
    # temps : une composante spatiale n'a pas besoin de tous les points pour
    # etre estimee, et cela divise le cout d'autant
    flat = X.transpose(1, 0, 2).reshape(n_channels, -1)[:, ::subsample].T

    ica = FastICA(n_components=n_components, whiten="unit-variance",
                  random_state=seed, max_iter=1000, tol=1e-3)
    ica.fit(flat)

    share = frontal_weight_share(ica.mixing_, ch_names)
    order = np.argsort(share)[::-1]
    drop = [int(i) for i in order if share[i] > threshold][:max_drop]

    report = {
        "n_components": int(n_components),
        "part_uniforme": round(float(uniform), 3),
        "seuil": round(float(threshold), 3),
        "dropped": drop,
        "dropped_share": [round(float(share[i]), 3) for i in drop],
        "max_share_kept": round(float(max([share[i] for i in range(n_components)
                                           if i not in drop], default=0.0)), 3),
    }
    if not drop:
        return X, report

    # Projection par les sources NON CENTREES. A rang plein, la decomposition
    # X = (X @ W.T) @ A.T est exacte sans terme de moyenne separe (verifie a
    # 1e-13), donc retirer une composante retire aussi sa composante continue.
    #
    # Les trois variantes naives echouent, et il vaut la peine de dire pourquoi.
    # Reconstruire depuis un sous-espace (n_components < n_channels) perd les
    # dimensions restantes : 18 uV d'ecart mesures a 10 composantes pour 20
    # canaux. Passer par transform/inverse_transform, ou soustraire
    # sources[:, drop] @ mixing_[:, drop].T, laisse derriere le terme
    # A[c,k] * moyenne(s_k) puisque transform centre les canaux : un clignement
    # unipolaire deposant un decalage continu sur les frontales, cela injecte ce
    # decalage dans les essais depourvus d'artefact au lieu de l'enlever.
    flat_all = X.transpose(1, 0, 2).reshape(n_channels, -1).T       # (echantillons, canaux)
    sources = flat_all @ ica.components_.T                          # non centrees
    ocular = sources[:, drop] @ ica.mixing_[:, drop].T
    cleaned = (flat_all - ocular).T.reshape(n_channels, n_trials, n_times)
    return np.ascontiguousarray(cleaned.transpose(1, 0, 2)), report
