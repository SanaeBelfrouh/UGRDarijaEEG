"""Validation technique du corpus, independante de tout decodage.

Quatre controles, chacun calcule sur une session XDF a la fois, a partir des
flux tels que ``pyxdf.load_xdf`` les renvoie :

1. **Qualite du signal.** Densite spectrale de puissance sur l'enregistrement
   continu : frequence du pic alpha, rapport du pic secteur a son voisinage, et
   canaux dont la puissance 1-40 Hz s'ecarte de la mediane de plus de trois
   ecarts absolus medians.
2. **Potentiel evoque visuel.** Epoques ancrees sur l'apparition du mot
   (``{cond}_cue_start``). Un potentiel evoque occipital net, a la meme latence
   d'une session a l'autre, valide a la fois l'EEG et l'alignement des
   marqueurs, sans rien demander au classifieur.
3. **Audio des essais imagines.** Niveau du microphone pendant la fenetre de
   production, rapporte au niveau du meme essai pendant la fixation. Un essai
   imagine dont le niveau monte comme celui d'un essai articule trahit une
   vocalisation.
4. **Debut de la parole articulee.** Premiere trame de 10 ms qui depasse
   le niveau de fixation d'un seuil donne et s'y maintient : donne la latence de
   la parole apres le go, donc l'instant ou l'EMG peut apparaitre.

Les fonctions ne lisent aucun fichier : elles prennent les dictionnaires de flux,
ce qui permet de les tester sur des flux synthetiques.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from .io_xdf import CONDITIONS, EEG_PREFIX, MARKER_STREAM

#: electrodes occipitales et parieto-occipitales pour le potentiel evoque visuel
OCCIPITAL = ("O1", "Oz", "O2", "PO3", "POz", "PO4", "PO7", "PO8")
LINE_FREQ = 50.0


# --------------------------------------------------------------------- flux
def find_streams(streams) -> tuple[dict, dict, dict | None]:
    """Repere les flux EEG, marqueurs et audio parmi ceux d'un fichier XDF."""
    eeg = markers = audio = None
    for s in streams:
        info = s["info"]
        name = str(info["name"][0])
        kind = str(info.get("type", [""])[0]).lower()
        n_ch = int(info["channel_count"][0])
        if name.startswith(EEG_PREFIX) and n_ch > 1:
            eeg = s
        elif name == MARKER_STREAM:
            markers = s
        elif kind == "audio" or "audio" in name.lower():
            audio = s
    if eeg is None or markers is None:
        raise RuntimeError("flux EEG ou marqueurs introuvable")
    return eeg, markers, audio


def channel_names(stream) -> list[str]:
    try:
        chans = stream["info"]["desc"][0]["channels"][0]["channel"]
        return [str(c["label"][0]) for c in chans]
    except Exception:
        return [f"ch{i:02d}" for i in range(int(stream["info"]["channel_count"][0]))]


@dataclass
class Trial:
    cond: str
    word: str
    fixation: float          # debut de la fixation (s, horloge LSL)
    cue: float               # apparition du mot
    go: float                # debut de la fenetre de production
    end: float               # fin de la fenetre de production


def trials_from_markers(markers) -> list[Trial]:
    """Reconstruit les essais complets a partir du flux de marqueurs."""
    vals = [str(v[0] if isinstance(v, (list, np.ndarray)) else v)
            for v in markers["time_series"]]
    stamps = np.asarray(markers["time_stamps"], float)
    trials, cur = [], {}
    for v, t in zip(vals, stamps):
        stage, _, word = v.partition(":")
        for cond in CONDITIONS:
            if not stage.startswith(cond + "_"):
                continue
            phase = stage[len(cond) + 1:]
            if phase == "fixation_start":
                cur = dict(cond=cond, fixation=t)
            elif phase == "cue_start" and cur.get("cond") == cond:
                cur.update(cue=t, word=word)
            elif phase == "start" and "cue" in cur:
                cur["go"] = t
            elif phase == "end" and "go" in cur:
                cur["end"] = t
                trials.append(Trial(cond=cond, word=cur["word"], fixation=cur["fixation"],
                                    cue=cur["cue"], go=cur["go"], end=t))
                cur = {}
            break
    return trials


# ------------------------------------------------------------ 1. spectre
@dataclass
class SpectrumReport:
    freqs: np.ndarray
    median_psd: np.ndarray           # mediane sur les canaux, uV^2/Hz
    alpha_peak_hz: float
    line_noise_db: float             # pic 50 Hz au-dessus de son voisinage
    outlier_channels: list[str] = field(default_factory=list)


def spectrum(data: np.ndarray, sfreq: float, ch_names, *, seconds: float = 4.0
             ) -> SpectrumReport:
    """Welch sur le continu, apres retrait de la moyenne de chaque canal.

    Aucun filtre n'est applique avant : le pic secteur doit rester visible pour
    etre mesure. La puissance large bande 1-40 Hz de chaque canal est comparee a
    la mediane des canaux en log, ce qui repere les electrodes plates ou
    decollees sans seuil absolu.
    """
    from scipy.signal import welch

    x = np.asarray(data, dtype=np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    freqs, psd = welch(x, fs=sfreq, nperseg=int(seconds * sfreq), axis=-1)
    med = np.median(psd, axis=0)

    a = (freqs >= 7) & (freqs <= 14)
    alpha = float(freqs[a][np.argmax(med[a])])

    line = np.argmin(np.abs(freqs - LINE_FREQ))
    ring = ((np.abs(freqs - LINE_FREQ) >= 3) & (np.abs(freqs - LINE_FREQ) <= 6))
    line_db = float(10 * np.log10(med[line] / np.median(med[ring])))

    band = (freqs >= 1) & (freqs <= 40)
    trapz = getattr(np, "trapezoid", None) or np.trapz
    logp = np.log10(trapz(psd[:, band], freqs[band], axis=-1))
    dev = logp - np.median(logp)
    mad = np.median(np.abs(dev)) * 1.4826 or 1e-12
    outliers = [str(ch_names[i]) for i in np.flatnonzero(np.abs(dev) > 3 * mad)]
    return SpectrumReport(freqs, med, alpha, line_db, outliers)


# ------------------------------------------------- 2. potentiel evoque
@dataclass
class ErpReport:
    times: np.ndarray
    occipital_mean: np.ndarray       # moyenne sur essais et electrodes occipitales, uV
    gfp: np.ndarray                  # puissance de champ global, moyenne sur essais
    peak_latency_s: float            # pic de GFP entre 60 et 250 ms
    peak_gfp_over_baseline: float    # rapport GFP du pic / GFP moyenne de la ligne de base
    split_half_r: float              # correlation essais pairs / impairs, occipital
    n_trials: int
    channels_used: list[str] = field(default_factory=list)
    #: latences des composantes occipitales. Le pic de GFP peut tomber sur P1 ou
    #: sur N1 d'une session a l'autre quand leurs amplitudes sont proches : ces
    #: latences, mesurees chacune dans sa fenetre, ne sautent pas de composante.
    p1_latency_s: float = float("nan")
    n1_latency_s: float = float("nan")


def cue_erp(data: np.ndarray, stamps: np.ndarray, sfreq: float, ch_names,
            cue_times, *, tmin: float = -0.2, tmax: float = 0.6) -> ErpReport:
    """Potentiel evoque par l'apparition du mot, ligne de base -200 a 0 ms.

    ``data`` doit deja etre filtre sur le continu (0,5-30 Hz). La reference
    moyenne commune est appliquee epoque par epoque.
    """
    n = int(round((tmax - tmin) * sfreq))
    epochs = []
    for t in cue_times:
        i = int(np.searchsorted(stamps, t + tmin))
        if 0 <= i and i + n <= data.shape[1]:
            epochs.append(data[:, i:i + n])
    X = np.asarray(epochs, dtype=np.float64)
    X = X - X.mean(axis=1, keepdims=True)                      # reference moyenne
    times = tmin + np.arange(n) / sfreq
    base = times < 0
    X = X - X[:, :, base].mean(axis=-1, keepdims=True)

    idx = [i for i, c in enumerate(ch_names) if c in OCCIPITAL]
    if not idx:
        raise RuntimeError("aucune electrode occipitale dans le montage")
    occ = X[:, idx].mean(axis=1)                               # (essais, temps)
    gfp = X.mean(axis=0).std(axis=0)                           # GFP de l'evoque moyen
    w = (times >= 0.06) & (times <= 0.25)
    k = np.flatnonzero(w)[np.argmax(gfp[w])]
    r = float(np.corrcoef(occ[0::2].mean(0), occ[1::2].mean(0))[0, 1])
    mean = occ.mean(0)
    w1 = (times >= 0.09) & (times < 0.165)
    w2 = (times >= 0.165) & (times <= 0.24)
    p1 = float(times[w1][np.argmax(mean[w1])])
    n1 = float(times[w2][np.argmin(mean[w2])])
    return ErpReport(times, mean, gfp, float(times[k]),
                     float(gfp[k] / gfp[base].mean()), r, len(X),
                     [ch_names[i] for i in idx], p1, n1)


# ------------------------------------------------------------ 3-4. audio
def _frame_db(x: np.ndarray, sfreq: float, frame_s: float = 0.010) -> np.ndarray:
    n = max(1, int(round(frame_s * sfreq)))
    m = len(x) // n
    if m == 0:
        return np.array([])
    rms = np.sqrt(np.mean(x[:m * n].reshape(m, n) ** 2, axis=1))
    return 20 * np.log10(np.maximum(rms, 1e-12))


def audio_levels(audio_stream, trials: list[Trial], *, onset_db: float = 15.0,
                 hold_s: float = 0.05) -> list[dict]:
    """Niveau sonore de chaque essai, et debut de parole pour les articules.

    Pour chaque essai : le niveau RMS (dB) pendant la fixation, qui sert de
    plancher de bruit propre a l'essai, et pendant la fenetre de production.
    ``rise_db`` est leur difference. Le debut de parole est la premiere trame de
    10 ms, apres le go, qui depasse le plancher de ``onset_db`` dB et s'y
    maintient ``hold_s`` secondes.
    """
    x = np.asarray(audio_stream["time_series"], dtype=np.float64)
    if x.ndim == 2:
        x = x.mean(axis=1)
    t = np.asarray(audio_stream["time_stamps"], float)
    sfreq = float(audio_stream["info"]["nominal_srate"][0])
    hold = max(1, int(round(hold_s / 0.010)))
    rows = []
    for tr in trials:
        def seg(a, b):
            i, j = np.searchsorted(t, [a, b])
            return x[i:j]
        fix = seg(tr.fixation, tr.cue)
        prod = seg(tr.go, tr.end)
        if len(fix) < sfreq * 0.2 or len(prod) < sfreq * 0.2:
            continue
        floor = 20 * np.log10(max(np.sqrt(np.mean(fix ** 2)), 1e-12))
        level = 20 * np.log10(max(np.sqrt(np.mean(prod ** 2)), 1e-12))
        onset = np.nan
        if tr.cond == "articulated":
            fr = _frame_db(prod, sfreq) > floor + onset_db
            run = np.convolve(fr.astype(int), np.ones(hold, int), "valid") == hold
            if run.any():
                onset = float(np.argmax(run) * 0.010)
        rows.append(dict(cond=tr.cond, word=tr.word, floor_db=floor, level_db=level,
                         rise_db=level - floor, onset_s=onset))
    return rows


def acoustic_features(x: np.ndarray, sfreq: float, floor_db: float, *,
                      event_db: float = 15.0) -> dict:
    """Ce qui distingue de la parole un bruit de meme niveau.

    Sur la fenetre de production d'un essai, par trames de 10 ms :

    - ``loud_s`` : duree totale des trames a plus de ``event_db`` dB au-dessus du
      plancher de l'essai (une toux ou un choc dure moins qu'un mot) ;
    - ``first_event_s`` : instant de la premiere de ces trames apres le go ;
    - ``voiced_frac`` : part des trames fortes qui sont voisees, c'est-a-dire dont
      l'autocorrelation a un pic de plus de 0,5 entre 2,5 et 13,3 ms de retard
      (75 a 400 Hz, la plage de la voix) ;
    - ``centroid_hz`` : centroide spectral des trames fortes.
    """
    n = max(1, int(round(0.010 * sfreq)))
    m = len(x) // n
    frames = np.asarray(x[:m * n], dtype=np.float64).reshape(m, n)
    db = 20 * np.log10(np.maximum(np.sqrt(np.mean(frames ** 2, axis=1)), 1e-12))
    loud = db > floor_db + event_db
    out = dict(loud_s=float(loud.sum() * 0.010),
               first_event_s=float(np.argmax(loud) * 0.010) if loud.any() else np.nan,
               peak_rise_db=float(db.max() - floor_db) if m else np.nan,
               voiced_frac=np.nan, centroid_hz=np.nan)
    if not loud.any():
        return out
    # voisement sur des trames de 40 ms centrees sur les trames fortes
    w = int(round(0.040 * sfreq))
    lo, hi = int(sfreq / 400), int(sfreq / 75)
    voiced, cent = [], []
    freqs = np.fft.rfftfreq(w, 1 / sfreq)
    for k in np.flatnonzero(loud):
        a = k * n
        seg = np.asarray(x[a:a + w], dtype=np.float64)
        if len(seg) < w:
            continue
        seg = seg - seg.mean()
        ac = np.correlate(seg, seg, "full")[w - 1:]
        if ac[0] <= 0:
            continue
        ac = ac / ac[0]
        voiced.append(ac[lo:hi].max() > 0.5)
        spec = np.abs(np.fft.rfft(seg * np.hanning(w))) ** 2
        cent.append(float((freqs * spec).sum() / max(spec.sum(), 1e-20)))
    if voiced:
        out["voiced_frac"] = float(np.mean(voiced))
        out["centroid_hz"] = float(np.median(cent))
    return out


def session_id(path) -> str:
    m = re.search(r"ses-(S\d+)", str(path))
    return m.group(1) if m else str(path).rsplit("/", 1)[-1].split(".")[0]
