"""Lecture des XDF bruts et construction des epoques verrouillees par essai.

Le point critique est le marqueur d'ancrage. Le protocole envoie, dans l'ordre :

    {cond}_fixation_start -> {cond}_fixation_end
    {cond}_cue_start:{mot} -> {cond}_cue_end:{mot}     (mot affiche 1.0 s)
    {cond}_isi_start -> {cond}_isi_end                  (ecran vide, 0.8-1.2 s)
    {cond}_start:{mot} -> {cond}_end:{mot}              (production, 2.0 s)
    {cond}_iti_start -> {cond}_iti_end                  (0.5 s)

L'ISI est tire uniformement dans 0.8-1.2 s : c'est une gigue *voulue*, pas une
imprecision d'horloge. Elle se situe entierement AVANT t=0 quand on s'ancre sur
``{cond}_start``, donc elle ne peut pas etaler la fenetre de production. Elle a
en revanche une consequence utile : l'indice visuel se termine a un delai
variable avant t=0, ce qui empeche un classifieur d'exploiter le potentiel evoque
par la lecture du mot. C'est la raison pour laquelle la fenetre par defaut
commence a -0.5 s et pas avant.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MARKER_STREAM = "DarijaSpeechMarkers"
EEG_PREFIX = "actiCHamp-"
CONDITIONS = ("imagined", "articulated")

#: fenetre d'epoque par defaut, en secondes autour du marqueur de production.
#: -0.5 s reste dans l'ISI (ecran vide) meme pour l'ISI le plus court (0.8 s),
#: donc aucune epoque ne contient l'indice visuel.
DEFAULT_TMIN = -0.5
DEFAULT_TMAX = 2.0


@dataclass
class SessionEpochs:
    """Epoques d'une session, deja re-echantillonnees a ``sfreq``."""

    session: str
    X: np.ndarray          # (n_trials, n_channels, n_times), float32
    y: np.ndarray          # (n_trials,) mot, str
    cond: np.ndarray       # (n_trials,) 'imagined' | 'articulated'
    trial_index: np.ndarray  # (n_trials,) rang chronologique dans la session
    sub_block: np.ndarray    # (n_trials,) index du sous-bloc d'enregistrement
    times: np.ndarray      # (n_times,) secondes relatives au go
    ch_names: list[str]
    sfreq: float
    orig_sfreq: float

    def __len__(self) -> int:
        return len(self.X)


def discover_xdf(root: str | Path) -> dict[str, Path]:
    """Associe un identifiant de session au fichier XDF correspondant."""
    root = Path(root)
    found: dict[str, Path] = {}
    for path in sorted(root.rglob("*.xdf")):
        match = re.search(r"ses-(S\d+)", str(path))
        found[match.group(1) if match else path.stem] = path
    return found


def _marker_values(stream) -> list[str]:
    vals = stream["time_series"]
    return [str(v[0] if isinstance(v, (list, np.ndarray)) else v) for v in vals]


def _channel_names(stream) -> list[str]:
    try:
        chans = stream["info"]["desc"][0]["channels"][0]["channel"]
        return [str(c["label"][0]) for c in chans]
    except Exception:
        n = int(stream["info"]["channel_count"][0])
        return [f"ch{i:02d}" for i in range(n)]


def read_events(path: str | Path) -> list[dict]:
    """Relit uniquement le flux de marqueurs. Aucun EEG n'est charge."""
    import pyxdf

    streams, _ = pyxdf.load_xdf(
        str(path), select_streams=[{"name": MARKER_STREAM}], dejitter_timestamps=False
    )
    values = _marker_values(streams[0])
    stamps = np.asarray(streams[0]["time_stamps"], float)

    events = []
    for value, t in zip(values, stamps):
        stage, _, word = value.partition(":")
        for cond in CONDITIONS:
            prefix = cond + "_"
            if stage.startswith(prefix):
                events.append(
                    dict(cond=cond, stage=stage[len(prefix):], word=word or None, t=float(t))
                )
                break
    return events


def _resample_poly(X: np.ndarray, up: int, down: int) -> np.ndarray:
    from scipy.signal import resample_poly

    return resample_poly(X, up, down, axis=-1).astype(np.float32)


def load_session(
    path: str | Path,
    *,
    tmin: float = DEFAULT_TMIN,
    tmax: float = DEFAULT_TMAX,
    target_sfreq: float = 250.0,
    conditions: tuple[str, ...] = CONDITIONS,
    session: str | None = None,
    low: float = 0.5,
    high: float = 45.0,
    do_notch: bool = True,
    scale: float = 1.0,
) -> SessionEpochs:
    """Charge une session et decoupe les epoques ancrees sur ``{cond}_start``.

    Les sessions enregistrees a 500 Hz et celles a 1000 Hz sont ramenees a
    ``target_sfreq`` pour etre analysables ensemble. Le rapport de decimation est
    entier dans les deux cas (4 et 2), donc ``resample_poly`` ne fait
    qu'appliquer un filtre anti-repliement puis sous-echantillonner.

    L'ordre des operations est impose par la physique du signal : le filtrage a
    lieu sur le CONTINU, avant le decoupage. Filtrer epoque par epoque laisse des
    transitoires de bord qui, sur un amplificateur couple en continu, depassent
    largement l'amplitude de l'EEG lui-meme.

    ``scale`` multiplie les donnees brutes. Il sert quand le flux LSL n'est pas
    en microvolts : passer 1e6 si le flux est en volts.
    """
    import pyxdf

    path = Path(path)
    streams, _ = pyxdf.load_xdf(str(path), dejitter_timestamps=True)

    eeg = markers = None
    for stream in streams:
        name = str(stream["info"]["name"][0])
        if name.startswith(EEG_PREFIX) and int(stream["info"]["channel_count"][0]) > 1:
            eeg = stream
        elif name == MARKER_STREAM:
            markers = stream
    if eeg is None or markers is None:
        raise RuntimeError(f"{path.name}: flux EEG ou marqueurs introuvable")

    orig_sfreq = float(eeg["info"]["nominal_srate"][0])
    ch_names = _channel_names(eeg)
    data = np.asarray(eeg["time_series"], dtype=np.float32).T   # (n_channels, n_samples)
    stamps = np.asarray(eeg["time_stamps"], float)
    if data.shape[0] != len(ch_names):
        raise RuntimeError(f"{path.name}: {data.shape[0]} canaux mais {len(ch_names)} noms")

    if scale != 1.0:
        data = data * np.float32(scale)

    # filtrage sur le continu, AVANT le decoupage : voir la note du docstring
    from .preprocess import filter_continuous

    data = filter_continuous(data, orig_sfreq, low=low, high=high, do_notch=do_notch)

    n_samples = int(round((tmax - tmin) * orig_sfreq))
    values = _marker_values(markers)
    mstamps = np.asarray(markers["time_stamps"], float)

    # Les sous-blocs d'enregistrement sont delimites par les marqueurs
    # break_start, que le protocole envoie tous les 30 essais
    # (isBreak = nDone % 30 == 0). Avec 225 essais par bloc cela fait sept
    # pauses, donc huit sous-blocs de 30, 30, 30, 30, 30, 30, 30 et 15 essais.
    # Le compteur repart a zero a chaque block_start, sinon le bloc articule
    # heriterait des pauses du bloc imagine.
    segments, words, conds, order, blocks = [], [], [], [], []
    breaks_seen = 0
    for rank, (value, t0) in enumerate(zip(values, mstamps)):
        stage, _, word = value.partition(":")
        if stage == "block_start":
            breaks_seen = 0
            continue
        if stage == "break_start":
            breaks_seen += 1
            continue
        for cond in conditions:
            if stage != f"{cond}_start":
                continue
            start = int(np.searchsorted(stamps, t0 + tmin))
            if start < 0 or start + n_samples > data.shape[1]:
                continue
            segments.append(data[:, start:start + n_samples])
            words.append(word or None)
            conds.append(cond)
            order.append(rank)
            blocks.append(breaks_seen)
            break

    if not segments:
        raise RuntimeError(f"{path.name}: aucun essai extrait")

    X = np.stack(segments)                       # (n_trials, n_channels, n_times)
    if abs(orig_sfreq - target_sfreq) > 1e-6:
        ratio = orig_sfreq / target_sfreq
        if abs(ratio - round(ratio)) > 1e-6:
            raise RuntimeError(f"{orig_sfreq} Hz -> {target_sfreq} Hz : rapport non entier")
        X = _resample_poly(X, 1, int(round(ratio)))

    times = tmin + np.arange(X.shape[-1]) / target_sfreq
    # rang chronologique 0..n-1 : sert aux plis temporels intra-session
    trial_index = np.argsort(np.argsort(np.asarray(order)))

    return SessionEpochs(
        session=session or re.search(r"ses-(S\d+)", str(path)).group(1),
        X=X.astype(np.float32),
        y=np.asarray(words, dtype=object),
        cond=np.asarray(conds),
        trial_index=trial_index,
        sub_block=np.asarray(blocks, dtype=int),
        times=times,
        ch_names=ch_names,
        sfreq=float(target_sfreq),
        orig_sfreq=orig_sfreq,
    )


def load_rest(
    path: str | Path,
    *,
    duration: float = 0.5,
    target_sfreq: float = 250.0,
    stage: str = "iti_start",
) -> SessionEpochs:
    """Extrait des epoques de repos (ITI ou ISI) de meme duree, pour la detection.

    Le test de detection "parole imaginee contre repos" borne tout le reste : si
    on ne distingue meme pas l'etat de tache de l'etat de repos, la question des
    15 classes est sans objet.
    """
    import pyxdf

    path = Path(path)
    streams, _ = pyxdf.load_xdf(str(path), dejitter_timestamps=True)
    eeg = markers = None
    for stream in streams:
        name = str(stream["info"]["name"][0])
        if name.startswith(EEG_PREFIX) and int(stream["info"]["channel_count"][0]) > 1:
            eeg = stream
        elif name == MARKER_STREAM:
            markers = stream
    if eeg is None or markers is None:
        raise RuntimeError(f"{path.name}: flux manquant")

    orig_sfreq = float(eeg["info"]["nominal_srate"][0])
    ch_names = _channel_names(eeg)
    data = np.asarray(eeg["time_series"], dtype=np.float32).T
    stamps = np.asarray(eeg["time_stamps"], float)
    n_samples = int(round(duration * orig_sfreq))

    segments, conds = [], []
    for value, t0 in zip(_marker_values(markers), np.asarray(markers["time_stamps"], float)):
        name, _, _ = value.partition(":")
        for cond in CONDITIONS:
            if name != f"{cond}_{stage}":
                continue
            start = int(np.searchsorted(stamps, t0))
            if start + n_samples > data.shape[1]:
                continue
            segments.append(data[:, start:start + n_samples])
            conds.append(cond)
            break

    if not segments:
        raise RuntimeError(f"{path.name}: aucune epoque de repos")

    X = np.stack(segments)
    if abs(orig_sfreq - target_sfreq) > 1e-6:
        X = _resample_poly(X, 1, int(round(orig_sfreq / target_sfreq)))

    return SessionEpochs(
        session=re.search(r"ses-(S\d+)", str(path)).group(1),
        X=X.astype(np.float32),
        y=np.asarray(["rest"] * len(X), dtype=object),
        cond=np.asarray(conds),
        trial_index=np.arange(len(X)),
        sub_block=np.zeros(len(X), dtype=int),
        times=np.arange(X.shape[-1]) / target_sfreq,
        ch_names=ch_names,
        sfreq=float(target_sfreq),
        orig_sfreq=orig_sfreq,
    )
