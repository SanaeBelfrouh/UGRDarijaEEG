#!/usr/bin/env python3
"""Etape 20 : le jeu de donnees publie, XDF anonymises et copie BIDS.

    python scripts/20_release.py --input <dossier des .xdf> --out <dossier de publication>

Ce que fait le script, session par session :

1. **XDF anonymise.** Le fichier est recopie octet par octet ; seules les valeurs
   des echantillons du flux audio sont remplacees, par la meme voix decalee de
   ``--semitones`` demi-tons (librosa, McFee et al. 2015). En-tetes, horodatages,
   decalages d'horloge, EEG et marqueurs restent identiques au bit pres, et le
   fichier garde exactement la meme taille. Le code de l'article lit donc les XDF
   publies comme les originaux.
2. **Copie BIDS** (MNE-BIDS) : l'EEG brut a sa frequence d'origine (1000 ou 500
   Hz), sans filtrage, ni reechantillonnage, ni changement de reference, au format
   BrainVision en float32 ; les marqueurs dans ``events.tsv``, chacun sur
   l'echantillon que le pipeline utilise (le premier dont l'horodatage LSL suit
   celui du marqueur, apres synchronisation des horloges, comme ``load_session``) ;
   l'audio anonymise en WAV a cote, avec son decalage par rapport au premier
   echantillon EEG. Les XDF anonymises sont places dans ``sourcedata/``.
3. **Verification**, sans laquelle rien n'est declare publiable :
   - XDF : EEG, marqueurs et horodatages relus identiques a l'original ; audio de
     meme longueur et memes horodatages ; taille de fichier identique ;
   - BIDS : EEG relu egal a l'XDF a la precision du float32 ; meme nombre
     d'evenements, chacun sur le meme echantillon ;
   - un rapport ``verification.csv`` par session.

Pour tenir en memoire sur 45 min d'audio a 48 kHz, le decalage est applique par
blocs d'environ 60 s, coupes dans un silence et raccordes par un fondu de 10 ms
(voir ``pitch_shift``).

Nommage BIDS : ``sub-01``, sessions ``01`` a ``08`` (S001 a S008), une seule tache
``darijaspeech`` par session, puisque chaque fichier contient le bloc imagine puis
le bloc articule ; la condition, la phase et le mot sont dans ``trial_type``.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import struct
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from darija.io_xdf import discover_xdf                                          # noqa: E402
from darija.validation import channel_names, find_streams                       # noqa: E402

TASK = "darijaspeech"
SUBJECT = "01"
NUMERIC = {"int8": "<i1", "int16": "<i2", "int32": "<i4", "int64": "<i8",
           "float32": "<f4", "double64": "<f8"}


# --------------------------------------------------------------------------
# lecture brute du format XDF
# --------------------------------------------------------------------------

def _read_varlen(buf: memoryview, pos: int) -> tuple[int, int]:
    width = buf[pos]
    if width not in (1, 4, 8):
        raise ValueError(f"longueur variable invalide ({width}) a l'octet {pos}")
    fmt = {1: "<B", 4: "<I", 8: "<Q"}[width]
    return struct.unpack_from(fmt, buf, pos + 1)[0], pos + 1 + width


def chunks(buf: memoryview):
    """(etiquette, debut du contenu, fin du contenu) de chaque bloc du fichier."""
    if bytes(buf[:4]) != b"XDF:":
        raise ValueError("pas un fichier XDF")
    pos = 4
    while pos < len(buf):
        length, pos = _read_varlen(buf, pos)
        tag = struct.unpack_from("<H", buf, pos)[0]
        yield tag, pos + 2, pos + length
        pos += length


def stream_headers(buf: memoryview) -> dict[int, dict]:
    """Identifiant -> nom, type, nombre de canaux et format, depuis les en-tetes."""
    out = {}
    for tag, start, end in chunks(buf):
        if tag != 2:
            continue
        sid = struct.unpack_from("<I", buf, start)[0]
        xml = bytes(buf[start + 4:end]).decode("utf-8", "replace")

        def field(name, default=""):
            m = re.search(rf"<{name}>(.*?)</{name}>", xml, re.S)
            return m.group(1).strip() if m else default

        out[sid] = dict(name=field("name"), type=field("type"),
                        channel_count=int(field("channel_count", "0")),
                        channel_format=field("channel_format"))
    return out


def audio_value_runs(buf: np.ndarray, sid: int, n_ch: int, itemsize: int):
    """Positions des valeurs du flux ``sid``, par series d'echantillons contigus.

    Chaque echantillon est ``[octet d'horodatage (0 ou 8)][horodatage double si
    8][n_ch valeurs]``. LabRecorder n'horodate en general que le premier
    echantillon d'un bloc : les suivants ont un octet 0 et un pas constant, ce qui
    permet de traiter une serie entiere d'un coup. Renvoie, bloc par bloc, des
    tableaux d'indices de debut des valeurs de chaque echantillon.
    """
    view = memoryview(buf)
    value_bytes = n_ch * itemsize
    for tag, start, end in chunks(view):
        if tag != 3 or struct.unpack_from("<I", view, start)[0] != sid:
            continue
        n, pos = _read_varlen(view, start + 4)
        offsets = np.empty(n, dtype=np.int64)
        i = 0
        while i < n:
            flag = buf[pos]
            if flag == 8:
                offsets[i] = pos + 9
                pos += 9 + value_bytes
                i += 1
                continue
            if flag != 0:
                raise ValueError(f"octet d'horodatage invalide ({flag}) a l'octet {pos}")
            # serie d'echantillons sans horodatage : pas constant 1 + value_bytes
            step = 1 + value_bytes
            m = n - i
            flags = buf[pos + step * np.arange(m)]
            run = int(np.argmax(flags != 0)) if np.any(flags != 0) else m
            offsets[i:i + run] = pos + 1 + step * np.arange(run)
            pos += step * run
            i += run
        if pos != end:
            raise ValueError(f"bloc d'echantillons mal lu (fin {pos} au lieu de {end})")
        yield offsets


def read_values(buf: np.ndarray, sid: int, n_ch: int, dtype: np.dtype) -> np.ndarray:
    parts = []
    width = n_ch * dtype.itemsize
    for offsets in audio_value_runs(buf, sid, n_ch, dtype.itemsize):
        idx = offsets[:, None] + np.arange(width)
        parts.append(buf[idx].reshape(-1).view(dtype).reshape(-1, n_ch))
    return np.concatenate(parts) if parts else np.empty((0, n_ch), dtype)


def write_values(buf: np.ndarray, sid: int, n_ch: int, values: np.ndarray) -> None:
    width = n_ch * values.dtype.itemsize
    raw = np.ascontiguousarray(values).view(np.uint8).reshape(len(values), width)
    k = 0
    for offsets in audio_value_runs(buf, sid, n_ch, values.dtype.itemsize):
        idx = offsets[:, None] + np.arange(width)
        buf[idx] = raw[k:k + len(offsets)]
        k += len(offsets)
    if k != len(values):
        raise ValueError(f"{k} echantillons ecrits sur {len(values)}")


# --------------------------------------------------------------------------
# anonymisation de la voix
# --------------------------------------------------------------------------

def quiet_boundaries(x: np.ndarray, sr: float, block_s: float, search_s: float) -> list[int]:
    """Frontieres de blocs placees au moment le plus calme pres de chaque multiple
    de ``block_s`` (trame de 50 ms d'energie minimale a +/- ``search_s``)."""
    n, frame = len(x), int(0.05 * sr)
    bounds = [0]
    for nominal in np.arange(block_s * sr, n - block_s * sr / 2, block_s * sr).astype(int):
        lo, hi = max(bounds[-1] + frame, nominal - int(search_s * sr)), min(n - frame, nominal + int(search_s * sr))
        if hi - lo < frame:
            continue
        seg = x[lo:hi].astype(np.float64)
        m = (hi - lo) // frame
        energy = (seg[:m * frame].reshape(m, frame, -1) ** 2).mean(axis=(1, 2))
        bounds.append(lo + int(np.argmin(energy)) * frame + frame // 2)
    return bounds + [n]


def pitch_shift(x: np.ndarray, sr: float, semitones: float, block_s: float = 60.0,
                search_s: float = 5.0, context_s: float = 0.25, fade_s: float = 0.01) -> np.ndarray:
    """Decale ``x`` (echantillons, canaux) de ``semitones``, meme longueur et type.

    Le decalage d'un seul tenant de 45 min a 48 kHz ne tient pas en memoire. Les
    blocs d'environ ``block_s`` sont donc coupes au moment le plus calme (un
    silence entre deux mots), chacun est decale avec ``context_s`` de contexte de
    chaque cote, puis les blocs sont raccordes par un fondu de ``fade_s``. Un fondu
    long entre deux versions decalees separement creerait des interferences (le
    vocodeur de phase ne garde pas la meme phase d'un bloc a l'autre) ; au milieu
    d'un silence, il n'y a presque rien a raccorder.
    """
    import librosa

    dtype = x.dtype
    scale = float(np.iinfo(dtype).max) if np.issubdtype(dtype, np.integer) else 1.0
    n = len(x)
    ctx, fade = int(context_s * sr), max(1, int(fade_s * sr))
    out = np.empty((n, x.shape[1]), dtype=np.float32)
    bounds = quiet_boundaries(x, sr, block_s, search_s)
    tail = None                                    # [debut, debut + fade) selon le bloc precedent
    for start, stop in zip(bounds[:-1], bounds[1:]):
        lo, hi = max(0, start - ctx), min(n, stop + ctx)
        seg = x[lo:hi].astype(np.float32).T / scale
        y = librosa.effects.pitch_shift(seg, sr=sr, n_steps=semitones).T
        body = y[start - lo:stop - lo].copy()
        if tail is not None:
            k = min(len(tail), len(body))
            w = np.linspace(0.0, 1.0, k, endpoint=False, dtype=np.float32)[:, None]
            body[:k] = (1 - w) * tail[:k] + w * body[:k]
        out[start:stop] = body
        tail = y[stop - lo:stop - lo + fade]
    out *= scale
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        out = np.clip(np.rint(out), info.min, info.max)
    return out.astype(dtype)


def anonymise_xdf(src: Path, dst: Path, semitones: float) -> dict:
    """Recopie ``src`` dans ``dst`` en remplacant la voix. Renvoie un resume."""
    buf = np.frombuffer(bytearray(src.read_bytes()), dtype=np.uint8)
    headers = stream_headers(memoryview(buf))
    audio = [(sid, h) for sid, h in headers.items()
             if h["type"].lower() == "audio" or "audio" in h["name"].lower()]
    if len(audio) != 1:
        raise SystemExit(f"{src.name} : {len(audio)} flux audio, 1 attendu")
    sid, h = audio[0]
    if h["channel_format"] not in NUMERIC:
        raise SystemExit(f"{src.name} : format audio {h['channel_format']} non numerique")
    dtype = np.dtype(NUMERIC[h["channel_format"]])
    original = read_values(buf, sid, h["channel_count"], dtype)

    import pyxdf
    streams, _ = pyxdf.load_xdf(str(src), dejitter_timestamps=True)
    _, markers, aud = find_streams(streams)
    if not np.array_equal(np.asarray(aud["time_series"]).reshape(original.shape), original):
        raise SystemExit(f"{src.name} : lecture brute de l'audio differente de pyxdf")
    sr = float(aud["info"]["nominal_srate"][0])
    shifted = pitch_shift(original, sr, semitones)
    write_values(buf, sid, h["channel_count"], shifted)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(buf.tobytes())

    # niveau de la voix avant et apres, dans chaque fenetre de production articulee
    astamps = np.asarray(aud["time_stamps"], float)
    change = []
    for value, t in zip(markers["time_series"], np.asarray(markers["time_stamps"], float)):
        value = str(value[0] if isinstance(value, (list, np.ndarray)) else value)
        if not value.startswith("articulated_start"):
            continue
        i, j = np.searchsorted(astamps, [t, t + 2.0])
        a0 = original[i:j].astype(np.float64)
        a1 = shifted[i:j].astype(np.float64)
        if len(a0) and a0.std() > 0:
            change.append(20 * np.log10(a1.std() / a0.std()))
    change = np.asarray(change)
    q = np.percentile(change, [10, 50, 90]) if len(change) else [np.nan] * 3
    return dict(audio_stream=h["name"], audio_format=h["channel_format"],
                audio_channels=h["channel_count"], audio_sfreq=sr,
                audio_samples=len(original),
                audio_values_changed=float(np.mean(shifted != original)),
                voice_level_change_db_p10=float(q[0]), voice_level_change_db_median=float(q[1]),
                voice_level_change_db_p90=float(q[2]), voice_windows=int(len(change)))


# --------------------------------------------------------------------------
# BIDS
# --------------------------------------------------------------------------

def load_streams(path: Path):
    """Flux EEG, marqueurs et audio, lus comme le pipeline les lit."""
    import pyxdf

    streams, _ = pyxdf.load_xdf(str(path), dejitter_timestamps=True)
    return find_streams(streams)


def marker_samples(eeg_stamps: np.ndarray, marker_stamps: np.ndarray) -> np.ndarray:
    """Echantillon EEG de chaque marqueur : le premier qui ne le precede pas.

    C'est la regle de ``load_session`` (``np.searchsorted(stamps, t)``), donc un
    essai relu depuis ``events.tsv`` commence sur le meme echantillon que dans
    l'article.
    """
    return np.searchsorted(eeg_stamps, marker_stamps).astype(int)


def eeg_sidecar(ses_label: str, sfreq: float) -> dict:
    return {
        "TaskName": TASK,
        "TaskDescription": (
            "Fifteen Moroccan Darija words, shown in Arabizi, imagined (first block, "
            "225 trials) then articulated aloud (second block, 225 trials). Trial: "
            "fixation 1.0-1.5 s, written word 1.0 s, blank 0.8-1.2 s, go cue (three "
            "asterisks) and 2.0 s production window, inter-trial interval 0.5 s."),
        "Instructions": ("Read the word without preparing articulation, remain still "
                         "during the blank screen, and at the go cue speak the word "
                         "aloud or imagine producing it without movement, according "
                         "to the block."),
        "InstitutionName": "University of Granada",
        "InstitutionalDepartmentName": "Brain, Mind and Behavior Research Center (CIMCYC)",
        "Manufacturer": "Brain Products",
        "ManufacturersModelName": "actiCHamp",
        "CapManufacturer": "Brain Products",
        "CapManufacturersModelName": "actiCAP snap, 64 channels",
        "SoftwareVersions": "BrainVision Recorder 1.20.0601; Lab Streaming Layer LabRecorder",
        "EEGPlacementScheme": "extended 10-20",
        "EEGReference": ("recording reference in the frontopolar region, exact position "
                         "not documented; FCz is a data channel (channel 64)"),
        "EEGGround": "Fpz (hard-wired in the cap)",
        "PowerLineFrequency": 50,
        "SoftwareFilters": "n/a",
        "RecordingType": "continuous",
        "SamplingFrequency": sfreq,
    }


def write_bids_session(xdf_path: Path, root: Path, ses_label: str) -> dict:
    """Ecrit une session en BIDS depuis l'XDF anonymise ; renvoie les controles."""
    import mne
    from mne_bids import BIDSPath, write_raw_bids, read_raw_bids
    from mne_bids.sidecar_updates import update_sidecar_json
    from scipy.io import wavfile

    eeg, markers, audio = load_streams(xdf_path)
    sfreq = float(eeg["info"]["nominal_srate"][0])
    names = channel_names(eeg)
    data_uv = np.asarray(eeg["time_series"], dtype=np.float32).T
    stamps = np.asarray(eeg["time_stamps"], float)
    values = [str(v[0] if isinstance(v, (list, np.ndarray)) else v)
              for v in markers["time_series"]]
    mstamps = np.asarray(markers["time_stamps"], float)
    samples = marker_samples(stamps, mstamps)
    keep = (samples >= 0) & (samples < data_uv.shape[1])

    info = mne.create_info(names, sfreq, "eeg")
    raw = mne.io.RawArray(data_uv.astype(np.float64) * 1e-6, info, verbose=False)
    raw.set_annotations(mne.Annotations(onset=samples[keep] / sfreq, duration=0.0,
                                        description=np.asarray(values)[keep]))
    raw.info["subject_info"] = {"his_id": f"sub-{SUBJECT}", "sex": 2, "hand": 1}
    raw.info["line_freq"] = 50

    bp = BIDSPath(subject=SUBJECT, session=ses_label, task=TASK, datatype="eeg", root=root)
    write_raw_bids(raw, bp, format="BrainVision", allow_preload=True, overwrite=True, readme=False,
                   verbose=False)
    update_sidecar_json(bp.copy().update(suffix="eeg", extension=".json"),
                        eeg_sidecar(ses_label, sfreq), verbose=False)

    # audio anonymise, a cote de l'EEG, avec son decalage dans le temps
    if audio is not None:
        a_sr = float(audio["info"]["nominal_srate"][0])
        a = np.asarray(audio["time_series"])
        beh = root / f"sub-{SUBJECT}" / f"ses-{ses_label}" / "beh"
        beh.mkdir(parents=True, exist_ok=True)
        stem = f"sub-{SUBJECT}_ses-{ses_label}_task-{TASK}_audio"
        wavfile.write(beh / f"{stem}.wav", int(round(a_sr)), a)
        (beh / f"{stem}.json").write_text(json.dumps({
            "SamplingFrequency": a_sr,
            "StartTime": float(audio["time_stamps"][0] - stamps[0]),
            "StartTimeDescription": "seconds from the first EEG sample (LSL clocks synchronised)",
            "Device": "Audio-Technica AT2020USB-XP, LSL AudioCapture",
            "Anonymisation": "fixed pitch shift with librosa (McFee et al. 2015); timing and level kept",
        }, indent=2))

    # relecture et controles
    back = read_raw_bids(bp, verbose=False)
    got = back.get_data() * 1e6
    diff = float(np.max(np.abs(got - data_uv))) if got.shape == data_uv.shape else np.inf
    ev = pd.read_csv(bp.copy().update(suffix="events", extension=".tsv").fpath, sep="\t")
    same_samples = (len(ev) == int(keep.sum())
                    and np.array_equal(ev["sample"].to_numpy(int), samples[keep]))
    grid = stamps[0] + np.arange(len(stamps)) / sfreq
    return dict(eeg_channels=len(names), eeg_sfreq=sfreq, eeg_samples=data_uv.shape[1],
                bids_max_abs_diff_uv=diff,
                bids_float32_tolerance_uv=float(np.max(np.abs(data_uv))) * 2 ** -23,
                markers=len(values), events_written=int(len(ev)),
                events_same_samples=bool(same_samples),
                stamps_max_dev_from_grid_samples=float(np.max(np.abs(stamps - grid)) * sfreq))


# --------------------------------------------------------------------------
# verification de l'XDF anonymise
# --------------------------------------------------------------------------

def compare_xdf(original: Path, released: Path) -> dict:
    import pyxdf

    a, _ = pyxdf.load_xdf(str(original), dejitter_timestamps=True)
    b, _ = pyxdf.load_xdf(str(released), dejitter_timestamps=True)
    ea, ma, aa = find_streams(a)
    eb, mb, ab = find_streams(b)
    va = [str(v[0]) if isinstance(v, (list, np.ndarray)) else str(v) for v in ma["time_series"]]
    vb = [str(v[0]) if isinstance(v, (list, np.ndarray)) else str(v) for v in mb["time_series"]]
    return dict(
        xdf_same_size=original.stat().st_size == released.stat().st_size,
        xdf_eeg_identical=bool(np.array_equal(np.asarray(ea["time_series"]),
                                              np.asarray(eb["time_series"]))),
        xdf_eeg_stamps_identical=bool(np.array_equal(ea["time_stamps"], eb["time_stamps"])),
        xdf_markers_identical=bool(va == vb and np.array_equal(ma["time_stamps"],
                                                               mb["time_stamps"])),
        xdf_audio_stamps_identical=bool(np.array_equal(aa["time_stamps"], ab["time_stamps"])),
        xdf_audio_changed=bool(not np.array_equal(np.asarray(aa["time_series"]),
                                                  np.asarray(ab["time_series"]))),
    )


def dataset_files(root: Path, semitones: float, license_: str) -> None:
    from mne_bids import make_dataset_description

    make_dataset_description(
        path=root, overwrite=True, dataset_type="raw", data_license=license_,
        name="DarijaEEG: imagined and articulated speech in Moroccan Darija",
        authors=["Sanae Belfrouh", "Fatima Zahra Salmam", "Rahhal Errattahi",
                 "Owais Mujtaba Khanday", "Mohamed Baha Ben Ticha", "Marc Ouellet",
                 "Jose A. Gonzalez-Lopez"],
        funding=["PID2022-141378OB-C22 and AIA2025-163317-C32, "
                 "MICIU/AEI/10.13039/501100011033 and ERDF/EU"],
        ethics_approvals=["Ethics Committee for Human Research, University of Granada, "
                          "4210/CEIH/2024"],
        references_and_links=["https://github.com/SanaeBelfrouh/UGRDarijaEEG"],
        verbose=False)
    (root / ".bidsignore").write_text("*_audio.wav\n*_audio.json\nverification.csv\n")
    (root / "README").write_text(
        "DarijaEEG\n=========\n\n"
        "One native speaker of Moroccan Darija, eight sessions on separate days, fifteen "
        "words imagined then articulated (225 trials each per session), 64-channel EEG "
        "(Brain Products actiCHamp), audio at 48 kHz and event markers, synchronised with "
        "the Lab Streaming Layer.\n\n"
        "- sub-01/ses-XX/eeg: raw EEG at the recorded rate (1000 Hz; 500 Hz in ses-06 and "
        "ses-07), BrainVision float32, no filtering; events.tsv holds every protocol "
        "marker (trial_type = condition_phase[:word]).\n"
        "- sub-01/ses-XX/beh: audio of the session, WAV, voice anonymised by a fixed pitch "
        f"shift of {semitones:+g} semitones (librosa); StartTime in the JSON gives its "
        "offset from the first EEG sample.\n"
        "- sourcedata/xdf: the eight XDF files as recorded, with only the audio samples "
        "replaced by the anonymised voice; EEG, markers, timestamps and clock offsets are "
        "unchanged byte for byte. The analysis code reads these files directly.\n\n"
        "Code: https://github.com/SanaeBelfrouh/UGRDarijaEEG (scripts/20_release.py "
        "produced this folder and its verification.csv).\n")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="dossier des XDF originaux")
    p.add_argument("--out", required=True, help="dossier de publication (racine BIDS)")
    p.add_argument("--semitones", type=float, default=5.0,
                   help="decalage de la voix en demi-tons (signe : vers l'aigu si positif)")
    p.add_argument("--license", default="CC-BY-4.0")
    p.add_argument("--sessions", nargs="*", default=None, help="S001 ... (defaut : toutes)")
    args = p.parse_args()
    root = Path(args.out)
    if root.resolve() == Path(args.input).resolve():
        raise SystemExit("--out doit differer de --input : les originaux ne sont jamais modifies")
    root.mkdir(parents=True, exist_ok=True)

    sessions = discover_xdf(args.input)
    if args.sessions:
        sessions = {k: v for k, v in sessions.items() if k in args.sessions}
    if not sessions:
        raise SystemExit(f"aucun XDF sous {args.input}")
    dataset_files(root, args.semitones, args.license)

    rows = []
    for ses, src in sorted(sessions.items()):
        label = f"{int(re.sub(r'\D', '', ses) or 0):02d}"
        t0 = time.time()
        dst = root / "sourcedata" / "xdf" / f"sub-{SUBJECT}_ses-{label}_task-{TASK}.xdf"
        row = dict(session=ses, bids_session=label, source=src.name)
        row.update(anonymise_xdf(src, dst, args.semitones))
        row.update(compare_xdf(src, dst))
        row.update(write_bids_session(dst, root, label))
        row["seconds"] = round(time.time() - t0, 1)
        ok = (row["xdf_same_size"] and row["xdf_eeg_identical"] and row["xdf_eeg_stamps_identical"]
              and row["xdf_markers_identical"] and row["xdf_audio_stamps_identical"]
              and row["xdf_audio_changed"] and row["events_same_samples"]
              and row["bids_max_abs_diff_uv"] <= row["bids_float32_tolerance_uv"])
        row["verifie"] = bool(ok)
        rows.append(row)
        print(f"{ses} -> ses-{label} : EEG {row['eeg_channels']} canaux a {row['eeg_sfreq']:g} Hz, "
              f"{row['markers']} marqueurs, audio {row['audio_samples']} echantillons "
              f"({100 * row['audio_values_changed']:.1f} % modifies, niveau de la voix "
              f"{row['voice_level_change_db_median']:+.1f} dB en mediane) ; ecart BIDS max "
              f"{row['bids_max_abs_diff_uv']:.2e} uV ; {'VERIFIE' if ok else 'ECHEC'} "
              f"({row['seconds']:.0f} s)", flush=True)

    report = pd.DataFrame(rows)
    report.to_csv(root / "verification.csv", index=False)
    bad = report[~report.verifie]
    print(f"\n{len(report) - len(bad)} / {len(report)} sessions verifiees ; rapport : "
          f"{root / 'verification.csv'}")
    if len(bad):
        print("ECHEC pour : " + ", ".join(bad.session) + " — ne pas publier ce dossier")
        return 1
    print("Avant depot : faire passer le validateur BIDS (https://bids-standard.github.io/"
          "bids-validator/) sur ce dossier.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
