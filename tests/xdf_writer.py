"""Ecriture d'un petit fichier XDF, pour les tests de ``20_release.py``.

Le format (https://github.com/sccn/xdf/wiki/Specifications) est une suite de
blocs : ``[octets de longueur][longueur][etiquette uint16][contenu]``. Ce module
n'ecrit que ce dont les tests ont besoin : en-tete de fichier, en-tetes de flux,
echantillons (numeriques ou chaines), decalages d'horloge et pieds de flux. Les
fichiers produits sont relus par ``pyxdf.load_xdf`` comme ceux de LabRecorder.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

NUMERIC = {"int8": "<i1", "int16": "<i2", "int32": "<i4", "int64": "<i8",
           "float32": "<f4", "double64": "<f8"}


def _varlen(n: int) -> bytes:
    if n < 256:
        return bytes([1]) + struct.pack("<B", n)
    if n < 2 ** 32:
        return bytes([4]) + struct.pack("<I", n)
    return bytes([8]) + struct.pack("<Q", n)


def _chunk(tag: int, content: bytes) -> bytes:
    return _varlen(len(content) + 2) + struct.pack("<H", tag) + content


def _header_xml(name: str, kind: str, fmt: str, labels: list[str], srate: float) -> bytes:
    chans = "".join(f"<channel><label>{c}</label></channel>" for c in labels)
    return (f'<?xml version="1.0"?><info><name>{name}</name><type>{kind}</type>'
            f"<channel_count>{len(labels)}</channel_count>"
            f"<nominal_srate>{srate}</nominal_srate><channel_format>{fmt}</channel_format>"
            f"<created_at>0</created_at><desc><channels>{chans}</channels></desc></info>"
            ).encode()


def write_xdf(path: str | Path, streams: list[dict], chunk_size: int = 500) -> None:
    """Ecrit ``streams`` dans ``path``.

    Chaque flux est un dict : ``name``, ``type``, ``format`` (``float32``,
    ``int16``, ``string``...), ``labels`` (noms de canaux), ``srate`` (0 pour un
    flux irregulier), ``data`` (tableau (n, canaux) ou liste de chaines) et
    ``stamps`` (n instants). Les echantillons sont ecrits par blocs de
    ``chunk_size``, entrelaces entre flux, avec un decalage d'horloge nul.
    """
    out = bytearray(b"XDF:")
    out += _chunk(1, b'<?xml version="1.0"?><info><version>1.0</version></info>')
    for sid, s in enumerate(streams, start=1):
        out += _chunk(2, struct.pack("<I", sid)
                      + _header_xml(s["name"], s["type"], s["format"], s["labels"], s["srate"]))
    positions = [0] * len(streams)
    while any(p < len(s["stamps"]) for p, s in zip(positions, streams)):
        for k, s in enumerate(streams):
            p = positions[k]
            n = min(chunk_size, len(s["stamps"]) - p)
            if n <= 0:
                continue
            body = bytearray(struct.pack("<I", k + 1)) + _varlen(n)
            for i in range(p, p + n):
                # comme LabRecorder : un flux regulier n'horodate que le premier
                # echantillon du bloc, pyxdf deduit les suivants de la frequence
                if i == p or not s["srate"]:
                    body += bytes([8]) + struct.pack("<d", float(s["stamps"][i]))
                else:
                    body += bytes([0])
                if s["format"] == "string":
                    value = str(s["data"][i]).encode()
                    body += _varlen(len(value)) + value
                else:
                    body += np.asarray(s["data"][i], dtype=NUMERIC[s["format"]]).tobytes()
            out += _chunk(3, bytes(body))
            positions[k] = p + n
    for sid, s in enumerate(streams, start=1):
        t0, t1 = float(s["stamps"][0]), float(s["stamps"][-1])
        out += _chunk(4, struct.pack("<Idd", sid, t0, 0.0))
        out += _chunk(4, struct.pack("<Idd", sid, t1, 0.0))
        out += _chunk(6, struct.pack("<I", sid) + (
            f'<?xml version="1.0"?><info><first_timestamp>{t0}</first_timestamp>'
            f"<last_timestamp>{t1}</last_timestamp>"
            f"<sample_count>{len(s['stamps'])}</sample_count></info>").encode())
    Path(path).write_bytes(bytes(out))
