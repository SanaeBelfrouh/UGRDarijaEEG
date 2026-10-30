#!/usr/bin/env python3
"""Etape 14 : un seul tableau de toutes les configurations, et les chiffres du manuscrit.

    python scripts/14_master_table.py --results results

Deux sorties, toutes deux recalculees depuis les CSV de cette execution :

- ``tableau_configurations.csv`` : chaque configuration evaluee par le pipeline
  (fichier source, modele, schema, contraste, recentrage, montage, fenetre,
  exactitude equilibree, seuil de detectabilite, p), pour le materiel
  supplementaire ;
- ``chiffres_manuscrit.csv`` : les nombres cites dans le manuscrit, chacun avec
  sa definition et le fichier dont il vient. Un nombre du texte qui n'est pas
  dans ce fichier n'a pas de source dans cette execution.

Comme tout vient de la meme archive et des memes essais (``run_manifest.json``),
chaque nombre n'a qu'une valeur.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

#: fichiers de resultats a une ligne par configuration -> description
SOURCES = {
    "temoin_articule.csv": "articule, 15 mots, temoin positif",
    "emg_montages.csv": "articule, 15 mots, par montage",
    "controle_temporel_articulated.csv": "articule, fenetre -0,5 a 0 s avant le go",
    "controle_temporel_imagined.csv": "imagine, fenetre -0,5 a 0 s avant le go",
    "imagine_mots15.csv": "imagine, 15 mots, dix modeles",
    "imagine_grille.csv": "imagine, six contrastes, quatre schemas",
    "reconciliation_brut.csv": "imagine, 15 mots, archive sans ICA",
    "reconciliation_ica.csv": "imagine, 15 mots, archive avec ICA",
    "rank_sweep.csv": "imagine, 15 mots, rang retenu",
    "imbrique_resultats.csv": "imagine, selection imbriquee",
    "imbrique_grille.csv": "imagine, chaque configuration de la grille imbriquee",
    "controles_exclusions.csv": "imagine, sans les essais avec son ou clignement",
    "controles_26hz.csv": "imagine, sous 26 Hz",
    "controles_muscle.csv": "puissance 55-95 Hz seule",
}


def configurations(res: Path) -> pd.DataFrame:
    tables = []
    for name, what in SOURCES.items():
        path = res / name
        if not path.exists():
            continue
        t = pd.read_csv(path)
        t.insert(0, "description", what)
        t.insert(0, "source", name)
        tables.append(t)
    return pd.concat(tables, ignore_index=True, sort=False) if tables else pd.DataFrame()


class Chiffres:
    def __init__(self, res: Path):
        self.res = res
        self.rows = []

    def add(self, section, quantite, valeur, source, detail=""):
        if valeur is None or (isinstance(valeur, float) and not np.isfinite(valeur)):
            return
        self.rows.append(dict(section=section, quantite=quantite, valeur=valeur,
                              source=source, detail=detail))

    def read(self, name):
        path = self.res / name
        return pd.read_csv(path) if path.exists() else None

    def best(self, section, name, label, **filters):
        t = self.read(name)
        if t is None or "equilibree" not in t:
            return
        for k, v in filters.items():
            if k in t:
                t = t[t[k] == v]
        if not len(t):
            return
        r = t.loc[t.equilibree.idxmax()]
        detail = " ".join(f"{k}={r[k]}" for k in ("modele", "schema", "contraste", "recentre",
                                                  "montage", "rang") if k in r and pd.notna(r[k]))
        self.add(section, f"{label} : exactitude equilibree", round(float(r.equilibree), 4), name, detail)
        for col, what in (("p_perm", "p (permutation)"), ("p_holm", "p (Holm, famille du fichier)"),
                          ("seuil_eq", "seuil de detectabilite 80 %")):
            if col in r and pd.notna(r[col]):
                self.add(section, f"{label} : {what}", round(float(r[col]), 4), name, detail)


def chiffres(res: Path) -> pd.DataFrame:
    c = Chiffres(res)

    # --- validation technique
    v = c.read("validation_sessions.csv")
    if v is not None:
        for col, what in (("alpha_peak_hz", "pic alpha (Hz)"), ("line_noise_db", "bruit secteur (dB)"),
                          ("n_outlier_channels", "canaux hors norme par session"),
                          ("erp_p1_latency_ms", "latence P1 (ms)"), ("erp_n1_latency_ms", "latence N1 (ms)"),
                          ("erp_peak_gfp_ratio", "pic de puissance de champ global / ligne de base"),
                          ("erp_split_half_r", "correlation essais pairs / impairs")):
            if col in v:
                c.add("validation", f"{what}, minimum", float(v[col].min()), "validation_sessions.csv")
                c.add("validation", f"{what}, maximum", float(v[col].max()), "validation_sessions.csv")
    a = c.read("validation_audio.csv")
    if a is not None and len(a):
        art, ima = a[a.cond == "articulated"], a[a.cond == "imagined"]
        seuil = float(np.percentile(art.rise_db, 5))
        c.add("audio", "montee mediane, articule (dB)", round(float(art.rise_db.median()), 1), "validation_audio.csv")
        c.add("audio", "montee mediane, imagine (dB)", round(float(ima.rise_db.median()), 1), "validation_audio.csv")
        c.add("audio", "5e centile des articules (dB)", round(seuil, 1), "validation_audio.csv")
        c.add("audio", "imagines au-dessus du 5e centile des articules", int((ima.rise_db > seuil).sum()),
              "validation_audio.csv")
        on = 1000 * art.onset_s.dropna()
        c.add("audio", "debut de parole articulee, mediane (ms)", round(float(on.median())), "validation_audio.csv")
        c.add("audio", "debut de parole articulee, 1er quartile (ms)", round(float(on.quantile(.25))), "validation_audio.csv")
        c.add("audio", "debut de parole articulee, 3e quartile (ms)", round(float(on.quantile(.75))), "validation_audio.csv")
        c.add("audio", "essais articules avec parole detectee", int(len(on)), "validation_audio.csv")
    f = c.read("essais.csv")
    if f is not None:
        im = f[f.cond == "imagined"]
        c.add("essais", "imagines avec son > 10 dB", int((im.rise_db > 10).sum()), "essais.csv")
        c.add("essais", "imagines avec clignement en production", int(im.blink_production.sum()), "essais.csv")

    # --- decodage
    c.best("articule", "temoin_articule.csv", "articule, inter-sessions", schema="loso")
    c.best("articule", "temoin_articule.csv", "articule, intra-session", schema="within_session")
    c.best("imagine", "imagine_mots15.csv", "imagine 15 mots, meilleure configuration, inter-sessions",
           schema="loso")
    c.best("imagine", "imagine_mots15.csv", "imagine 15 mots, meilleure configuration, tous schemas")
    t = c.read("imagine_mots15.csv")
    if t is not None and "significatif_holm" in t:
        c.add("imagine", "configurations des 15 mots survivant a Holm", int(t.significatif_holm.astype(bool).sum()),
              "imagine_mots15.csv", f"sur {len(t)}")
        c.add("imagine", "configurations des 15 mots au-dessus du seuil", int((t.equilibree > t.seuil_eq).sum()),
              "imagine_mots15.csv", f"sur {len(t)}")
    n = c.read("imbrique_resultats.csv")
    if n is not None:
        for _, r in n.iterrows():
            c.add("imagine, selection imbriquee", f"{r.contraste} : {r.procedure} : exactitude equilibree",
                  round(float(r.equilibree), 4), "imbrique_resultats.csv", f"{int(r.essais)} essais")
            c.add("imagine, selection imbriquee", f"{r.contraste} : {r.procedure} : p",
                  round(float(r.p), 4), "imbrique_resultats.csv")
    c.best("contrastes", "imagine_grille.csv", "syllabes, meilleure configuration", contraste="syllabes")
    g = c.read("imagine_grille.csv")
    if g is not None and "significatif_holm" in g:
        for contraste, block in g.groupby("contraste"):
            c.add("contrastes", f"{contraste} : configurations survivant a Holm",
                  int(block.significatif_holm.astype(bool).sum()), "imagine_grille.csv", f"sur {len(block)}")
    for cond in ("articulated", "imagined"):
        c.best("controle temporel", f"controle_temporel_{cond}.csv", f"{cond}, avant le go")
    for name in ("controles_exclusions.csv", "controles_26hz.csv", "controles_muscle.csv"):
        t = c.read(name)
        if t is None:
            continue
        for _, r in t.iterrows():
            label = " | ".join(str(r[k]) for k in ("condition", "contraste", "exclusion", "bandes",
                                                   "essais_retenus", "caracteristiques") if k in r and pd.notna(r[k]))
            c.add("controles", f"{label} : exactitude equilibree", round(float(r.equilibree), 4), name)
            c.add("controles", f"{label} : p", round(float(r.p), 4), name)
    h = c.read("controles_hf_hausse.csv")
    if h is not None:
        for (cond, groupe), block in h.groupby(["condition", "groupe"]):
            c.add("controles", f"hausse 55-95 Hz, {cond}, {groupe}, minimum (dB)", round(float(block.hausse_db.min()), 3),
                  "controles_hf_hausse.csv")
            c.add("controles", f"hausse 55-95 Hz, {cond}, {groupe}, maximum (dB)", round(float(block.hausse_db.max()), 3),
                  "controles_hf_hausse.csv")
            c.add("controles", f"hausse 55-95 Hz, {cond}, {groupe}, sessions significatives apres Holm",
                  int((block.p_holm < 0.05).sum()), "controles_hf_hausse.csv", f"sur {len(block)}")
    for name in ("paires_emg_montages.csv", "paires_reconciliation_brut.csv", "paires_reconciliation_ica.csv",
                 "paires_imagine_mots15.csv"):
        t = c.read(name)
        if t is None:
            continue
        for _, r in t.iterrows():
            c.add("comparaisons appariees", f"{r.configuration_b} moins {r.configuration_a} : difference",
                  float(r.difference), name)
            c.add("comparaisons appariees", f"{r.configuration_b} moins {r.configuration_a} : p apparie",
                  float(r.p_apparie), name)
    return pd.DataFrame(c.rows)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default="results")
    p.add_argument("--max-hours", type=float, default=None, help="ignore, pour run_all")
    args = p.parse_args()
    res = Path(args.results)
    manifest = {}
    if (res / "run_manifest.json").exists():
        manifest = json.loads((res / "run_manifest.json").read_text())
    print(f"archive de reference : {manifest.get('archive', 'inconnue (pas de manifeste)')}")

    conf = configurations(res)
    conf.to_csv(res / "tableau_configurations.csv", index=False)
    print(f"tableau_configurations.csv : {len(conf)} configurations, "
          f"{conf.source.nunique() if len(conf) else 0} fichiers")
    ch = chiffres(res)
    ch.to_csv(res / "chiffres_manuscrit.csv", index=False)
    print(f"chiffres_manuscrit.csv : {len(ch)} nombres\n")
    if len(ch):
        with pd.option_context("display.width", 200, "display.max_colwidth", 90, "display.max_rows", 500):
            print(ch[["section", "quantite", "valeur"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
