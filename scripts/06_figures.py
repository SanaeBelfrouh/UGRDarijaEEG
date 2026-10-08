#!/usr/bin/env python3
"""Etape 6 : les six figures du manuscrit, depuis les CSV de resultats.

Les figures sont produites par le pipeline et non a la main, pour la meme raison
que les chiffres : si un resultat change, la figure change avec lui. Les noms de
fichiers sont ceux qu'attend manuscript/tex/darija_eeg.tex.

Trois conventions, et elles portent la lisibilite du manuscrit.

**Les bornes sont tracees.** Hasard, seuil de significativite, et detectabilite
a 80 % et 95 % de puissance apparaissent en lignes horizontales sur toute figure
dont l'ordonnee est une exactitude. Un chiffre de decodage ne se lit pas sans
elles, et c'est precisement ce que la plupart des articles du domaine omettent.
Elles sont RECALCULEES depuis les effectifs par classe, pas codees en dur.

**Pas de titre dans la figure.** La legende LaTeX s'en charge. Un titre duplique
est systematiquement releve en relecture.

**La couleur ne porte jamais l'identite seule.** La palette est validee pour la
daltonie — separation de 9,1 au pire couple adjacent sur l'echelle OKLab x100,
ou le seuil est 8 — mais chaque serie recoit en plus un marqueur et un style de
trait propres. La figure reste donc lisible imprimee en niveaux de gris, et une
courbe isolee se lit sans revenir a la legende.

Une source manquante fait sauter sa figure avec un message, sans emporter les
autres.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import pandas as pd                      # noqa: E402

from darija.stats import minimum_detectable_balanced_accuracy  # noqa: E402

#: 8 pt apres reduction a la largeur de colonne : plancher de la plupart des revues
plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,                  # polices TrueType : exigees par certaines revues
    "axes.spines.top": False,
    "axes.spines.right": False,
})

#: Palette categorique validee : separation CVD (deutan/protan/tritan) de 9,1
#: au pire couple adjacent et 20,8 en vision normale, sur l'echelle OKLab x100.
#: Les seuils sont 8 et 15. Deux slots passent sous 3:1 de contraste avec le
#: fond blanc, ce qui impose des etiquettes directes : elles sont presentes.
#:
#: La couleur ne porte JAMAIS l'identite seule. Chaque serie recoit aussi un
#: marqueur et un style de trait propres, pour trois raisons : une impression en
#: niveaux de gris, une daltonie non couverte par la validation, et la lecture
#: d'une courbe isolee sans revenir a la legende.
COULEURS = {
    "articule": "#2a78d6",      # bleu
    "imagine":  "#eb6834",      # orange
    "complet":       "#2a78d6",
    "sans_machoire": "#1baf7a",  # vert d'eau
    "machoire10":    "#eda100",  # jaune
    "occipital":     "#e34948",  # rouge : le temoin qui se comporte autrement
    "neutre":   "#6b6a66",
    "repere":   "#8a8984",
}

#: Libelles anglais des figures. Le code et les CSV gardent leurs noms internes
#: (francais) ; seul ce qui s'affiche est traduit.
LIBELLES = {
    "articule": "articulated", "imagine": "imagined",
    "loso": "leave-one-session-out", "within_session": "within session",
    "subblock": "grouped by sub-block", "pooled_blocked": "pooled blocked",
    "puissance_lda": "power + LDA", "puissance_logreg": "power + logistic",
    "csp_lda": "CSP + LDA", "cov_mdm": "cov. MDM",
    "cov_tangent": "TS + logistic", "cov_tangent_lda": "TS + sLDA",
    "bankcov_tangent": "FB-TS + logistic", "bankcov_tangent_lda": "FB-TS + sLDA",
    "augcov_tangent": "delay-TS + logistic", "xdawn_tangent_lda": "xDAWN-TS + sLDA",
    "complet": "full", "sans_machoire": "no jaw/occiput", "machoire10": "jaw ring",
    "peripherique14": "peripheral (14)",
    "syllabes": "syllables", "pharyngale": "pharyngeal", "semantique": "semantic",
    "mots15": "15 words", "attaque": "onset", "oui_non": "yes / no",
    "onset_voiced": "onset voicing", "cue_length": "cue length",
    "avant": "pre-production", "isi": "ISI", "iti": "ITI",
}
#: barres du panneau (c) de la figure 3 : montage et nombre d'electrodes
LIBELLES_BARRES = {"complet": "full montage (60)", "machoire10": "jaw ring (10)",
                   "sans_machoire": "no jaw/occiput (46)", "occipital": "occipital (8)"}


def lib(x) -> str:
    return LIBELLES.get(str(x), str(x))


#: formes et traits distincts, pour que la figure survive a une impression en gris
MARQUEURS = ("o", "s", "^", "D", "v", "P", "X", "*", "<", ">")
TRAITS = ("-", "--", "-.", ":")
FORMES_MONTAGE = {"complet": ("o", "-"), "sans_machoire": ("^", "--"),
                  "machoire10": ("s", "-."), "occipital": ("D", "-")}

INTERVALLES = ("fixation", "cue", "isi", "production", "iti")
PRESCRIT = {"fixation": ("uniforme", 1.0, 1.5), "cue": ("fixe", 1.0, 1.0),
            "isi": ("uniforme", 0.8, 1.2), "production": ("fixe", 2.0, 2.0),
            "iti": ("fixe", 0.5, 0.5)}


def bornes(n_trials: int, n_classes: int) -> dict[str, float]:
    """Hasard, significativite et detectabilite, recalcules depuis les effectifs.

    Les effectifs par classe sont supposes equilibres, ce qui est le cas du
    protocole (quinze repetitions par mot et par session). Le seuil de
    significativite est le cas limite de la detectabilite a 50 % de puissance :
    l'exactitude qu'un test unilateral declare significative.
    """
    counts = [n_trials / n_classes] * n_classes
    return {
        "hasard": 1.0 / n_classes,
        "significativite": minimum_detectable_balanced_accuracy(counts, power=0.50),
        "mde80": minimum_detectable_balanced_accuracy(counts, power=0.80),
        "mde95": minimum_detectable_balanced_accuracy(counts, power=0.95),
    }


def trace_bornes(ax, b: dict[str, float], horizontal: bool = True) -> None:
    style = (("hasard", ":", "chance"),
             ("significativite", "-.", "significance"),
             ("mde80", "--", "detectability 80 %"),
             ("mde95", "-", "detectability 95 %"))
    for cle, trait, etiquette in style:
        v = b[cle]
        if not np.isfinite(v):
            continue
        f = ax.axhline if horizontal else ax.axvline
        f(v, color=COULEURS["repere"], lw=0.8, ls=trait, zorder=0,
          label=etiquette)


def ic(serie: pd.Series) -> np.ndarray:
    """Convertit la colonne 'ic95_brute' (du texte '[bas, haut]') en deux bornes."""
    bas, haut = [], []
    for v in serie:
        try:
            a, z = str(v).strip("[]").split(",")
            bas.append(float(a)); haut.append(float(z))
        except Exception:
            bas.append(np.nan); haut.append(np.nan)
    return np.vstack([bas, haut])


# ------------------------------------------------------------------ figure 1
def fig_protocole(res: Path, out: Path) -> str | None:
    src = res / "timing.csv"
    if not src.exists():
        return f"timing.csv absent"
    t = pd.read_csv(src)
    presents = [c for c in INTERVALLES if c in t.columns]
    if not presents:
        return "aucune colonne d'intervalle dans timing.csv"

    fig = plt.figure(figsize=(7.0, 5.2))
    haut = fig.add_subplot(2, 1, 1)
    bas = fig.add_subplot(2, 1, 2)

    # (a) schema d'un essai, dessine plutot que trace
    etapes = [("fixation", 1.27, "0.85"), ("cue", 1.00, "0.65"),
              ("ISI\nrandom", 1.00, "0.45"), ("production\n2.0 s fixed", 2.00, "0.25"),
              ("ITI", 0.50, "0.75")]
    x = 0.0
    for nom, duree, couleur in etapes:
        haut.barh(0, duree, left=x, height=0.5, color=couleur,
                  edgecolor="black", lw=0.6)
        haut.text(x + duree / 2, 0, nom, ha="center", va="center", fontsize=7.5)
        x += duree
    haut.axvline(etapes[0][1] + etapes[1][1] + etapes[2][1], color="black",
                 lw=1.2, ls="--")
    haut.text(etapes[0][1] + etapes[1][1] + etapes[2][1], 0.42, " go, t = 0",
              fontsize=7.5, va="bottom")
    haut.set(xlim=(0, x), ylim=(-0.45, 0.65), yticks=[], xlabel="time (s)")
    haut.text(-0.02, 1.0, "a", transform=haut.transAxes, fontweight="bold",
              ha="right", va="top")

    # (b) ecart-type mesure, en millisecondes, sur un axe logarithmique.
    # Trace en secondes, les trois intervalles pilotes par l'horloge sont des
    # traits plats invisibles a cote des deux jitters voulus ; en echelle log
    # c'est justement leur rapport de 1 a 100 qui porte le message.
    sds = [t[c].dropna().std() * 1000 for c in presents]
    voulu = [PRESCRIT[c][0] == "uniforme" for c in presents]
    y = np.arange(len(presents))
    bas.barh(y, sds, color=[COULEURS["articule"] if v else COULEURS["neutre"]
                            for v in voulu],
             edgecolor="black", lw=0.6, height=0.6)
    for i, (c, sd, v) in enumerate(zip(presents, sds, voulu)):
        forme, lo, hi = PRESCRIT[c]
        attendu = (hi - lo) / np.sqrt(12) * 1000 if forme == "uniforme" else None
        bas.text(sd * 1.15, i, f"{sd:.2f} ms", va="center", fontsize=7.5)
        if attendu:
            # trace EN BLANC sur la barre : pose a cote, le repere disparait
            # sous le bord de la barre puisqu'il tombe au meme endroit
            bas.plot([attendu], [i], marker="|", ms=13, color="white",
                     mew=1.8, lw=0, zorder=5)
    bas.set(yticks=y, yticklabels=[lib(c) for c in presents], xscale="log",
            xlabel="measured standard deviation (ms, log scale)")
    bas.set_xlim(0.5, max(sds) * 4)
    bas.invert_yaxis()
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    bas.legend(handles=[
        Patch(facecolor=COULEURS["articule"], edgecolor="black", lw=0.6,
              label="jitter intended by the protocol"),
        Patch(facecolor=COULEURS["neutre"], edgecolor="black", lw=0.6,
              label="driven by the software clock"),
        Line2D([], [], marker="|", ls="", color="0.2", mew=1.8, ms=10,
               label="expected SD of the uniform draw"),
    ], frameon=False, fontsize=7, loc="lower right")
    bas.text(-0.02, 1.0, "b", transform=bas.transAxes, fontweight="bold",
             ha="right", va="top")

    fig.tight_layout()
    fig.savefig(out / "fig1-protocol-timing.pdf")
    plt.close(fig)
    return None


# ------------------------------------------------------------------ figure 2
def fig_principal(res: Path, out: Path) -> str | None:
    art = res / "temoin_articule.csv"
    ima = res / "imagine_mots15.csv"
    if not art.exists() or not ima.exists():
        return "temoin_articule.csv ou imagine_mots15.csv absent"

    a = pd.read_csv(art); i = pd.read_csv(ima)
    a["condition"] = "articule"; i["condition"] = "imagine"
    emg = res / "emg_montages.csv"
    if emg.exists():                      # complete l'articule des modeles manquants
        e = pd.read_csv(emg)
        e = e[e.get("montage", "complet") == "complet"].copy()
        e["condition"] = "articule"
        a = pd.concat([a, e], ignore_index=True).drop_duplicates(
            subset=["modele", "schema", "recentre", "condition"])
    t = pd.concat([a, i], ignore_index=True)
    t = t[t.contraste == "mots15"]
    if not len(t):
        return "aucune ligne mots15"

    b = bornes(int(t.n.max()), int(t.classes.max()))
    schemas = [s for s in ("loso", "within_session", "subblock") if s in set(t.schema)]
    modeles = sorted(set(t.modele))
    fig, axes = plt.subplots(1, len(schemas), figsize=(3.4 * len(schemas), 4.6),
                             sharey=True, sharex=True)
    axes = np.atleast_1d(axes)

    for ax, sch in zip(axes, schemas):
        # l'exactitude est en ABSCISSE : les bornes sont donc des verticales
        trace_bornes(ax, b, horizontal=False)
        for j, (cond, marq) in enumerate((("articule", "o"), ("imagine", "s"))):
            sub = t[(t.schema == sch) & (t.condition == cond)]
            if not len(sub):
                continue
            y = [modeles.index(m) + (j - 0.5) * 0.26 for m in sub.modele]
            bornes_ic = ic(sub.ic95_brute) if "ic95_brute" in sub else None
            err = (np.abs(bornes_ic - sub.equilibree.values)
                   if bornes_ic is not None else None)
            c = COULEURS[cond]
            ax.errorbar(sub.equilibree.values, y, xerr=err, fmt=marq,
                        ms=5.0, lw=0, elinewidth=0.9, capsize=2,
                        mfc=c if cond == "articule" else "white",
                        mec=c, mew=1.2, ecolor=c, label=lib(cond), zorder=3)
        ax.set(yticks=range(len(modeles)), xlabel="balanced accuracy",
               title=lib(sch), ylim=(-0.6, len(modeles) - 0.4))
        ax.set_xlim(left=0.0)
    axes[0].set_yticklabels([lib(m) for m in modeles])
    axes[0].tick_params(axis="y", labelsize=7.5)
    # une seule legende, dedupliquee
    h, l = axes[0].get_legend_handles_labels()
    vus = dict(zip(l, h))
    fig.legend(vus.values(), vus.keys(), loc="lower center", ncol=3,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out / "fig2-main-result.pdf")
    plt.close(fig)
    return None


# ------------------------------------------------------------------ figure 3
def fig_controles(res: Path, out: Path) -> str | None:
    emg = res / "emg_montages.csv"
    if not emg.exists():
        return "emg_montages.csv absent"
    e = pd.read_csv(emg)
    b = bornes(int(e.n.max()), int(e.classes.max()))

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.9))

    # (a) controle temporel : avant production contre production
    ax = axes[0]
    lignes = []
    for cond in ("articulated", "imagined"):
        f = res / f"controle_temporel_{cond}.csv"
        if f.exists():
            d = pd.read_csv(f); d["fenetre"] = "avant production"
            d["condition"] = cond
            lignes.append(d)
    prod = e[e.get("montage", "complet") == "complet"].copy()
    prod["fenetre"] = "production"; prod["condition"] = "articulated"
    lignes.append(prod)
    ima = res / "imagine_mots15.csv"
    if ima.exists():
        d = pd.read_csv(ima)
        d = d[d.contraste == "mots15"].copy()
        d["fenetre"] = "production"; d["condition"] = "imagined"
        lignes.append(d)
    if len(lignes) > 1:
        d = pd.concat(lignes, ignore_index=True)
        trace_bornes(ax, b)
        etiquettes, xpos = [], 0
        for cond in ("articulated", "imagined"):
            for fen, marq in (("avant production", "s"), ("production", "o")):
                sub = d[(d.condition == cond) & (d.fenetre == fen)]
                if not len(sub):
                    continue
                c = COULEURS["articule" if cond == "articulated" else "imagine"]
                ax.scatter([xpos] * len(sub), sub.equilibree, marker=marq, s=30,
                           facecolor="white" if "avant" in fen else c,
                           edgecolor=c, lw=1.2, zorder=3)
                etiquettes.append(f"{'artic.' if cond == 'articulated' else 'imag.'}\n"
                                  f"{'pre' if 'avant' in fen else 'prod.'}")
                xpos += 1
        ax.set(xticks=range(len(etiquettes)), xticklabels=etiquettes,
               ylabel="balanced accuracy")
        ax.set_ylim(0.0, max(0.5, d.equilibree.max() * 1.1))
    ax.text(-0.02, 1.02, "a", transform=ax.transAxes, fontweight="bold",
            ha="right", va="bottom")

    # (b) l'invariance de session des occipitales
    ax = axes[1]
    if {"loso", "within_session"} <= set(e.schema) and "montage" in e.columns:
        piv = (e[e.contraste == "mots15"]
               .pivot_table(index=["montage", "modele"], columns="schema",
                            values="equilibree"))
        deja = set()
        for (mont, mod), r in piv.iterrows():
            if not {"loso", "within_session"} <= set(r.index):
                continue
            plat = mont == "occipital"
            marq, trait = FORMES_MONTAGE.get(mont, ("o", "--"))
            ax.plot([0, 1], [r["loso"], r["within_session"]],
                    marker=marq, ms=4.0, lw=2.0 if plat else 1.2,
                    color=COULEURS.get(mont, COULEURS["neutre"]),
                    ls=trait, zorder=3 if plat else 2)
            deja.add(mont)
        # une etiquette par montage, posee a la hauteur mediane de ses courbes,
        # sinon les trois modeles ecrivent trois fois le meme nom l'un sur l'autre
        hauteurs = sorted(((piv.xs(m, level="montage")["within_session"].median(), m)
                           for m in deja), reverse=True)
        # deux montages aux exactitudes voisines ecrivent leurs noms l'un sur
        # l'autre : on impose un ecart minimal en descendant la liste
        etendue = ax.get_ylim()[1] - ax.get_ylim()[0]
        mini, precedent = etendue * 0.055, None
        for h, mont in hauteurs:
            if precedent is not None and precedent - h < mini:
                h = precedent - mini
            ax.annotate(lib(mont), (1.03, h), fontsize=7, va="center",
                        fontweight="bold" if mont == "occipital" else "normal")
            precedent = h
        trace_bornes(ax, b)
        ax.set(xticks=[0, 1], xticklabels=["across sessions", "within session"],
               ylabel="balanced accuracy", xlim=(-0.15, 1.5))
    ax.text(-0.02, 1.02, "b", transform=ax.transAxes, fontweight="bold",
            ha="right", va="bottom")

    # (c) l'apport de xDAWN, classifieur constant
    ax = axes[2]
    apports = []
    for nom, cond in (("paires_emg_montages", "articule"),
                      ("paires_reconciliation_brut", "imagine"),
                      ("paires_reconciliation_ica", "imagine")):
        f = res / f"{nom}.csv"
        if not f.exists():
            continue
        p = pd.read_csv(f)
        p = p[p.dimension == "modele"]
        for _, r in p.iterrows():
            a_, b_ = str(r.configuration_a), str(r.configuration_b)
            if "cov_tangent_lda" not in a_ or "xdawn_tangent_lda" not in b_:
                continue
            if "loso" not in a_:
                continue
            mont = a_.split("|")[-1]
            apports.append(dict(condition=cond, montage=mont,
                                apport=-float(r.difference),
                                significatif=bool(r.get("significatif", False))))
    if apports:
        d = pd.DataFrame(apports).drop_duplicates(subset=["condition", "montage"])
        d = d.sort_values("apport")
        couleurs = [COULEURS[c] for c in d.condition]
        ax.barh(range(len(d)), d.apport, color=couleurs, edgecolor="black", lw=0.6)
        ax.axvline(0, color="black", lw=0.8)
        ax.set(yticks=range(len(d)),
               yticklabels=[("imagined, " if r.condition == "imagine" else "")
                            + LIBELLES_BARRES.get(r.montage, lib(r.montage))
                            for _, r in d.iterrows()],
               xlabel="xDAWN gain (accuracy points)")
        for i, (_, r) in enumerate(d.iterrows()):
            if not r.significatif:
                ax.text(r.apport, i, " n.s.", fontsize=6.5, va="center")
    ax.text(-0.02, 1.02, "c", transform=ax.transAxes, fontweight="bold",
            ha="right", va="bottom")

    fig.tight_layout()
    fig.savefig(out / "fig3-muscle-controls.pdf")
    plt.close(fig)
    return None


# ------------------------------------------------------------------ figure 4
def fig_rang(res: Path, out: Path) -> str | None:
    src = res / "rank_sweep.csv"
    if not src.exists():
        return "rank_sweep.csv absent"
    t = pd.read_csv(src)
    if "rang" not in t.columns:
        return "colonne 'rang' absente"
    b = bornes(int(t.n.max()), int(t.classes.max()))

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    trace_bornes(ax, b)
    schemas = sorted(set(t.schema))
    for (sch, port), g in t.groupby(["schema", "portee"]):
        g = g.sort_values("rang")
        i = schemas.index(sch)
        ax.plot(g.rang, g.equilibree,
                marker=MARQUEURS[i], ms=4.5, lw=1.4,
                color=COULEURS["articule" if i == 0 else "imagine"],
                ls="-" if port == "global" else "--",
                mfc="none" if port != "global" else None,
                label=f"{'LOSO' if sch == 'loso' else 'sub-block' if sch == 'subblock' else lib(sch)}, "
                      f"{'pooled basis' if port == 'global' else 'per-session'}")
    ax.axvline(32, color="0.5", lw=0.8, ls=":")
    ax.annotate("rank used elsewhere\nin the paper", (32, ax.get_ylim()[0]),
                fontsize=6.5, ha="center", va="bottom",
                textcoords="offset points", xytext=(0, 4))
    ax.set(xlabel="retained rank", ylabel="balanced accuracy",
           xticks=sorted(set(t.rang)))

    # second axe : la dimension de l'espace tangent, qui explique la decroissance
    sec = ax.twiny()
    sec.set_xlim(ax.get_xlim())
    rangs = sorted(set(t.rang))
    sec.set(xticks=rangs,
            xticklabels=[f"{r * (r + 1) // 2 * 4:,}".replace(",", " ")
                         for r in rangs])
    sec.set_xlabel("tangent-space dimensions (4-band filter bank)")
    sec.tick_params(labelsize=7)
    sec.spines["top"].set_visible(True)

    ax.legend(frameon=False, loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig4-rank-sweep.pdf")
    plt.close(fig)
    return None


# ------------------------------------------------------------------ figure 5
def fig_contrastes(res: Path, out: Path) -> str | None:
    src = res / "imagine_grille.csv"
    if not src.exists():
        return "imagine_grille.csv absent"
    t = pd.read_csv(src)
    # chaque contraste a son propre hasard et son propre seuil : tracer brut est
    # illisible. On trace l'ecart au seuil, ce qui ramene tout sur un axe commun.
    t = t.assign(ecart=t.equilibree - t.seuil_eq)
    ordre = (t.groupby("contraste").ecart.max().sort_values(ascending=False).index)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.axhline(0, color="black", lw=1.0)
    for i, c in enumerate(ordre):
        sub = t[t.contraste == c]
        jitter = np.linspace(-0.22, 0.22, len(sub))
        survit = sub.get("significatif_holm", pd.Series(False, index=sub.index)).astype(bool)
        ax.scatter(i + jitter[~survit.values], sub.ecart[~survit.values],
                   marker="o", s=16, facecolor="white",
                   edgecolor=COULEURS["neutre"], lw=0.6, zorder=2)
        ax.scatter(i + jitter[survit.values], sub.ecart[survit.values],
                   marker="D", s=40, facecolor=COULEURS["articule"],
                   edgecolor="black", lw=0.6, zorder=4,
                   label="survives Holm" if i == 0 else None)
    ax.set(xticks=range(len(ordre)), xticklabels=[lib(c) for c in ordre],
           ylabel="balanced accuracy minus its own bound")
    ax.annotate("above the detectability bound", (-0.45, 0),
                fontsize=7, va="bottom", ha="left", color="0.3",
                textcoords="offset points", xytext=(0, 4))
    h, l = ax.get_legend_handles_labels()
    if h:
        ax.legend(frameon=False, loc="lower left", fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig5-contrasts.pdf")
    plt.close(fig)
    return None


# ------------------------------------------------------------------ figure 6
def fig_rsa(res: Path, out: Path) -> str | None:
    rdms, tables = {}, {}
    for cond in ("imagined", "articulated"):
        f = res / f"rsa_{cond}.rdm.npz"
        c = res / f"rsa_{cond}.csv"
        if f.exists():
            with np.load(f, allow_pickle=True) as z:
                rdms[cond] = (z["rdm"], [str(x) for x in z["labels"]])
        if c.exists():
            tables[cond] = pd.read_csv(c)
    if not rdms and not tables:
        return "aucun fichier rsa_*"

    fig = plt.figure(figsize=(10.0, 4.4))
    # (a) les deux matrices, chacune sur SON echelle : la moyenne differe d'un
    #     facteur 15 entre conditions, et l'imposer sur une echelle commune
    #     effacerait la structure de l'imagine
    for j, cond in enumerate(("imagined", "articulated")):
        if cond not in rdms:
            continue
        ax = fig.add_subplot(1, 3, j + 1)
        m, labels = rdms[cond]
        im = ax.imshow(m, cmap="Blues")
        ax.set(xticks=range(len(labels)), yticks=range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=5.5)
        ax.set_yticklabels(labels, fontsize=5.5)
        moy = float(np.nanmean(m[np.tril_indices(len(labels), -1)]))
        ax.set_xlabel(f"{cond}, mean dissimilarity {moy:+.2f}", fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046)
        if j == 0:
            ax.text(-0.25, 1.02, "a", transform=ax.transAxes, fontweight="bold",
                    ha="right", va="bottom")

    # (b) les correlations de Mantel
    if tables:
        ax = fig.add_subplot(1, 3, 3)
        ref = next(iter(tables.values()))
        modeles = list(ref.modele)
        y = np.arange(len(modeles))
        for j, (cond, d) in enumerate(tables.items()):
            d = d.set_index("modele").reindex(modeles)
            ax.barh(y + (j - 0.5) * 0.38, d.rho_spearman, height=0.36,
                    color=COULEURS["imagine" if cond == "imagined" else "articule"],
                    edgecolor="black", lw=0.5, label=cond)
            for k, (rho, p) in enumerate(zip(d.rho_spearman, d.p_holm)):
                if np.isfinite(p) and p < 0.05:
                    ax.text(rho, y[k] + (j - 0.5) * 0.38, "*", fontsize=9,
                            va="center")
        ax.axvline(0, color="black", lw=0.8)
        ax.set(yticks=y, yticklabels=[lib(m) for m in modeles],
               xlabel="Spearman's rho (Mantel)")
        ax.tick_params(labelsize=7)
        ax.legend(frameon=False, fontsize=7, loc="lower right")
        ax.text(-0.02, 1.02, "b", transform=ax.transAxes, fontweight="bold",
                ha="right", va="bottom")

    fig.tight_layout()
    fig.savefig(out / "fig6-rsa.pdf")
    plt.close(fig)
    return None


# ------------------------------------------------------------------ figure 7
def fig_topo(res: Path, out: Path) -> str | None:
    src = res / "topo_electrodes.csv"
    if not src.exists():
        return "topo_electrodes.csv absent (etape topo)"
    from darija.topo import draw_topomap

    t = pd.read_csv(src)
    if "fenetre" not in t.columns:                 # ancien format, une fenetre
        t = t.assign(fenetre="production")
    conditions = [c for c in ("articulated", "imagined") if c in set(t.condition)]
    fenetres = [f for f in ("production", "avant") if f in set(t.fenetre)]
    fenetres += [f for f in sorted(set(t.fenetre)) if f not in fenetres]
    if not conditions or not fenetres:
        return "ni condition ni fenetre exploitable"

    # Echelle COMMUNE a toutes les cartes : c'est l'ecart entre elles qui est le
    # resultat. Des echelles propres donneraient des cartes d'allure semblable
    # et effaceraient exactement ce qu'on veut montrer.
    vmin, vmax = float(t.equilibree.min()), float(t.equilibree.max())
    seuil = float(t.seuil_eq.iloc[0])

    nl, nc = len(fenetres), len(conditions)
    fig, axes = plt.subplots(nl, nc, figsize=(3.3 * nc + 1.1, 3.5 * nl),
                             squeeze=False)
    im = None
    for li, fen in enumerate(fenetres):
        for ci, cond in enumerate(conditions):
            ax = axes[li][ci]
            g = t[(t.fenetre == fen) & (t.condition == cond)].reset_index(drop=True)
            if not len(g):
                ax.axis("off")
                continue
            surligne = [i for i, r in enumerate(g.itertuples())
                        if bool(getattr(r, "significatif_holm", False))]
            im = draw_topomap(ax, g.equilibree.values, g[["x", "y"]].values,
                              cmap="Blues", vmin=vmin, vmax=vmax,
                              surligne=surligne)
            best = g.loc[g.equilibree.idxmax()]
            if li == 0:
                ax.set_title(cond, fontsize=9, pad=2)
            # le detail chiffre va SOUS la tete : au-dessus, deux titres de
            # trois lignes se touchent des que les panneaux se resserrent
            ax.annotate(f"max {best.equilibree:.3f} ({best.electrode})\n"
                        f"{len(surligne)} electrode{'' if len(surligne) == 1 else 's'} after Holm",
                        (0.5, -0.01), xycoords="axes fraction", ha="center",
                        va="top", fontsize=7.5)
            if ci == 0:
                ax.annotate(lib(fen), (-0.04, 0.5), xycoords="axes fraction",
                            rotation=90, ha="right", va="center", fontsize=9)
    if im is None:
        return "aucune carte tracee"

    fig.subplots_adjust(left=0.06, right=0.84, top=0.94, bottom=0.08,
                        wspace=0.06, hspace=0.22)
    cax = fig.add_axes([0.87, 0.18, 0.022, 0.62])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("balanced accuracy, single electrode", fontsize=8,
                 labelpad=2)
    cb.ax.axhline(seuil, color=COULEURS["occipital"], lw=1.6)
    cb.ax.annotate("detectability\nbound", (-0.6, seuil),
                   xycoords=("axes fraction", "data"), fontsize=7,
                   va="center", ha="right", color=COULEURS["occipital"])
    fig.savefig(out / "fig7-topomaps.pdf")
    plt.close(fig)
    return None


FIGURES = (
    ("fig1-protocol-timing", fig_protocole, "protocole et timing mesure"),
    ("fig2-main-result", fig_principal, "les deux conditions sur un seul axe"),
    ("fig3-muscle-controls", fig_controles, "les trois controles musculaires"),
    ("fig4-rank-sweep", fig_rang, "la courbe rang / exactitude"),
    ("fig5-contrasts", fig_contrastes, "les six contrastes sur un axe commun"),
    ("fig6-rsa", fig_rsa, "matrices de dissimilarite et correlations de Mantel"),
    ("fig7-topomaps", fig_topo, "discriminabilite par electrode sur le scalp"),
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default="results",
                        help="dossier contenant les CSV")
    parser.add_argument("--out", default=None,
                        help="dossier des figures (defaut : <results>/figures)")
    parser.add_argument("--only", nargs="+", default=None,
                        help="ne produire que ces figures")
    args = parser.parse_args()

    res = Path(args.results)
    out = Path(args.out) if args.out else res / "figures"
    out.mkdir(parents=True, exist_ok=True)
    print(f"resultats : {res}\nfigures   : {out}\n")

    produites, sautees = [], []
    for nom, fonction, quoi in FIGURES:
        if args.only is not None and nom not in args.only:
            continue
        try:
            raison = fonction(res, out)
        except Exception as e:
            # une figure qui leve ne doit pas emporter les suivantes
            raison = f"{type(e).__name__}: {e}"
        if raison is None:
            taille = (out / f"{nom}.pdf").stat().st_size / 1e3
            print(f"  [OK]    {nom:22s} {quoi}  ({taille:.0f} ko)")
            produites.append(nom)
        else:
            print(f"  [SAUTE] {nom:22s} {raison}")
            sautees.append((nom, raison))

    print(f"\n{len(produites)} figure(s) produite(s), {len(sautees)} sautee(s)")
    if sautees:
        print("\nPour celles qui manquent, relancer l'etape qui produit leur source :")
        for nom, raison in sautees:
            print(f"  {nom:22s} {raison}")
    return 0 if produites else 1


if __name__ == "__main__":
    raise SystemExit(main())
