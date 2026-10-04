#!/usr/bin/env python3
"""Déroule toutes les analyses du dépôt, dans l'ordre de leur valeur.

Pourquoi ce script existe. Le pipeline vivait dans les cellules d'un notebook
Kaggle, et les cellules d'un notebook Kaggle sont stockées chez Kaggle, pas dans
le dépôt. Un ``git clone`` met donc à jour ``scripts/`` et ``src/`` sans toucher
aux cellules : le notebook dérive silencieusement du code qu'il appelle, et on
relance pendant des heures une version périmée en croyant le contraire. C'est
arrivé. Ici le notebook se réduit à cloner puis appeler ce script, et il n'y a
plus rien à synchroniser.

L'ordre des étapes est celui de leur valeur pour le manuscrit : si le budget est
épuisé, ce qui manque est ce qui compte le moins. Chaque étape reprend là où
elle s'est arrêtée — les CSV déjà écrits sont relus et les blocs faits sautés —
et une étape qui échoue est signalée sans emporter les suivantes.

    python scripts/run_all.py --input /kaggle/input --work /kaggle/working
    python scripts/run_all.py --list
    python scripts/run_all.py --only rang grille
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

#: CSV à supprimer plutôt qu'à reprendre. La référence moyenne commune de
#: l'archive porte sur les 64 canaux : extraire un sous-montage laissait
#: -moyenne(64) dans chaque canal retenu, donc une projection du signal global.
#: Les huit occipitales décodaient ainsi les 15 mots articulés à 11,4 %, ce
#: qu'elles ne captent pas. Corrigé, mais les chiffres d'avant sont faux et le
#: fichier portait alors ce nom.
PERIMES = ("controle_emg",)

#: repères des runs précédents, imprimés pour que la dérive se voie tout de suite
REPERES = {
    "temoin": "articulé : 0,2940 intra-session / 0,1454 LOSO (cov_tangent)",
    "reconciliation": "imaginé : 0,0818 LOSO (bankcov_tangent_lda) = le 8,1 % du manuscrit",
    "mots15": "imaginé 15 mots : 0,0785 au mieux, seuil de détectabilité 0,0813",
}


class Runner:
    """Exécute les scripts du dépôt sous un budget de temps commun."""

    def __init__(self, work: Path, xdf_root: Path, max_hours: float):
        self.t0 = time.time()
        self.max_hours = max_hours
        self.work = work
        self.xdf_root = xdf_root
        self.results = work / "results"
        self.results.mkdir(parents=True, exist_ok=True)
        self.epochs = work / "epochs.npz"
        self.epochs_ica = work / "epochs_ica.npz"
        self.failed: list[tuple[str, str]] = []
        self.skipped: list[str] = []

    # ------------------------------------------------------------------ budget
    def remaining_hours(self) -> float:
        """Réserve proportionnelle, plafonnée à 15 min."""
        reserve = min(0.25, 0.05 * self.max_hours)
        return max(0.0, self.max_hours - (time.time() - self.t0) / 3600 - reserve)

    def run(self, script: str, *args, label: str = "", budget: bool = True) -> int:
        hours = self.remaining_hours()
        if hours <= 0.1:
            print(f"[SAUTE] {label or script} : budget épuisé", flush=True)
            self.skipped.append(label or script)
            return 1
        cmd = [sys.executable, str(HERE / script), *map(str, args)]
        # seuls 02 et 04 balaient une grille et connaissent --max-hours
        if budget:
            cmd += ["--max-hours", f"{hours:.2f}"]
        print(f"\n{'=' * 90}\n{label or script}  (budget restant {hours:.2f} h)\n{'=' * 90}",
              flush=True)
        code = subprocess.run(cmd, cwd=ROOT).returncode
        if code != 0:
            print(f"\n!!! {label or script} a échoué (code {code}) — les étapes "
                  f"suivantes continuent", flush=True)
            self.failed.append((label or script, f"code de sortie {code}"))
        return code

    # ---------------------------------------------------------------- archives
    def archive(self) -> Path:
        """L'archive de la configuration du manuscrit, sinon celle sans ICA."""
        return self.epochs_ica if self.epochs_ica.exists() else self.epochs


def usable(path) -> bool:
    """Une archive n'est réutilisable que si elle porte ``sub_block``.

    Sans ce champ le schéma ``subblock`` est impossible, et c'est celui du
    chiffre principal du manuscrit. Une archive récupérée d'un run antérieur à
    son introduction doit donc être reconstruite, pas réutilisée.
    """
    try:
        with np.load(path, allow_pickle=True) as a:
            return "sub_block" in a.files
    except Exception:
        return False


def is_our_output(name: str) -> bool:
    """Ce fichier est-il une sortie de ce pipeline ?

    La recherche est recursive sur ``/kaggle/input``, qui contient aussi les
    autres datasets attaches au notebook. Sans ce filtre, un ``results.csv``
    quelconque venu d'ailleurs serait copie dans ``results/`` — et s'il portait
    par hasard le nom d'une de nos sorties, il serait repris comme un resultat
    deja calcule.
    """
    if name.startswith("epochs"):
        return False                       # les archives ont leur propre chemin
    base = name
    for suffixe in (".predictions.npz", ".confusions.npz", ".rdm.npz"):
        if name.endswith(suffixe):
            base = name[: -len(suffixe)] + ".csv"
            break
    return base in CONNUS


def flatten_into_results(r: Runner) -> None:
    """Rapatrie dans ``results/`` ce qui a été déposé à la racine du dossier.

    ``kaggle kernels output <kernel> -p /kaggle/working`` télécharge les sorties
    du run précédent directement dans le dossier demandé, et selon la version de
    l'outil il **aplatit** l'arborescence : ``results/imagine.csv`` arrive alors
    à la racine. La reprise ne le voyait pas et recalculait tout.
    """
    for candidate in sorted([*r.work.glob("*.csv"), *r.work.glob("*.npz")]):
        if not is_our_output(candidate.name):        # les archives restent là
            continue
        dest = r.results / candidate.name
        if dest.exists():
            candidate.unlink()
            continue
        shutil.move(str(candidate), dest)
        print(f"{candidate.name} était à la racine -> results/ (téléchargement aplati)")


def recover(r: Runner, inputs: Path, extra: tuple[Path, ...] = ()) -> None:
    """Récupère archives et résultats d'un run précédent, en les validant."""
    flatten_into_results(r)

    for target in (r.epochs, r.epochs_ica):
        if target.exists() and not usable(target):
            print(f"{target.name} présente mais sans sub_block : supprimée, à reconstruire")
            target.unlink()

    sources = [inputs, *extra]
    for target in (r.epochs, r.epochs_ica):
        if target.exists():
            continue
        candidates = [c for src in sources
                      for c in glob.glob(f"{src}/**/{target.name}", recursive=True)]
        for candidate in candidates:
            if usable(candidate):
                size = Path(candidate).stat().st_size / 1e9
                print(f"{target.name} récupérée depuis {candidate} ({size:.2f} Go)")
                shutil.copy(candidate, target)
            else:
                print(f"{candidate} ignorée : pas de champ sub_block, à reconstruire")
            break

    # les .npz de prédictions comptent autant que les CSV : les tests appariés
    # les relisent, et ils sont écrits bloc par bloc comme eux
    # un téléchargement aplati n'a pas de sous-dossier results/, d'où les deux
    # motifs ; le filtre is_our_output empêche de ramasser les CSV des autres
    # datasets attachés au notebook
    recovered = 0
    for src in sources:
        for motif in ("results/*.csv", "results/*.npz", "*.csv", "*.npz"):
            for candidate in glob.glob(f"{src}/**/{motif}", recursive=True):
                name = Path(candidate).name
                if not is_our_output(name):
                    continue
                dest = r.results / name
                if not dest.exists():
                    shutil.copy(candidate, dest)
                    recovered += 1

    for perime in PERIMES:
        for stale in [*r.results.glob(f"{perime}*"), *r.work.glob(f"{perime}*")]:
            stale.unlink()
            print(f"{stale.name} supprimé : invalidé par la fuite de référence "
                  f"des sous-montages, à recalculer")

    present = len(list(r.results.glob("*.csv")))
    if recovered:
        print(f"{recovered} fichiers de résultats récupérés : les blocs faits seront sautés")
    elif present:
        print(f"{present} CSV déjà présents : la reprise les utilisera")
    else:
        print("aucun résultat antérieur : premier run")


# --------------------------------------------------------------------- étapes
def etape_timing(r: Runner) -> None:
    """La gigue observée est-elle l'ISI voulu, ou une dérive d'horloge ?"""
    r.run("00_timing_check.py", "--input", r.xdf_root,
          "--out", r.results / "timing.csv",
          label="4. VÉRIFICATION DU TIMING", budget=False)


def etape_archives(r: Runner) -> None:
    """Les deux archives : sans et avec retrait oculaire par ICA."""
    for target, extra, label in (
        (r.epochs, [], "SANS RETRAIT OCULAIRE"),
        (r.epochs_ica, ["--remove-ocular"], "AVEC RETRAIT OCULAIRE (ICA)"),
    ):
        if target.exists():
            print(f"{target.name} déjà présente "
                  f"({target.stat().st_size / 1e9:.2f} Go) — construction sautée")
            continue
        r.run("01_build_epochs.py", "--input", r.xdf_root, "--out", target, *extra,
              label=f"5. ARCHIVE {label} (~33 min)", budget=False)
        if not target.exists():
            r.failed.append((f"archive {target.name}", "non produite, voir le rapport"))


def etape_temoin(r: Runner) -> None:
    """Le pipeline est-il sensible à un vrai signal ? Sans cela, rien n'est lisible."""
    print(f"  repère : {REPERES['temoin']}")
    r.run("02_benchmark.py",
          "--epochs", r.epochs, "--condition", "articulated",
          "--models", "puissance_lda", "cov_tangent", "xdawn_tangent_lda",
          "--schemes", "within_session", "loso",
          "--contrasts", "mots15",
          "--n-components", 32, "--bank", "manuscrit",
          "--out", r.results / "temoin_articule.csv",
          label="6. TÉMOIN POSITIF — articulé, 15 mots")


def etape_reconciliation(r: Runner) -> None:
    """Le 8,5 % / 8,1 % du manuscrit se reproduit-il, et que vaut l'ICA ?"""
    print(f"  repère : {REPERES['reconciliation']}")
    for archive, suffixe, etiquette in ((r.epochs, "brut", "sans ICA"),
                                        (r.epochs_ica, "ica", "avec ICA")):
        if not archive.exists():
            print(f"[SAUTE] {archive.name} absente")
            continue
        r.run("02_benchmark.py",
              "--epochs", archive, "--condition", "imagined", "--recenter",
              "--models", "bankcov_tangent_lda", "cov_tangent_lda", "xdawn_tangent_lda",
              "--schemes", "subblock", "loso",
              "--contrasts", "mots15",
              "--n-components", 32, "--bank", "manuscrit",
              "--save-predictions",
              "--out", r.results / f"reconciliation_{suffixe}.csv",
              label=f"7. RÉCONCILIATION — imaginé, 15 mots, {etiquette}")


def etape_montages(r: Runner) -> None:
    """Les dix électrodes mandibulaires battent-elles les soixante ?

    Trois modèles sur les MÊMES essais et dans le MÊME fichier. Le modèle
    intermédiaire, ``cov_tangent_lda``, est indispensable : comparer
    ``cov_tangent`` à ``xdawn_tangent_lda`` mélange le filtre et le classifieur,
    donc l'apport de xDAWN seul reste indéterminé sans lui. Écrire dans un seul
    CSV est ce qui accumule les prédictions par essai dans un seul
    ``.predictions.npz``, sans quoi le test apparié n'a rien à comparer.
    """
    for montage in ("complet", "machoire10", "occipital", "sans_machoire"):
        r.run("02_benchmark.py",
              "--epochs", r.epochs, "--condition", "articulated",
              "--montage", montage,
              "--models", "cov_tangent", "cov_tangent_lda", "xdawn_tangent_lda",
              "--schemes", "within_session", "loso",
              "--contrasts", "mots15",
              "--n-components", 32, "--bank", "manuscrit",
              "--save-predictions",
              "--out", r.results / "emg_montages.csv",
              label=f"8. CONTRÔLE MUSCULAIRE — montage {montage}")


def etape_apparies(r: Runner) -> None:
    """Quel écart est réel ? Deux montages sont évalués sur les mêmes essais.

    Comparer deux intervalles de confiance marginaux ignore la corrélation entre
    eux et répond à une autre question. Le test correct échange les prédictions
    essai par essai.
    """
    for nom in ("emg_montages", "reconciliation_brut", "reconciliation_ica",
                "imagine_mots15"):
        preds = r.results / f"{nom}.predictions.npz"
        if not preds.exists():
            print(f"[SAUTE] {preds.name} absent — l'étape qui le produit n'a pas abouti")
            continue
        r.run("05_paired_compare.py", "--predictions", preds,
              "--out", r.results / f"paires_{nom}.csv",
              label=f"9. TEST APPARIÉ — {nom}", budget=False)


def etape_temporel(r: Runner) -> None:
    """Le décodage survit-il à une fenêtre qui ne contient aucun EMG ?

    C'est le seul contrôle que la conduction volumique n'invalide pas : un
    contrôle spatial échoue par construction, puisque l'électromyogramme
    mandibulaire se propage à travers le scalp. Une fenêtre antérieure à la
    contraction ne contient rien à propager.

    Sortie dans un CSV distinct : la clé de reprise de 02_benchmark.py ne
    contient pas la fenêtre, donc écrire au même endroit ferait sauter le bloc
    comme « déjà fait » et afficherait silencieusement l'autre fenêtre.
    """
    for condition in ("articulated", "imagined"):
        r.run("02_benchmark.py",
              "--epochs", r.epochs, "--condition", condition,
              "--tmin", -0.5, "--tmax", 0.0,
              "--models", "cov_tangent", "xdawn_tangent_lda",
              "--schemes", "within_session", "loso",
              "--contrasts", "mots15",
              "--n-components", 32, "--bank", "manuscrit",
              "--out", r.results / f"controle_temporel_{condition}.csv",
              label=f"10. CONTRÔLE TEMPOREL — {condition}, fenêtre -0,5 à 0,0 s")


def etape_rang(r: Runner) -> None:
    """L'effet disparaît-il à rang plein ? Les 32 composantes tiennent-elles ?

    Plateau → seule la nécessité de régulariser compte, le rang exact n'importe
    pas, et la phrase du manuscrit sur le rang plein est à retirer. Pic isolé à
    32 → artefact de sélection, à déclarer comme tel.
    """
    r.run("04_rank_sweep.py",
          "--epochs", r.archive(), "--condition", "imagined",
          "--model", "bankcov_tangent_lda",
          "--schemes", "subblock", "loso",
          "--contrasts", "mots15",
          "--ranks", 8, 16, 24, 32, 40, 48, 56, 64,
          "--scopes", "global", "session",
          "--bank", "manuscrit",
          "--out", r.results / "rank_sweep.csv",
          label="11. COURBE RANG / EXACTITUDE — imaginé, 15 mots")


def etape_mots15(r: Runner) -> None:
    """La question centrale : quelque chose dépasse-t-il 8,2 % sur les 15 mots ?

    Modèles ordonnés du moins cher au plus cher, parce que 02_benchmark.py les
    parcourt dans cet ordre et reprend où il s'est arrêté : une coupure de
    budget ne perd alors que les derniers. ``cov_mdm`` demande 300 à 500 s par
    combinaison et s'effondre sur une classe ; il est là comme référence sans
    hyperparamètre, pas comme candidat.
    """
    print(f"  repère : {REPERES['mots15']}")
    r.run("02_benchmark.py",
          "--epochs", r.archive(), "--condition", "imagined", "--recenter",
          "--models", "puissance_lda", "puissance_logreg", "csp_lda",
          "cov_tangent", "cov_tangent_lda", "bankcov_tangent",
          "bankcov_tangent_lda", "xdawn_tangent_lda", "augcov_tangent",
          "cov_mdm",
          "--schemes", "subblock", "loso",
          "--contrasts", "mots15",
          "--n-components", 32, "--bank", "manuscrit",
          "--save-predictions",
          "--out", r.results / "imagine_mots15.csv",
          label="12. LES 15 MOTS IMAGINÉS — dix modèles")


def etape_grille(r: Runner) -> None:
    """Un contraste structuré ressort-il là où les 15 mots échouent ?

    Moins de classes, plus d'essais par classe, donc un seuil de détectabilité
    plus bas. Le hasard d'un contraste déséquilibré n'est pas 1/k : le tableau
    rapporte l'exactitude équilibrée et son propre seuil.
    """
    r.run("02_benchmark.py",
          "--epochs", r.archive(), "--condition", "imagined", "--recenter",
          "--models", "puissance_lda", "csp_lda", "cov_tangent", "cov_tangent_lda",
          "bankcov_tangent", "bankcov_tangent_lda", "xdawn_tangent_lda",
          "--schemes", "subblock", "within_session", "pooled_blocked", "loso",
          "--contrasts", "mots15", "syllabes", "attaque", "semantique",
          "pharyngale", "oui_non",
          "--n-components", 32, "--bank", "manuscrit",
          "--out", r.results / "imagine_grille.csv",
          label="13. GRILLE COMPLÈTE — 4 schémas x 6 contrastes")


def etape_rsa(r: Runner) -> None:
    """Une structure par paires de mots, là où aucun classifieur ne nomme."""
    for condition in ("imagined", "articulated"):
        r.run("03_rsa.py", "--epochs", r.archive(), "--condition", condition,
              "--out", r.results / f"rsa_{condition}.csv",
              label=f"14. RSA — {condition}, 105 paires de mots", budget=False)


def etape_topo(r: Runner) -> None:
    """Ou se trouve le signal sur le scalp, electrode par electrode.

    Chaque valeur de la carte est une exactitude validee croisee sur une seule
    electrode, avec son propre seuil de detectabilite — pas un patron de filtre
    ajuste sur toutes les donnees. La carte se lit donc comme le reste.

    C'est l'argument visuel du §4.3 : si l'articule s'allume sur l'anneau
    mandibulaire et que l'imagine reste plat, le lecteur le voit avant d'avoir
    lu un tableau.
    """
    # Deux fenetres, et la seconde est celle qui tranche. En production seule,
    # une carte diffuse se lit aussi bien comme un reseau cortical etendu que
    # comme un artefact conduit. Si elle s'eteint AVANT la production, la
    # premiere lecture tombe : un reseau cortical de parole serait deja actif
    # en preparation.
    r.run("07_topomaps.py",
          "--epochs", r.epochs, "--conditions", "articulated", "imagined",
          "--windows", "production:0.0:2.0", "avant:-0.5:0.0",
          "--scheme", "loso", "--n-perm", 5000,
          "--out", r.results / "topo_electrodes.csv",
          label="14 bis. DISCRIMINABILITE PAR ELECTRODE", budget=False)


def etape_figures(r: Runner) -> None:
    """Les six figures du manuscrit, depuis les CSV déjà produits.

    En dernier, parce qu'elle lit tout le reste. Une figure dont la source
    manque est signalée et sautée, les autres sont produites. Rien à relancer si
    un chiffre change : l'étape se rejoue en quelques secondes.
    """
    r.run("06_figures.py", "--results", r.results,
          "--out", r.results / "figures",
          label="15. FIGURES DU MANUSCRIT", budget=False)


#: nom court -> (fonction, ce que l'étape décide). L'ordre est celui de la
#: valeur pour le manuscrit : une coupure ne perd que la fin.
ETAPES = (
    ("timing",         etape_timing,         "la gigue est-elle voulue"),
    ("archives",       etape_archives,       "prérequis, ~33 min x2"),
    ("temoin",         etape_temoin,         "le pipeline est-il sensible"),
    ("reconciliation", etape_reconciliation, "le 8,1 % du manuscrit"),
    ("montages",       etape_montages,       "dix électrodes contre soixante"),
    ("temporel",       etape_temporel,       "fenêtre sans EMG"),
    ("rang",           etape_rang,           "l'effet à rang plein"),
    ("mots15",         etape_mots15,         "quelque chose dépasse-t-il 8,2 %"),
    ("grille",         etape_grille,         "contrastes structurés"),
    ("rsa",            etape_rsa,            "structure par paires"),
    ("apparies",       etape_apparies,       "quel écart est réel"),
    ("topo",           etape_topo,           "où est le signal sur le scalp"),
    ("figures",        etape_figures,        "les sept figures du manuscrit"),
)

#: sortie attendue -> (étape, ce qu'elle décide), pour nommer ce qui manque
ATTENDUS = (
    ("timing.csv", "timing", "la gigue est-elle voulue"),
    ("temoin_articule.csv", "temoin", "le pipeline est-il sensible"),
    ("reconciliation_brut.csv", "reconciliation", "8,1 % sans ICA"),
    ("reconciliation_ica.csv", "reconciliation", "8,1 % avec ICA"),
    ("emg_montages.csv", "montages", "dix électrodes contre soixante"),
    ("paires_emg_montages.csv", "apparies", "écart réel entre montages et modèles"),
    ("controle_temporel_articulated.csv", "temporel", "fenêtre sans EMG, articulé"),
    ("controle_temporel_imagined.csv", "temporel", "fenêtre sans EMG, imaginé"),
    ("rank_sweep.csv", "rang", "l'effet à rang plein"),
    ("imagine_mots15.csv", "mots15", "quelque chose dépasse-t-il 8,2 %"),
    ("paires_imagine_mots15.csv", "apparies", "quel modèle bat quel modèle"),
    ("imagine_grille.csv", "grille", "contrastes structurés"),
    ("rsa_imagined.csv", "rsa", "structure par paires, imaginé"),
    ("rsa_articulated.csv", "rsa", "structure par paires, articulé"),
    ("figures/fig1-protocol-timing.pdf", "figures", "protocole et timing"),
    ("figures/fig2-main-result.pdf", "figures", "les deux conditions sur un axe"),
    ("figures/fig3-muscle-controls.pdf", "figures", "les trois contrôles musculaires"),
    ("figures/fig4-rank-sweep.pdf", "figures", "la courbe rang / exactitude"),
    ("figures/fig5-contrasts.pdf", "figures", "les six contrastes sur un axe commun"),
    ("figures/fig6-rsa.pdf", "figures", "dissimilarité et corrélations de Mantel"),
    ("topo_electrodes.csv", "topo", "discriminabilité électrode par électrode"),
    ("figures/fig7-topomaps.pdf", "figures", "où est le signal sur le scalp"),
)

#: les noms de sortie reconnus, pour ne récupérer que les nôtres. Inclut les
#: anciens noms, afin qu'un run précédent reste réutilisable.
CONNUS = frozenset(
    [name for name, _, _ in ATTENDUS]
    + ["rsa.csv", "imagine.csv", "benchmark.csv"]      # noms d'avant
)

#: seuils sur les 15 mots imaginés, à 1797 essais. SEUIL_80 vaut aussi la
#: colonne `seuil_eq` des CSV, qui est la détectabilité à 80 % de puissance ;
#: la significativité est une quantité distincte et plus basse.
SEUIL_SIG, SEUIL_80, SEUIL_95 = 0.0763, 0.0813, 0.0860


def bilan(r: Runner) -> None:
    import pandas as pd

    print(f"\n{'#' * 90}\nBILAN — {(time.time() - r.t0) / 3600:.2f} h écoulées\n{'#' * 90}")
    print("\nfichiers produits :")
    for f in sorted(r.results.glob("*")):
        if f.is_dir():
            continue
        print(f"  {f.name:40s} {f.stat().st_size / 1e3:9.1f} ko")
    for f in sorted((r.results / "figures").glob("*.pdf")):
        print(f"  figures/{f.name:32s} {f.stat().st_size / 1e3:9.1f} ko")
    for a in (r.epochs, r.epochs_ica):
        if a.exists():
            print(f"  {a.name:40s} {a.stat().st_size / 1e9:9.2f} Go")

    cols = ("modele", "schema", "contraste", "recentre", "montage", "rang",
            "portee", "equilibree", "hasard_eq", "seuil_eq", "p_perm", "p_holm")

    manquants = []
    for name, etape, question in ATTENDUS:
        path = r.results / name
        if not path.exists():
            manquants.append((name, etape, question))
            continue
        if path.suffix != ".csv":          # une figure n'a pas de tableau à lire
            print(f"\n{name} — présent ({path.stat().st_size / 1e3:.0f} ko) — {question}")
            continue
        table = pd.read_csv(path)
        print(f"\n{'=' * 90}\n{name} — {len(table)} lignes — {question}\n{'=' * 90}")
        if "equilibree" not in table.columns:          # tables appariées, RSA
            print(table.to_string(index=False))
            continue
        keep = [c for c in cols if c in table.columns]
        print(table.sort_values("equilibree", ascending=False).head(12)[keep]
              .to_string(index=False))
        above = table[table.equilibree > table.seuil_eq]
        print(f"\nau-dessus du seuil de détectabilité : {len(above)} / {len(table)}")
        if len(above):
            print(above.sort_values("equilibree", ascending=False)[keep]
                  .to_string(index=False))
        if "significatif_holm" in table.columns:
            holm = table[table.significatif_holm.astype(bool)]
            print(f"survivent à Holm dans leur famille : {len(holm)} / {len(table)}")

    print(f"\n{'=' * 90}\nRÉPONSE À LA QUESTION CENTRALE\n{'=' * 90}")
    path = r.results / "imagine_mots15.csv"
    if path.exists():
        table = pd.read_csv(path)
        best = table.loc[table.equilibree.idxmax()]
        print(f"  meilleur sur les 15 mots imaginés : {best.equilibree:.4f} "
              f"({best.modele}/{best.schema}, recentré={bool(best.recentre)})")
        # seuil_eq EST la détectabilité à 80 % : l'afficher sous le libellé
        # « significativité » donnait deux fois le même nombre sous deux noms.
        print(f"  hasard {best.hasard_eq:.4f} "
              f"| significativité {SEUIL_SIG:.4f} "
              f"| détectabilité 80 % {best.seuil_eq:.4f} | 95 % {SEUIL_95:.4f}")
        if best.equilibree > SEUIL_95:
            print("  -> au-dessus de la détectabilité à 95 % : résultat pleinement soutenu")
        elif best.equilibree > best.seuil_eq:
            print("  -> significatif mais sous la détectabilité à 95 % : à publier comme")
            print("     un effet à la limite de résolution de 1800 essais, borne inférieure")
        else:
            print("  -> sous le seuil : aucun signal AU-DESSUS de ce seuil, ce qui n'est")
            print("     pas la même chose qu'aucun signal")
    else:
        print("  imagine_mots15.csv absent : relancer --only mots15")

    if r.failed:
        print(f"\n{'!' * 90}\n{len(r.failed)} ÉTAPE(S) EN ÉCHEC\n{'!' * 90}")
        for label, why in r.failed:
            print(f"  {label} : {why}")
    if r.skipped:
        print(f"\n{'!' * 90}\n{len(r.skipped)} ÉTAPE(S) SAUTÉE(S), BUDGET ÉPUISÉ\n{'!' * 90}")
        for label in r.skipped:
            print(f"  {label}")
    if manquants:
        print(f"\n{'!' * 90}\n{len(manquants)} SORTIE(S) MANQUANTE(S)\n{'!' * 90}")
        etapes = []
        for name, etape, question in manquants:
            print(f"  {name:40s} --only {etape:15s} {question}")
            if etape not in etapes:
                etapes.append(etape)
        print(f"\n  Add Input -> Notebook Output -> ce notebook, puis relancer.")
        print(f"  Pour ne refaire que ce qui manque :")
        print(f"      python scripts/run_all.py --only {' '.join(etapes)}")
    else:
        print("\nToutes les sorties attendues sont présentes. Rien ne manque.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="/kaggle/input", help="racine des XDF")
    parser.add_argument("--work", default="/kaggle/working")
    parser.add_argument("--max-hours", type=float, default=11.0,
                        help="budget total ; Kaggle coupe à 12 h")
    parser.add_argument("--only", nargs="+", default=None,
                        help="n'exécuter que ces étapes")
    parser.add_argument("--skip", nargs="+", default=(),
                        help="sauter ces étapes")
    parser.add_argument("--list", action="store_true",
                        help="lister les étapes et sortir")
    parser.add_argument("--recover-from", nargs="+", default=(),
                        help="dossiers supplémentaires où chercher archives et "
                             "résultats d'un run précédent, par exemple la "
                             "destination d'un `kaggle kernels output -p ...`")
    args = parser.parse_args()

    noms = [n for n, _, _ in ETAPES]
    if args.list:
        print("étapes, dans l'ordre d'exécution :\n")
        for i, (nom, _, quoi) in enumerate(ETAPES, 1):
            print(f"  {i:2d}. {nom:15s} {quoi}")
        return 0

    for groupe in (args.only or (), args.skip):
        inconnus = [n for n in groupe if n not in noms]
        if inconnus:
            raise SystemExit(f"étape(s) inconnue(s) : {inconnus}\ndisponibles : {noms}")

    inputs = Path(args.input)
    xdf = sorted(glob.glob(f"{inputs}/**/*.xdf", recursive=True))
    if not xdf:
        raise SystemExit(f"aucun .xdf sous {inputs}")
    # commonpath sur UN seul fichier renvoie le fichier, pas son dossier : on
    # réduit donc les dossiers parents, jamais les fichiers
    xdf_root = Path(os.path.commonpath([str(Path(f).parent) for f in xdf]))
    print(f"{len(xdf)} fichiers XDF sous {xdf_root}")

    r = Runner(Path(args.work), xdf_root, args.max_hours)
    print(f"dépôt : {ROOT}")
    head = subprocess.run(["git", "-C", str(ROOT), "log", "--oneline", "-1"],
                          capture_output=True, text=True).stdout.strip()
    # imprimer le commit évite de relancer des heures sur une version périmée
    # sans le voir : c'est exactement ce qui est arrivé avec un clone en retard
    print(f"commit : {head or '(inconnu)'}")
    print(f"budget : {args.max_hours} h\n")

    recover(r, inputs, tuple(Path(d) for d in args.recover_from))

    for nom, fonction, quoi in ETAPES:
        if args.only is not None and nom not in args.only:
            continue
        if nom in args.skip:
            print(f"\n[IGNORÉ] {nom} (--skip)")
            continue
        print(f"\n\n{'#' * 90}\n### {nom.upper()} — {quoi}\n{'#' * 90}", flush=True)
        try:
            fonction(r)
        except Exception as e:
            # une étape qui lève ne doit pas emporter les suivantes : sur une
            # nuit de calcul, cela coûterait tout ce qui vient après
            print(f"\n!!! étape {nom} interrompue ({type(e).__name__}: {e})", flush=True)
            r.failed.append((nom, f"{type(e).__name__}: {e}"))

    bilan(r)
    return 1 if r.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
