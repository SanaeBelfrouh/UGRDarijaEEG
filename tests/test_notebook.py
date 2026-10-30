"""Rejoue chaque etape de scripts/run_all.py sur une archive synthetique.

Le pipeline ne tourne normalement que sur Kaggle, apres une demi-heure de
construction d'epoques. Une option inexistante, un nom de modele mal
orthographie ou un contraste qui ne se resout pas n'apparaissaient donc qu'au
bout de trente minutes de calcul perdu. Pire : les cellules d'un notebook Kaggle
sont stockees chez Kaggle, pas dans le depot, donc le notebook derivait
silencieusement du code qu'il appelle et on relancait sur une version perimee.

Les deux problemes sont traites a la racine : tout le pipeline est dans
scripts/run_all.py, le notebook se reduit a cloner et appeler, et ce test
rejoue chaque etape pour de vrai.

1. **Le notebook reste mince.** Il ne doit appeler aucun script du depot
   directement, sinon la derive recommence.
2. **Vocabulaire.** Chaque option, modele, schema, contraste et montage cite
   dans run_all.py doit exister, verifie contre les add_argument des scripts et
   contre les registres du paquet.
3. **Contrastes.** Chacun doit se RESOUDRE sur les vrais mots, pas seulement
   porter un nom valide.
4. **Execution.** Chaque etape est rejouee sur une archive synthetique au
   format reel. Seules celles qui lisent les XDF bruts sont sautees : il n'y a
   pas de XDF synthetique.

L'archive porte de vrais noms d'electrodes 10-20 et les quinze vrais mots, sans
quoi les montages ne resolvent rien et les contrastes structures levent.
"""

from __future__ import annotations

import ast
import builtins
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from darija.contrasts import CONTRASTS, WORD_FEATURES  # noqa: E402
from darija.montages import CENTRAL, JAW_RING, MONTAGES, OCCIPITAL, PERIPHERAL_14  # noqa: E402
from darija.pipelines import MODELS  # noqa: E402
from synthetic import make_dataset  # noqa: E402

NOTEBOOK = ROOT / "notebooks" / "kaggle_run.ipynb"
RUN_ALL = ROOT / "scripts" / "run_all.py"
SCHEMES = ("loso", "within_session", "pooled_blocked", "subblock")

#: noms que la cellule de reglage publie pour toutes les suivantes
SETUP_NAMES = {
    "T_START", "BUDGET_HOURS", "WORK", "EPOCHS", "EPOCHS_ICA", "RESULTS",
    "XDF_ROOT", "run", "remaining_hours", "usable", "PERIMES",
}

#: valeurs prises par les variables de boucle, pour developper les appels
LOOP_VALUES = {
    "montage": tuple(MONTAGES),
    "condition": ("articulated", "imagined"),
}


# --------------------------------------------------------------------- archive
def build_archive(path: Path, *, n_per_word: int = 2, sfreq: float = 100.0) -> Path:
    """Archive au format exact de scripts/01_build_epochs.py.

    Les noms de canaux sont ceux des montages, completes pour faire 64. Sans
    cela ``resolve`` ne trouve rien et le controle musculaire tourne a vide en
    silence, ce qui est pire qu'une erreur.
    """
    names = list(dict.fromkeys(JAW_RING + OCCIPITAL + PERIPHERAL_14 + CENTRAL))
    names += [f"X{i}" for i in range(64 - len(names))]
    names = names[:64]

    # la fenetre doit couvrir -0,5 a 2,0 s : le controle temporel lit la partie
    # negative, et sfreq doit laisser passer la bande 26-40 Hz du banc
    times = np.arange(-0.5, 2.0, 1 / sfreq)

    # Les etiquettes doivent etre les VRAIS mots : les contrastes structures
    # (attaque, semantique, pharyngale, oui_non) lisent WORD_FEATURES par cle.
    # Avec des w00..w14 ils levent un KeyError, et le test passerait a cote des
    # quatre contrastes qui portent la plus grande partie de la grille.
    mots = np.asarray(sorted(WORD_FEATURES), dtype=object)
    assert len(mots) == 15, f"15 mots attendus, {len(mots)} dans WORD_FEATURES"

    parts = []
    for seed, (cond, effect) in enumerate((("imagined", 0.06), ("articulated", 0.30))):
        X, y, groups, trial_index, _ = make_dataset(
            n_sessions=4, n_words=15, n_per_word=n_per_word, n_channels=64,
            n_times=len(times), sfreq=sfreq, effect=effect,
            session_shift=0.2, seed=seed,
        )
        # make_dataset nomme ses mots w00..w14 ; la correspondance est un simple
        # renommage, l'ordre des essais et le motif spatial sont inchanges
        y = np.asarray([mots[int(str(w)[1:])] for w in y], dtype=object)
        parts.append((X, y, groups, trial_index,
                      np.full(len(X), cond, dtype=object)))

    X = np.concatenate([p[0] for p in parts]).astype(np.float32)
    trial_index = np.concatenate([p[3] for p in parts])
    np.savez_compressed(
        path,
        X=X, times=times, sfreq=sfreq,
        y=np.concatenate([p[1] for p in parts]),
        cond=np.concatenate([p[4] for p in parts]),
        session=np.concatenate([p[2] for p in parts]),
        trial_index=trial_index,
        # trois sous-blocs par session, comme les marqueurs break_start
        sub_block=(trial_index // max(1, len(np.unique(trial_index)) // 3)).astype(int),
        ch_names=np.asarray(names, dtype=object),
    )
    return path


# ------------------------------------------------------------------- lecture
def code_cells() -> list[str]:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return [''.join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def run_all_steps() -> list[tuple[str, str, list[str]]]:
    """Les appels r.run("NN_script.py", ...) de run_all.py, par etape.

    Lus par analyse du source plutot qu'en important le module : l'import
    exigerait pandas, un dossier de travail et les XDF, alors qu'on ne veut ici
    que les arguments litteraux.
    """
    tree = ast.parse(RUN_ALL.read_text(encoding="utf-8"))
    steps = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or not node.name.startswith("etape_"):
            continue
        name = node.name[len("etape_"):]
        for call in ast.walk(node):
            if not (isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and call.func.attr == "run"):
                continue
            args = []
            for a in call.args[1:]:
                if isinstance(a, ast.Constant):
                    args.append(str(a.value))
                elif isinstance(a, ast.UnaryOp) and isinstance(a.op, ast.USub):
                    args.append(str(-a.operand.value))
                elif isinstance(a, ast.Name):
                    args.append(f"<{a.id}>")
                else:
                    args.append("<expr>")
            steps.append((name, call.args[0].value, args))
    return steps


def bound_names(tree: ast.AST) -> set[str]:
    """Tout nom que la cellule definit elle-meme."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {(a.asname or a.name).split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            names |= {a.asname or a.name for a in node.names}
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
            a = node.args
            names |= {x.arg for x in (*a.posonlyargs, *a.args, *a.kwonlyargs)}
            names |= {x.arg for x in (a.vararg, a.kwarg) if x is not None}
        elif isinstance(node, ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, ast.withitem) and node.optional_vars is not None:
            names |= {n.id for n in ast.walk(node.optional_vars)
                      if isinstance(n, ast.Name)}
    return names


def run_calls(src: str) -> list[tuple[str, list[str]]]:
    """Les appels run("NN_script.py", ...) d'une cellule, arguments litteraux.

    Un argument non litteral devient un marqueur : soit un chemin a substituer,
    soit une variable de boucle a developper.
    """
    calls = []
    for node in ast.walk(ast.parse(src)):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "run"):
            continue
        args = []
        for a in node.args[1:]:
            if isinstance(a, ast.Constant):
                args.append(str(a.value))
            elif isinstance(a, ast.UnaryOp) and isinstance(a.op, ast.USub):
                args.append(str(-a.operand.value))
            elif isinstance(a, ast.Name):
                args.append(f"<{a.id}>")
            else:
                args.append("<expr>")
        calls.append((node.args[0].value, args))
    return calls


def script_flags(script: str) -> set[str]:
    txt = (ROOT / "scripts" / script).read_text(encoding="utf-8")
    return set(re.findall(r'add_argument\("(--[a-z0-9-]+)"', txt))


# --------------------------------------------------------------------- tests
def test_notebook_stays_thin():
    """Le notebook ne doit appeler aucun script du depot directement.

    C'est la garantie contre la derive : les cellules vivent chez Kaggle, le
    pipeline vit dans le depot. Si une cellule reprend un appel direct a
    02_benchmark.py, le notebook peut a nouveau vieillir sans qu'on le voie.
    """
    cells = code_cells()
    assert len(cells) <= 4, (
        f"{len(cells)} cellules de code : le notebook doit rester mince "
        f"(prevol, clone, run)"
    )
    for i, src in enumerate(cells):
        directs = re.findall(r'"(\d\d_[a-z_]+\.py)"', src)
        assert not directs, (
            f"cellule {i} appelle {directs} directement : passer par "
            f"scripts/run_all.py, sinon la derive notebook/depot recommence"
        )
    joined = "\n".join(cells)
    assert "run_all.py" in joined, "aucune cellule n'appelle scripts/run_all.py"
    # la commande est passee en liste d'arguments, pas en chaine
    assert '"--oneline"' in joined or "'--oneline'" in joined, (
        "le notebook doit afficher le commit obtenu : c'est ce qui rend un "
        "clone en retard visible avant de lancer des heures de calcul"
    )


def test_every_cell_parses_and_is_self_sufficient():
    for i, src in enumerate(code_cells()):
        tree = ast.parse(src)          # leve SyntaxError si la cellule est cassee
        used = {n.id for n in ast.walk(tree)
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        free = used - bound_names(tree) - set(dir(builtins))
        assert not free, (
            f"cellule {i} utilise sans les definir : {sorted(free)}. Une reprise "
            f"a mi-parcours echouerait sur un NameError."
        )


def test_every_option_and_name_exists():
    flags = {s.name: script_flags(s.name) for s in (ROOT / "scripts").glob("*.py")}
    vocab = {
        "--models": set(MODELS), "--model": set(MODELS),
        "--schemes": set(SCHEMES), "--contrasts": set(CONTRASTS),
        "--montage": set(MONTAGES),
    }
    steps = run_all_steps()
    assert steps, "aucun appel run() trouve dans run_all.py"
    for etape, script, args in steps:
        unknown = {a for a in args if a.startswith("--")} - flags[script]
        assert not unknown, (
            f"etape {etape} passe a {script} des options inexistantes : "
            f"{sorted(unknown)}"
        )
        for pos, arg in enumerate(args):
            if arg not in vocab:
                continue
            for value in args[pos + 1:]:
                if value.startswith("--"):
                    break
                if value.startswith("<"):       # variable de boucle
                    continue
                assert value in vocab[arg], (
                    f"etape {etape}, {script} {arg} : « {value} » n'existe pas "
                    f"(connus : {sorted(vocab[arg])})"
                )


def test_structured_contrasts_resolve_on_real_words():
    """Un nom valide ne suffit pas : apply_contrast indexe WORD_FEATURES par mot."""
    from darija.contrasts import apply_contrast

    words = np.asarray(sorted(WORD_FEATURES), dtype=object)
    trials = np.tile(words, 8)
    cited = set()
    for _, _, args in run_all_steps():
        if "--contrasts" not in args:
            continue
        for value in args[args.index("--contrasts") + 1:]:
            if value.startswith(("--", "<")):
                break
            cited.add(value)
    assert cited, "aucun contraste cite dans run_all.py"
    for name in sorted(cited):
        labels, mask, chance = apply_contrast(name, trials)
        assert mask.sum() > 0, f"le contraste {name} ne retient aucun essai"
        assert len(set(labels)) > 1, f"le contraste {name} n'a qu'une classe"
        assert 0.0 < chance < 1.0, f"hasard aberrant pour {name} : {chance}"


def test_temporal_control_writes_to_its_own_file():
    """La cle de reprise de 02_benchmark ne contient pas la fenetre.

    Ecrire une fenetre non standard dans le CSV de la fenetre de production
    ferait sauter le bloc comme « deja fait », et le tableau afficherait
    silencieusement le chiffre de l'autre fenetre.
    """
    windowed = []
    for etape, script, args in run_all_steps():
        if script != "02_benchmark.py" or "--tmin" not in args:
            continue
        out = args[args.index("--out") + 1] if "--out" in args else ""
        windowed.append((etape, args[args.index("--tmin") + 1], out))
    assert windowed, "aucun controle temporel dans run_all.py"
    for etape, tmin, out in windowed:
        assert float(tmin) < 0.0, f"fenetre attendue avant 0 s, vu {tmin} ({etape})"
        assert "temporel" in out or out.startswith("<"), (
            f"le controle temporel ecrit dans {out} : il lui faut un fichier "
            f"distinct, sinon la reprise saute le bloc"
        )


def test_paired_comparison_runs_after_its_producers():
    """Les etapes appariees doivent venir APRES celles qui ecrivent leurs predictions."""
    import ast as _ast

    tree = _ast.parse(RUN_ALL.read_text(encoding="utf-8"))
    etapes = next(n for n in tree.body
                  if isinstance(n, _ast.Assign)
                  and getattr(n.targets[0], "id", None) == "ETAPES")
    ordre = [e.elts[0].value for e in etapes.value.elts]
    assert "apparies" in ordre, "l'etape appariee est absente de ETAPES"
    producteurs = {etape for etape, script, args in run_all_steps()
                   if "--save-predictions" in args}
    assert producteurs, "aucune etape ne sauvegarde de predictions par essai"
    rang_apparies = ordre.index("apparies")
    for etape in producteurs:
        assert ordre.index(etape) < rang_apparies, (
            f"l'etape {etape} produit des predictions mais passe APRES "
            f"l'etape appariee : celle-ci ne trouverait rien a comparer"
        )


def replay(quick: bool = True) -> list[str]:
    """Rejoue pour de vrai chaque etape de run_all.py sur l'archive synthetique."""
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        archive = build_archive(tmp / "epochs.npz", n_per_word=2 if quick else 4)
        out = tmp / "results"
        out.mkdir()
        with np.load(archive, allow_pickle=True) as a:
            print(f"archive synthetique : {a['X'].shape}, "
                  f"{len(np.unique(a['session']))} sessions, "
                  f"{len(a['ch_names'])} canaux nommes")

        index = 0
        for etape, script, args in run_all_steps():
            if "--input" in args:
                print(f"[SAUTE] {etape:15s} {script} : demande les XDF bruts")
                continue
            variants = [list(args)]
            for name, values in LOOP_VALUES.items():
                token = f"<{name}>"
                if any(token in v for v in variants[0]):
                    variants = [[v.replace(token, val) for v in var]
                                for var in variants for val in values]
            for var in variants:
                index += 1
                cmd, skip = [sys.executable, str(ROOT / "scripts" / script)], False
                i = 0
                while i < len(var):
                    a = var[i]
                    if a == "--epochs":
                        cmd += ["--epochs", str(archive)]; i += 2; continue
                    if a == "--out":
                        cmd += ["--out", str(out / f"{index:02d}_{script[:2]}.csv")]
                        i += 2; continue
                    if a == "--predictions":
                        found = sorted(out.glob("*.predictions.npz"))
                        if not found:
                            skip = True
                            break
                        cmd += ["--predictions", str(found[0])]; i += 2; continue
                    if a.startswith("<"):
                        skip = True
                        break
                    cmd.append(a); i += 1
                if skip:
                    print(f"[SAUTE] {etape:15s} {script} : argument non resoluble "
                          f"hors Kaggle")
                    continue
                if quick:
                    # argparse retient la DERNIERE occurrence : ces options
                    # ecrasent celles de run_all.py. Le rang 8 ramene toute la
                    # geometrie riemannienne a des matrices 8x8, donc les memes
                    # chemins de code cent fois plus vite. On valide les
                    # commandes, pas les chiffres.
                    cmd += ["--n-perm", "200"]
                    if script in ("02_benchmark.py", "04_rank_sweep.py"):
                        cmd += ["--n-splits", "3", "--max-hours", "0.5"]
                    if script == "02_benchmark.py":
                        cmd += ["--n-components", "8"]
                    if script == "04_rank_sweep.py":
                        # 04 balaie les rangs lui-meme et ne connait pas
                        # --n-components : deux rangs suffisent a valider
                        cmd += ["--ranks", "8", "16", "--scopes", "global"]
                label = f"{etape:15s} {script} " + " ".join(
                    x for x in var if not x.startswith(("/", "<")))
                t0 = time.time()
                done = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
                if done.returncode != 0:
                    tail = (done.stderr.strip().splitlines() or ["(vide)"])[-1]
                    print(f"[ECHEC] {label}\n        {tail[:300]}")
                    failures.append(f"{label}\n{done.stderr.strip()[-800:]}")
                else:
                    print(f"[OK]    {label}  ({time.time() - t0:.0f} s)")
    return failures


def test_every_step_runs():
    failures = replay(quick=True)
    assert not failures, "\n\n".join(failures)


if __name__ == "__main__":
    t0 = time.time()
    statiques = [f for f in dir() if f.startswith("test_") and f != "test_every_step_runs"]
    for name in sorted(statiques):
        globals()[name]()
        print(f"[OK] {name}")
    echecs = replay(quick="--full" not in sys.argv)
    print(f"\n{time.time() - t0:.0f} s")
    if echecs:
        print("=" * 78)
        for e in echecs:
            print(e); print("=" * 78)
        raise SystemExit(f"{len(echecs)} commande(s) du notebook echouent")
    print("Toutes les commandes du notebook passent.")
