#!/usr/bin/env python3
"""Etape 2 : l'echelle complete, modele x schema x contraste.

    python scripts/02_benchmark.py --epochs epochs.npz --condition imagined

Ce que le tableau de sortie permet de lire, et que la ligne de base actuelle ne
permettait pas :

1. ``puissance_lda`` en ``loso`` reproduit le resultat actuel. Toute autre ligne
   se lit comme un ecart par rapport a elle, sur les memes plis.
2. La colonne ``schema`` separe deux questions que le resultat a 6-8 % melangeait :
   "y a-t-il du signal" (``within_session``) et "le signal survit-il au changement
   de session" (``loso``).
3. La metrique rapportee et testee est l'exactitude EQUILIBREE, dont le hasard
   vaut exactement 1/n_classes quel que soit le desequilibre des classes.
   L'exactitude brute et le taux de la classe majoritaire restent dans le CSV,
   a titre descriptif, mais ne sont jamais testes : sur des classes
   desequilibrees ils produisent des lignes ininterpretables.
4. La colonne ``seuil_eq`` donne, pour chaque ligne, la plus petite exactitude
   equilibree vraie que ce nombre d'essais permettait de distinguer du hasard.
   Un p non significatif au-dessous de ce seuil n'est pas une absence d'effet.
5. La condition ``articulated`` sert de temoin positif : elle doit ressortir. Si
   elle ne ressort pas, c'est le pipeline qui est en cause, pas le sujet.
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from darija.evaluate import COVARIANCE_REPRESENTATIONS, evaluate_many  # noqa: E402
from darija.features import FILTER_BANK, MANUSCRIPT_BANK                # noqa: E402
from darija.montages import describe, missing_from, resolve             # noqa: E402
from darija.pipelines import build_representations, representation_for  # noqa: E402
from darija.preprocess import crop                                     # noqa: E402
from darija.stats import holm_bonferroni                               # noqa: E402

warnings.filterwarnings("ignore")

# cov_mdm est disponible mais hors defaut : il coute 300-500 s par combinaison et
# s'effondre systematiquement sur une classe, ce que l'exactitude equilibree rend
# desormais visible. Le rajouter explicitement avec --models si on veut la
# reference sans hyperparametre.
DEFAULT_MODELS = ("puissance_lda", "csp_lda", "cov_tangent", "bankcov_tangent")
# subblock est le schema « block-grouped » du manuscrit : groupe par couple
# (session, sous-bloc), donc plus permissif que loso.
DEFAULT_SCHEMES = ("subblock", "within_session", "pooled_blocked", "loso")
DEFAULT_CONTRASTS = ("mots15", "oui_non", "syllabes", "semantique")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", default="epochs.npz")
    parser.add_argument("--condition", default="imagined",
                        choices=("imagined", "articulated", "both"))
    parser.add_argument("--tmin", type=float, default=0.0,
                        help="debut de la fenetre analysee, relatif au go")
    parser.add_argument("--tmax", type=float, default=2.0)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--schemes", nargs="+", default=list(DEFAULT_SCHEMES))
    parser.add_argument("--contrasts", nargs="+", default=list(DEFAULT_CONTRASTS))
    parser.add_argument("--recenter", action="store_true",
                        help="ajoute la variante recentree par session")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-perm", type=int, default=5000)
    parser.add_argument("--out", default="results/benchmark.csv")
    parser.add_argument("--n-components", type=int, default=32,
                        help="reduction de rang avant covariances ; 0 = rang plein")
    parser.add_argument("--rank-scope", default="global", choices=("global", "session"),
                        help="base de la reduction de rang : commune ou par session")
    parser.add_argument("--montage", default="complet",
                        help="sous-ensemble d'electrodes ; 'liste' affiche les montages")
    parser.add_argument("--no-subset-reref", action="store_true",
                        help="ne pas recalculer la reference moyenne apres selection "
                             "d'un sous-montage (permet de reproduire la fuite)")
    parser.add_argument("--bank", default="manuscrit", choices=("manuscrit", "large"),
                        help="banc de filtres : 4 bandes du manuscrit, ou 6 bandes")
    parser.add_argument("--max-hours", type=float, default=11.0,
                        help="budget de temps ; le script s'arrete proprement avant")
    parser.add_argument("--save-predictions", action="store_true",
                        help="enregistre les predictions par essai, requis pour les "
                             "comparaisons appariees entre montages ou modeles")
    parser.add_argument("--restart", action="store_true",
                        help="ignore les resultats deja presents et repart de zero")
    args = parser.parse_args()

    t_start = time.time()
    archive = np.load(args.epochs, allow_pickle=True)
    X, times, sfreq = archive["X"], archive["times"], float(archive["sfreq"])
    y, cond = archive["y"], archive["cond"]
    session, trial_index = archive["session"], archive["trial_index"]
    # les archives construites avant l'ajout des sous-blocs n'ont pas ce champ
    sub_block = archive["sub_block"] if "sub_block" in archive.files else None
    if sub_block is None and "subblock" in args.schemes:
        raise SystemExit("archive sans sub_block : reconstruire avec scripts/01_build_epochs.py")

    ch_names = [str(c) for c in archive["ch_names"]]
    if args.montage == "liste":
        print(describe(ch_names))
        return 0
    if args.montage != "complet":
        keep = resolve(ch_names, args.montage)
        absent = missing_from(ch_names, args.montage)
        X = X[:, keep]
        print(f"montage {args.montage} : {len(keep)} canaux "
              f"({', '.join(ch_names[i] for i in keep)})")
        if absent:
            print(f"  electrodes attendues et absentes du fichier : {', '.join(absent)}")

        # La reference moyenne commune de l'archive porte sur les 64 canaux. En
        # extraire un sous-ensemble laisse donc -moyenne(64) dans chaque canal
        # retenu, c'est-a-dire une projection du signal GLOBAL, EMG mandibulaire
        # compris. Un montage occipital herite ainsi d'information qu'il ne
        # capte pas lui-meme, et la comparaison entre sous-montages perd tout
        # sens : elle compare des canaux qui contiennent tous la meme moyenne.
        # On re-reference donc a l'interieur du sous-ensemble.
        if not args.no_subset_reref:
            from darija.preprocess import common_average_reference
            X = common_average_reference(X)
            print(f"  reference moyenne recalculee sur les {len(keep)} canaux retenus")
        else:
            print("  reference de l'archive conservee (--no-subset-reref) : "
                  "les canaux retenus contiennent la moyenne des 64")

    if args.condition != "both":
        mask = cond == args.condition
        X, y, session, trial_index = X[mask], y[mask], session[mask], trial_index[mask]
        if sub_block is not None:
            sub_block = sub_block[mask]

    X, times = crop(X, times, args.tmin, args.tmax)
    print(f"{len(X)} essais | {X.shape[1]} canaux | fenetre "
          f"[{times[0]:.2f}, {times[-1]:.2f}] s | {len(np.unique(session))} sessions\n")

    bands = MANUSCRIPT_BANK if args.bank == "manuscrit" else FILTER_BANK
    n_components = args.n_components or None

    needed = {representation_for(m) for m in args.models}
    t0 = time.time()
    representations = build_representations(
        X, sfreq, bands=bands, include=tuple(needed),
        n_components=n_components, groups=session, rank_scope=args.rank_scope,
    )
    print(f"banc {args.bank} ({len(bands)} bandes) | rang "
          f"{n_components or X.shape[1]} ({args.rank_scope})")
    print(f"representations calculees en {time.time() - t0:.0f} s : "
          f"{ {k: v.shape for k, v in representations.items()} }\n")

    recenter_options = (False, True) if args.recenter else (False,)

    # Reprise sur interruption. Kaggle coupe a 12 h : sans cela, une coupure
    # perd la totalite du calcul. Les resultats sont ecrits apres CHAQUE bloc
    # (modele, schema, recentrage), et un redemarrage saute ce qui est deja la.
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    done: set[tuple] = set()
    if out_path.exists() and not args.restart:
        previous = pd.read_csv(out_path)
        rows = previous.to_dict("records")
        done = {(r["modele"], r["schema"], bool(r["recentre"]), r["contraste"],
                 r.get("montage", "complet")) for r in rows}
        print(f"reprise : {len(rows)} lignes deja presentes dans {out_path}\n")

    # Les predictions par essai font PARTIE de la sortie : sans elles le test
    # apparie n'a rien a comparer. Un bloc deja present dans le CSV mais absent
    # du magasin de predictions doit donc etre recalcule. Le cas arrive des
    # qu'on reprend un run anterieur a --save-predictions, ou qu'on ajoute un
    # modele a une commande deja passee : la reprise sautait alors les blocs
    # faits, le magasin ne contenait que le nouveau modele, et la comparaison
    # appariee repondait « aucune paire comparable » sans dire pourquoi.
    pred_path = out_path.with_suffix(".predictions.npz")
    pred_done: set[tuple] = set()
    if args.save_predictions and pred_path.exists() and not args.restart:
        with np.load(pred_path, allow_pickle=True) as store:
            for key in store.files:
                parts = key.rsplit("|", 1)[0].split("|")
                if len(parts) == 5:
                    # cle : modele|schema|contraste|recentre|montage
                    pred_done.add((parts[0], parts[1], parts[3] == "1",
                                   parts[2], parts[4]))
        print(f"          dont {len(pred_done)} avec leurs predictions par essai\n")

    def block_keys(model, scheme, recenter):
        return [(model, scheme, recenter, c, args.montage) for c in args.contrasts]

    def block_is_done(model, scheme, recenter):
        keys = block_keys(model, scheme, recenter)
        if not all(k in done for k in keys):
            return False
        if args.save_predictions and not all(k in pred_done for k in keys):
            return False
        return True

    def forget(model, scheme, recenter):
        """Retire les anciennes lignes d'un bloc recalcule, pour ne pas doubler."""
        stale = set(block_keys(model, scheme, recenter))
        rows[:] = [r for r in rows
                   if (r["modele"], r["schema"], bool(r["recentre"]), r["contraste"],
                       r.get("montage", "complet")) not in stale]

    def flush():
        pd.DataFrame(rows).to_csv(out_path, index=False)

    results = []
    deadline = t_start + args.max_hours * 3600
    stopped_early = []
    failed = []

    # les contrastes sont en boucle INTERNE : pour un couple (modele, schema), le
    # prefixe independant des etiquettes est ajuste une fois par pli et sert a
    # tous les contrastes partageant le meme masque d'essais
    for model, scheme, recenter in itertools.product(
        args.models, args.schemes, recenter_options
    ):
        if recenter and representation_for(model) not in COVARIANCE_REPRESENTATIONS:
            continue
        if block_is_done(model, scheme, recenter):
            print(f"  {model:18s} {scheme:15s} "
                  f"{'recentre' if recenter else '        '} : deja fait, saute")
            continue
        if time.time() > deadline:
            stopped_early.append((model, scheme, recenter))
            continue

        t0 = time.time()
        try:
            block = evaluate_many(
                representations, y, session, trial_index,
                model=model, scheme=scheme, contrasts=tuple(args.contrasts),
                recenter=recenter, n_splits=args.n_splits, n_perm=args.n_perm,
                sub_block=sub_block,
            )
        except Exception as e:
            # Un seul couple (modele, schema) qui leve ne doit pas emporter la
            # grille entiere : sur une nuit de calcul cela coute tous les blocs
            # suivants. Le bloc est signale, aucune ligne n'est ecrite pour lui
            # — donc une relance le reprendra — et la grille continue.
            print(f"  {model:18s} {scheme:15s} "
                  f"{'recentre' if recenter else '        '} : ECHEC "
                  f"({type(e).__name__}: {e})", flush=True)
            failed.append((model, scheme, recenter, f"{type(e).__name__}: {e}"))
            continue
        elapsed = time.time() - t0
        forget(model, scheme, recenter)
        results.extend(block)
        for result in block:
            row = result.row()
            row["montage"] = args.montage
            row["reref_sous_montage"] = not args.no_subset_reref
            rows.append(row)
            flag = "  <-- au-dessus du seuil" if (
                row["equilibree"] > row["seuil_eq"]) else ""
            print(f"  {model:18s} {scheme:15s} {row['contraste']:12s} "
                  f"{'recentre' if recenter else '        '} : "
                  f"equilibree {row['equilibree']:.4f} "
                  f"(hasard {row['hasard_eq']:.4f}, seuil {row['seuil_eq']:.4f})  "
                  f"p={row['p_perm']:.4f}{flag}", flush=True)
        # ecriture immediate : une coupure ne coute plus que le bloc en cours
        flush()
        print(f"    [{elapsed:.0f}s pour {len(block)} contrastes, ecrit dans "
              f"{out_path.name}]", flush=True)

    if failed:
        print(f"\n{'!' * 88}")
        print(f"{len(failed)} BLOC(S) EN ECHEC — aucune ligne ecrite pour eux, "
              f"une relance les reprendra :")
        for model, scheme, recenter, message in failed:
            print(f"  {model} / {scheme}" + ("  (recentre)" if recenter else "")
                  + f"\n      {message}")
        print("!" * 88)

    if stopped_early:
        print(f"\n{'!' * 88}")
        print(f"BUDGET DE {args.max_hours} h ATTEINT — {len(stopped_early)} blocs non calcules :")
        for model, scheme, recenter in stopped_early:
            print(f"  {model} / {scheme}" + ("  (recentre)" if recenter else ""))
        print("Relancer la MEME commande : les blocs deja faits seront sautes.")
        print("!" * 88)

    table = pd.DataFrame(rows)
    # Holm par contraste : c'est la famille de tests qui repond a une question
    table["p_holm"] = np.nan
    for _, block in table.groupby("contraste"):
        table.loc[block.index, "p_holm"] = holm_bonferroni(block["p_perm"].values)
    table["significatif_holm"] = table["p_holm"] < 0.05

    table.to_csv(args.out, index=False)

    print("\n" + "=" * 100)
    print("TABLEAU COMPLET")
    print("=" * 100)
    print(table.to_string(index=False))

    print("\n" + "=" * 100)
    print("LECTURE")
    print("=" * 100)
    survivors = table[table.significatif_holm]
    if len(survivors):
        print("Combinaisons survivant a la correction de Holm dans leur famille :")
        print(survivors.to_string(index=False))
    else:
        print("Aucune combinaison ne survit a la correction pour tests multiples.")

    print("\nPar contraste, meilleure exactitude equilibree obtenue :")
    for contrast, block in table.groupby("contraste"):
        best = block.loc[block.equilibree.idxmax()]
        verdict = ("AU-DESSUS du seuil" if best.equilibree > best.seuil_eq
                   else "sous le seuil de detectabilite")
        print(f"  {contrast:12s} {best.equilibree:.4f} "
              f"(hasard {best.hasard_eq:.4f}, seuil {best.seuil_eq:.4f}) "
              f"par {best.modele}/{best.schema}  -> {verdict}")

    flat = table[table.contraste == "mots15"]
    if len(flat) and flat.equilibree.max() < flat.seuil_eq.iloc[0]:
        print("\n  Sur les 15 mots, aucun modele n'atteint le seuil de detectabilite.")
        print("  Cela ne dit PAS 'aucun signal' : cela dit 'aucun signal au-dessus de")
        print(f"  {flat.seuil_eq.iloc[0]:.4f} en exactitude equilibree'. Avec {int(flat.n.iloc[0])} essais et")
        print("  15 classes, l'experience ne pouvait pas trancher en dessous de ce seuil.")

    def key(r):
        return f"{r.model}|{r.scheme}|{r.contrast}|{int(r.recentered)}|{args.montage}"

    np.savez_compressed(
        Path(args.out).with_suffix(".confusions.npz"),
        **{key(r): r.confusion for r in results},
    )
    if args.save_predictions:
        # Les predictions par essai sont la seule chose qui permette une
        # comparaison APPARIEE entre deux montages : ils sont evalues sur les
        # memes essais, donc comparer deux intervalles de confiance marginaux
        # ignore la correlation entre les deux et repond a une autre question.
        path = Path(args.out).with_suffix(".predictions.npz")
        store = dict(np.load(path, allow_pickle=True)) if path.exists() else {}
        for r in results:
            store[f"{key(r)}|vrai"] = np.asarray(r.y_true, dtype=object)
            store[f"{key(r)}|pred"] = np.asarray(r.y_pred, dtype=object)
        np.savez_compressed(path, **store)
        print(f"predictions par essai -> {path.name}")
    print(f"\nresultats -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
