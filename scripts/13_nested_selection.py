#!/usr/bin/env python3
"""Etape 13 : decodage imagine avec selection imbriquee de la configuration.

    python scripts/13_nested_selection.py --epochs epochs.npz --epochs-alt epochs_ica.npz \
        --out results

Le balayage des configurations (scripts/02) rapporte la meilleure d'entre elles,
et choisir la meilleure apres coup gonfle le score. Ici le choix fait partie de
la methode et se fait a l'interieur de chaque pli :

    pour chaque session de test s :
        pour chaque configuration c de la grille (fixee a l'avance) :
            score interieur(c) = leave-one-session-out sur les 7 autres sessions
        entrainer la meilleure sur les 7 sessions, predire s

L'exactitude mesure la procedure entiere, choix compris (Varma et Simon 2006 ;
Cawley et Talbot 2010).

Grille, fixee avant tout calcul :

- pretraitement : archive de reference (``--epochs``) et, si fournie, archive
  alternative (``--epochs-alt``, par exemple avec retrait oculaire) ;
- bandes : celles du manuscrit (1-8, 8-14, 14-26, 26-40 Hz), physiologiques
  (0,5-4, 4-8, 8-13, 13-30, 30-45 Hz), une bande large 0,5-45 Hz ;
- rang : 8, 16, 32 ;
- geometrie : recentree par session, ou stricte (aucune statistique de la
  session de test).

Classifieur : la LDA a retrecissement du manuscrit, resolue sous forme duale
pour la vitesse ; l'identite des predictions avec scikit-learn est verifiee au
debut.

Sorties : ``imbrique_resultats.csv`` (grille complete, ensemble des trois
meilleures, grille stricte, grille sous 30 Hz, pour les 15 mots),
``imbrique_apparie.csv`` (comparaison appariee avec la configuration
du manuscrit sur les memes essais), ``imbrique_grille.csv`` (chaque
configuration evaluee directement, pour transparence seulement),
``imbrique_choix_<contraste>.csv`` (configuration choisie dans chaque pli).
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from darija.contrasts import apply_contrast                                     # noqa: E402
from darija.evaluate import evaluate_many                                       # noqa: E402
from darija.features import MANUSCRIPT_BANK                                     # noqa: E402
from darija.pipelines import build_representations                              # noqa: E402
from darija.preprocess import bandpass, crop                                    # noqa: E402
from darija.riemann import covariances, recenter_by_group, tangent_vectors      # noqa: E402
from darija.stats import (balanced_accuracy,                                    # noqa: E402
                          minimum_detectable_balanced_accuracy, permutation_pvalue)

TOP_K = 3
WINDOW = (0.0, 2.0)
BANKS = {
    "manuscrit": ((1.0, 8.0), (8.0, 14.0), (14.0, 26.0), (26.0, 40.0)),
    "physiologique": ((0.5, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 45.0)),
    "large": ((0.5, 45.0),),
}
BANKS_LOW = {
    "manuscrit < 26 Hz": ((1.0, 8.0), (8.0, 14.0), (14.0, 26.0)),
    "physiologique < 30 Hz": ((0.5, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0)),
    "large < 30 Hz": ((0.5, 30.0),),
}
RANKS = (8, 16, 32)
MODES = ("recentre", "strict")


class ShrinkageLDA:
    """StandardScaler + LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
    resolue dans l'espace des essais (formule de Woodbury).

    Comme scikit-learn : chaque classe est standardisee par ses propres
    ecarts-types, sa covariance retrecie par Ledoit-Wolf (coefficient propre a
    la classe) puis remise a l'echelle, et les classes sont moyennees selon
    leurs frequences. La covariance s'ecrit A + Z^T D Z (A diagonale, Z les
    residus intra-classe).
    """

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        Xs = (X - self.mean_) / self.scale_
        self.classes_ = np.array(sorted(set(y.tolist())), dtype=object)
        idx = np.searchsorted(self.classes_, y)
        n, d = Xs.shape
        K = len(self.classes_)
        priors = np.bincount(idx, minlength=K) / n
        M = np.vstack([Xs[idx == k].mean(axis=0) for k in range(K)])
        Z = Xs - M[idx]
        A = np.zeros(d)
        Dw = np.empty(n)
        for k in range(K):
            rows = idx == k
            nk = rows.sum()
            sk = Z[rows].std(axis=0)
            sk[sk == 0] = 1.0
            Zh = Z[rows] / sk
            G = Zh @ Zh.T
            mu = np.trace(G) / (nk * d)
            g2 = np.sum(G ** 2)
            beta = (np.sum(np.diag(G) ** 2) / nk - g2 / nk ** 2) / (d * nk)
            delta = (g2 / nk ** 2 - d * mu ** 2) / d
            beta = min(beta, delta)
            lam = 0.0 if beta == 0 else beta / delta
            A += priors[k] * lam * mu * sk ** 2
            Dw[rows] = priors[k] * (1.0 - lam) / nk
        Dh = np.sqrt(Dw)[:, None]
        B = M.T / A[:, None]
        U = Dh * Z
        C = np.eye(n) + (U / A) @ U.T
        W = B - (U.T / A[:, None]) @ np.linalg.solve(C, U @ B)
        self.coef_ = W.T
        self.intercept_ = -0.5 * np.sum(M * self.coef_, axis=1) + np.log(priors)
        return self

    def decision_function(self, X):
        Xs = (np.asarray(X, dtype=np.float64) - self.mean_) / self.scale_
        return Xs @ self.coef_.T + self.intercept_

    def predict(self, X):
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]

    def predict_proba(self, X):
        s = self.decision_function(X)
        s -= s.max(axis=1, keepdims=True)
        e = np.exp(s)
        return e / e.sum(axis=1, keepdims=True)


def features(X, times, ses, banks):
    """Toutes les configurations d'une archive, calculees sans etiquettes."""
    Xc, _ = crop(X, times, *WINDOW)
    _, V = np.linalg.eigh(covariances(Xc).mean(axis=0))
    P = V[:, -max(RANKS):]                          # base commune aux sessions
    Xp = np.einsum("kc,nct->nkt", P.T, X)
    del Xc
    out = {}
    for bank_name, bands in banks.items():
        filtered = [crop(bandpass(Xp, 250.0, lo, hi), times, *WINDOW)[0] for lo, hi in bands]
        for rank in RANKS:
            covs = [covariances(fb[:, -rank:]) for fb in filtered]
            for mode in MODES:
                if mode == "recentre":
                    vec = [tangent_vectors(recenter_by_group(c, ses), np.eye(rank)) for c in covs]
                else:
                    vec = [tangent_vectors(c, np.eye(rank)) for c in covs]
                out[(bank_name, rank, mode)] = np.hstack(vec).astype(np.float32)
        del filtered
    return out


class Nested:
    def __init__(self):
        self.cache = {}

    def fold_predictions(self, key, F, labels, groups):
        if key in self.cache:
            return self.cache[key]
        sessions = sorted(set(groups))
        inner = {}
        for a, b in itertools.combinations(sessions, 2):
            train = (groups != a) & (groups != b)
            m = ShrinkageLDA().fit(F[train], labels[train])
            inner[(a, b)] = m.predict(F[groups == b])
            inner[(b, a)] = m.predict(F[groups == a])
        outer_pred = np.empty(len(labels), dtype=object)
        outer_proba = np.zeros((len(labels), len(set(labels))))
        for s in sessions:
            test = groups == s
            m = ShrinkageLDA().fit(F[~test], labels[~test])
            outer_pred[test] = m.predict(F[test])
            outer_proba[test] = m.predict_proba(F[test])
        self.cache[key] = (inner, outer_pred, outer_proba)
        return self.cache[key]

    def run(self, feats, labels, groups, contrast, label):
        sessions = sorted(set(groups))
        classes = sorted(set(labels))
        configs = list(feats)
        t0 = time.time()
        for k, c in enumerate(configs):
            self.fold_predictions((contrast,) + c, feats[c], labels, groups)
            if k % 6 == 5:
                print(f"  {label} : {k + 1}/{len(configs)} configurations ({time.time() - t0:.0f} s)",
                      flush=True)
        best = np.empty(len(labels), dtype=object)
        proba = np.zeros((len(labels), len(classes)))
        outer = {c: self.cache[(contrast,) + c][1] for c in configs}
        rows = []
        for s in sessions:
            test = groups == s
            yt = np.concatenate([labels[groups == i] for i in sessions if i != s])
            scores = {}
            for c in configs:
                inner = self.cache[(contrast,) + c][0]
                yp = np.concatenate([inner[(s, i)] for i in sessions if i != s])
                scores[c] = balanced_accuracy(yt, yp, classes)
            ranking = sorted(configs, key=lambda c: -scores[c])
            best[test] = outer[ranking[0]][test]
            for c in ranking[:TOP_K]:
                proba[test] += self.cache[(contrast,) + c][2][test] / TOP_K
            rows.append({"session_test": s, "configuration": " | ".join(map(str, ranking[0])),
                         "score_interieur": scores[ranking[0]],
                         "equilibree_test": balanced_accuracy(labels[test], best[test], classes)})
        ensemble = np.asarray(classes, dtype=object)[proba.argmax(axis=1)]
        return dict(best=best, ensemble=ensemble, choices=pd.DataFrame(rows), outer=outer)


def score(labels, pred, groups, name, contrast, n_perm):
    classes = sorted(set(labels))
    acc = balanced_accuracy(labels, pred, classes)
    p, _, _ = permutation_pvalue(labels, pred, groups, n_perm=n_perm)
    return {"contraste": contrast, "procedure": name, "essais": len(labels), "equilibree": acc,
            "p": p, "hasard": 1 / len(classes),
            "seuil_eq": minimum_detectable_balanced_accuracy([int(np.sum(labels == c)) for c in classes])}


def paired_p(yt, pa, pb, n_perm, seed=0):
    labels = sorted(set(yt))
    obs = balanced_accuracy(yt, pb, labels) - balanced_accuracy(yt, pa, labels)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for k in range(n_perm):
        swap = rng.random(len(yt)) < 0.5
        null[k] = balanced_accuracy(yt, np.where(swap, pa, pb), labels) - \
            balanced_accuracy(yt, np.where(swap, pb, pa), labels)
    return obs, (np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1)


def main() -> int:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", required=True, help="archive de reference")
    p.add_argument("--epochs-alt", default=None, help="archive alternative (pretraitement)")
    p.add_argument("--out", default="results")
    p.add_argument("--n-perm", type=int, default=5000)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    archives = {"reference": Path(args.epochs)}
    if args.epochs_alt and Path(args.epochs_alt).exists():
        archives["alternative"] = Path(args.epochs_alt)
    loaded, keys = {}, None
    for name, path in archives.items():
        a = np.load(path, allow_pickle=True)
        m = a["cond"].astype(str) == "imagined"
        k = pd.DataFrame({"session": a["session"][m].astype(str),
                          "trial_index": a["trial_index"][m].astype(int),
                          "word": a["y"][m].astype(str)})
        k[f"pos_{name}"] = np.flatnonzero(m)
        loaded[name] = a
        keys = k if keys is None else keys.merge(k, on=["session", "trial_index", "word"])
        print(f"{name} : {path.name}, {int(m.sum())} essais imagines", flush=True)
    keys = keys.sort_values(["session", "trial_index"]).reset_index(drop=True)
    y = keys.word.values.astype(str)
    ses = keys.session.values.astype(str)
    tix = keys.trial_index.values.astype(int)
    print(f"essais communs aux archives : {len(y)}", flush=True)

    feats, feats_low = {}, {}
    t0 = time.time()
    for name, a in loaded.items():
        X = a["X"][keys[f"pos_{name}"].values].astype(np.float64)
        times = a["times"]
        for (bank, rank, mode), v in features(X, times, ses, BANKS).items():
            feats[(name, bank, rank, mode)] = v
        for (bank, rank, mode), v in features(X, times, ses, BANKS_LOW).items():
            feats_low[(name, bank, rank, mode)] = v
        del X
        print(f"{name} : caracteristiques pretes ({time.time() - t0:.0f} s)", flush=True)

    # la LDA duale doit reproduire celle de scikit-learn, prediction par prediction
    c0 = next(iter(feats))
    test = ses == sorted(set(ses))[0]
    fast = ShrinkageLDA().fit(feats[c0][~test], y[~test]).predict(feats[c0][test])
    ref = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")) \
        .fit(feats[c0][~test], y[~test]).predict(feats[c0][test])
    agree = float(np.mean(fast == ref))
    print(f"LDA duale contre scikit-learn : {100 * agree:.1f} % de predictions identiques", flush=True)
    if agree < 0.99:
        raise SystemExit("la LDA duale ne reproduit pas scikit-learn : resultats non comparables")

    # configurations du manuscrit, sur les memes essais
    a = loaded["reference"]
    Xr, _ = crop(a["X"][keys["pos_reference"].values].astype(np.float64), a["times"], *WINDOW)
    reps = build_representations(Xr, float(a["sfreq"]), bands=MANUSCRIPT_BANK,
                                 include=("bankcov",), n_components=32, groups=ses,
                                 rank_scope="global")
    del Xr
    published = {
        "mots15": evaluate_many(reps, y, ses, tix, model="bankcov_tangent_lda", contrasts=("mots15",),
                                recenter=False, n_perm=args.n_perm)[0],
    }

    nested = Nested()
    results, paired, grid = [], [], []
    for contrast in ("mots15",):
        labels, mask, _ = apply_contrast(contrast, y)
        labels = np.asarray(labels, dtype=object)
        g = ses[mask]
        print(f"\n=== {contrast} ===", flush=True)
        full = nested.run({k: v[mask] for k, v in feats.items()}, labels, g, contrast, contrast)
        strict = nested.run({k: v[mask] for k, v in feats.items() if k[3] == "strict"}, labels, g,
                            contrast, contrast + " strict")
        low = nested.run({k: v[mask] for k, v in feats_low.items()}, labels, g, contrast,
                         contrast + " < 30 Hz")
        full["choices"].to_csv(out / f"imbrique_choix_{contrast}.csv", index=False)
        variants = {"selection imbriquee": full["best"],
                    f"selection imbriquee, ensemble des {TOP_K} meilleures": full["ensemble"],
                    "selection imbriquee, grille stricte": strict["best"],
                    "selection imbriquee, sous 30 Hz": low["best"]}
        pub = published[contrast]
        if not np.array_equal(np.asarray(pub.y_true, dtype=object), labels):
            raise SystemExit("ordre des essais different entre configuration publiee et selection imbriquee")
        results.append(score(labels, np.asarray(pub.y_pred, dtype=object), g,
                             "configuration du manuscrit", contrast, args.n_perm))
        for name, pred in variants.items():
            r = score(labels, pred, g, name, contrast, args.n_perm)
            results.append(r)
            d, pp = paired_p(labels, np.asarray(pub.y_pred, dtype=object), pred, args.n_perm)
            paired.append({"contraste": contrast, "procedure": name, "equilibree": r["equilibree"],
                           "difference_points": 100 * d, "p_apparie": pp})
            print(f"  {name:52s} {100 * r['equilibree']:6.2f} %  p = {r['p']:.4f}  "
                  f"(manuscrit {100 * pub.balanced_accuracy:.2f} %)", flush=True)
        for c, pred in full["outer"].items():
            grid.append({"contraste": contrast, "pretraitement": c[0], "bandes": c[1], "rang": c[2],
                         "geometrie": c[3],
                         "equilibree": balanced_accuracy(labels, pred, sorted(set(labels)))})
    pd.DataFrame(results).to_csv(out / "imbrique_resultats.csv", index=False)
    pd.DataFrame(paired).to_csv(out / "imbrique_apparie.csv", index=False)
    pd.DataFrame(grid).to_csv(out / "imbrique_grille.csv", index=False)
    print(f"\nsorties : imbrique_resultats.csv, imbrique_apparie.csv, imbrique_grille.csv dans {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
