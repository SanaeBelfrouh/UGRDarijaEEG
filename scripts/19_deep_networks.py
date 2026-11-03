#!/usr/bin/env python3
"""Etape 19 : deux reseaux convolutifs, EEGNet et ShallowConvNet, sur les 15 mots.

    python scripts/19_deep_networks.py --epochs epochs.npz --out results

Question d'un relecteur (Jose, p. 8) : les representations riemanniennes sont-elles
battues par les reseaux profonds de reference de l'EEG ? Deux architectures,
ecrites ici en PyTorch d'apres leurs articles pour ne dependre d'aucune version de
bibliotheque :

- EEGNet-8,2 (Lawhern et al., J. Neural Eng. 2018), noyau temporel d'une
  demi-seconde (125 echantillons a 250 Hz) ;
- ShallowConvNet (Schirrmeister et al., Hum. Brain Mapp. 2017), concu pour
  250 Hz : convolution temporelle de 25 echantillons, carre, moyenne glissante de
  75 echantillons (pas de 15), logarithme.

Protocole, le meme que le reste de l'article : 15 mots, fenetre 0-2 s apres le
go, 64 canaux de l'archive de reference (reference moyenne commune,
0,5-45 Hz), leave-one-session-out. Chaque canal est standardise par session, sans
etiquette (comme le recentrage). Entrainement par recadrage : a chaque passe, un
segment d'une seconde tire au hasard dans la fenetre de 2 s ; au test, la moyenne
des probabilites sur onze segments (pas de 100 ms). L'arret anticipe se decide sur
une session d'entrainement mise de cote (la suivante dans l'ordre), jamais sur la
session de test. AdamW, lots de 64, au plus ``--max-epochs`` passes, patience de
20 passes sur la perte de validation.

Les predictions de la configuration fixe de l'article (espace tangent du banc de
filtres, LDA a retrecissement, sans recentrage) sont recalculees sur les memes
essais, pour une comparaison appariee.

Sorties : ``reseaux.csv`` (exactitude equilibree, seuil de detectabilite, p par
permutation dans les sessions, Holm par condition) et ``reseaux_apparie.csv``
(reseau moins configuration fixe, permutation appariee, Holm par condition).
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from darija.evaluate import evaluate_many                                       # noqa: E402
from darija.features import MANUSCRIPT_BANK                                     # noqa: E402
from darija.pipelines import build_representations                              # noqa: E402
from darija.preprocess import crop                                              # noqa: E402
from darija.stats import (balanced_accuracy, holm_bonferroni,                   # noqa: E402
                          minimum_detectable_balanced_accuracy, permutation_pvalue)


# ----------------------------------------------------------------- reseaux
def build_networks():
    import torch
    from torch import nn

    class MaxNormLinear(nn.Linear):
        """Couche dense dont chaque vecteur de poids est borne en norme (EEGNet)."""

        def __init__(self, *a, max_norm=0.25, **k):
            super().__init__(*a, **k)
            self.max_norm = max_norm

        def forward(self, x):
            with torch.no_grad():
                self.weight.data = torch.renorm(self.weight.data, 2, 0, self.max_norm)
            return super().forward(x)

    class MaxNormConv(nn.Conv2d):
        def __init__(self, *a, max_norm=1.0, **k):
            super().__init__(*a, **k)
            self.max_norm = max_norm

        def forward(self, x):
            with torch.no_grad():
                self.weight.data = torch.renorm(self.weight.data, 2, 0, self.max_norm)
            return super().forward(x)

    class EEGNet(nn.Module):
        """EEGNet-8,2 : F1 = 8 filtres temporels, D = 2 filtres spatiaux par filtre."""

        def __init__(self, n_chans, n_times, n_classes, sfreq, f1=8, d=2, f2=16, dropout=0.5):
            super().__init__()
            k = int(sfreq // 2)
            self.features = nn.Sequential(
                nn.Conv2d(1, f1, (1, k), padding=(0, k // 2), bias=False),
                nn.BatchNorm2d(f1),
                MaxNormConv(f1, f1 * d, (n_chans, 1), groups=f1, bias=False, max_norm=1.0),
                nn.BatchNorm2d(f1 * d), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(dropout),
                nn.Conv2d(f1 * d, f1 * d, (1, 16), padding=(0, 8), groups=f1 * d, bias=False),
                nn.Conv2d(f1 * d, f2, 1, bias=False),
                nn.BatchNorm2d(f2), nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(dropout),
                nn.Flatten())
            with torch.no_grad():
                n = self.features(torch.zeros(1, 1, n_chans, n_times)).shape[1]
            self.classifier = MaxNormLinear(n, n_classes, max_norm=0.25)

        def forward(self, x):
            return self.classifier(self.features(x.unsqueeze(1)))

    class ShallowConvNet(nn.Module):
        """ShallowConvNet : filtrage temporel puis spatial, carre, moyenne, logarithme."""

        def __init__(self, n_chans, n_times, n_classes, n_filters=40, dropout=0.5):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, n_filters, (1, 25)),
                nn.Conv2d(n_filters, n_filters, (n_chans, 1), bias=False),
                nn.BatchNorm2d(n_filters))
            self.pool = nn.AvgPool2d((1, 75), stride=(1, 15))
            self.dropout = nn.Dropout(dropout)
            with torch.no_grad():
                n = self._body(torch.zeros(1, 1, n_chans, n_times)).shape[1]
            self.classifier = nn.Linear(n, n_classes)

        def _body(self, x):
            x = self.features(x)
            x = torch.log(torch.clamp(self.pool(x * x), min=1e-6))
            return self.dropout(x).flatten(1)

        def forward(self, x):
            return self.classifier(self._body(x.unsqueeze(1)))

    return {"eegnet": EEGNet, "shallowconvnet": ShallowConvNet}


LEARNING_RATES = {"eegnet": 1e-3, "shallowconvnet": 6.25e-4}


def standardise_by_session(X, ses):
    """Moyenne nulle et variance unite par canal et par session, sans etiquette."""
    out = np.empty_like(X, dtype=np.float32)
    for s in np.unique(ses):
        m = ses == s
        mu = X[m].mean(axis=(0, 2), keepdims=True)
        sd = X[m].std(axis=(0, 2), keepdims=True) + 1e-8
        out[m] = (X[m] - mu) / sd
    return out


def crop_starts(n_times, crop, n_test_crops):
    return np.linspace(0, n_times - crop, n_test_crops).round().astype(int)


def train_and_predict(name, X, y, train, val, test, sfreq, crop, args, seed, device):
    """Entraine sur train (arret anticipe sur val) et renvoie les probabilites sur test."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    nets = build_networks()
    n_classes = int(y.max()) + 1
    kwargs = {"sfreq": sfreq} if name == "eegnet" else {}
    model = nets[name](X.shape[1], crop, n_classes, **kwargs).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATES[name], weight_decay=0.0)
    loss_fn = nn.CrossEntropyLoss()
    Xt = torch.as_tensor(X, dtype=torch.float32)
    yt = torch.as_tensor(y, dtype=torch.long)
    starts = crop_starts(X.shape[-1], crop, args.test_crops)

    def predict(idx):
        model.eval()
        probs = np.zeros((len(idx), n_classes))
        with torch.no_grad():
            for b in range(0, len(idx), 256):
                xb = Xt[idx[b:b + 256]].to(device)
                p = sum(torch.softmax(model(xb[:, :, s0:s0 + crop]), 1) for s0 in starts) / len(starts)
                probs[b:b + 256] = p.cpu().numpy()
        return probs

    best, best_state, wait = np.inf, None, 0
    tr = np.flatnonzero(train)
    for epoch in range(args.max_epochs):
        model.train()
        order = rng.permutation(tr)
        for b in range(0, len(order), args.batch_size):
            idx = order[b:b + args.batch_size]
            if len(idx) < 2:
                continue
            offs = rng.integers(0, X.shape[-1] - crop + 1, len(idx))
            xb = torch.stack([Xt[i, :, o:o + crop] for i, o in zip(idx, offs)]).to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yt[idx].to(device))
            loss.backward()
            opt.step()
        vp = predict(np.flatnonzero(val))
        vloss = float(-np.mean(np.log(vp[np.arange(len(vp)), y[val]] + 1e-12)))
        if vloss < best - 1e-4:
            best, wait = vloss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= args.patience:
                break
    model.load_state_dict(best_state)
    return predict(np.flatnonzero(test)), epoch + 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", required=True, help="archive de reference")
    p.add_argument("--out", default="results")
    p.add_argument("--conditions", nargs="+", default=("imagined", "articulated"))
    p.add_argument("--models", nargs="+", default=("eegnet", "shallowconvnet"))
    p.add_argument("--crop", type=float, default=1.0, help="longueur des segments (s)")
    p.add_argument("--test-crops", type=int, default=11)
    p.add_argument("--max-epochs", type=int, default=150)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-perm", type=int, default=5000)
    p.add_argument("--device", default="auto", help="auto, cuda ou cpu")
    p.add_argument("--max-hours", type=float, default=None, help="ignore, pour run_all")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    try:
        import torch
    except ImportError:
        raise SystemExit("PyTorch absent : pip install torch (sur Kaggle, il est deja installe)")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    print(f"appareil : {device}" + ("" if device == "cuda" else " (lent : activer un GPU sur Kaggle)"),
          flush=True)

    spec = importlib.util.spec_from_file_location("pc", HERE / "05_paired_compare.py")
    pc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pc)

    a = np.load(args.epochs, allow_pickle=True)
    sf = float(a["sfreq"])
    crop_len = int(round(args.crop * sf))
    rows, pairs = [], []
    for condition in args.conditions:
        m = a["cond"].astype(str) == condition
        words = a["y"][m].astype(str)
        ses = a["session"][m].astype(str)
        tix = a["trial_index"][m].astype(int)
        X, _ = crop(a["X"][m].astype(np.float64), a["times"], 0.0, 2.0)
        labels = sorted(set(words))
        y = np.array([labels.index(w) for w in words])
        sessions = list(np.unique(ses))
        print(f"\n{condition} : {len(y)} essais, {X.shape[1]} canaux, {X.shape[-1]} echantillons",
              flush=True)

        # configuration fixe de l'article, sur les memes essais
        reps = build_representations(X, sf, bands=MANUSCRIPT_BANK, include=("bankcov",),
                                     n_components=32, groups=ses, rank_scope="global")
        ref = evaluate_many(reps, words, ses, tix, model="bankcov_tangent_lda", scheme="loso",
                            contrasts=("mots15",), recenter=False, n_perm=1)[0]
        if not np.array_equal(np.asarray(ref.y_true).astype(str), words):
            raise SystemExit("ordre des essais different entre reference et reseaux")
        ref_pred = np.asarray(ref.y_pred).astype(str)
        del reps
        Xs = standardise_by_session(X, ses)
        del X

        for name in args.models:
            t0 = time.time()
            proba = np.zeros((len(y), len(labels)))
            n_ep = []
            for k, s in enumerate(sessions):
                test = ses == s
                val = ses == sessions[(k + 1) % len(sessions)]
                train = ~test & ~val
                proba[test], e = train_and_predict(name, Xs, y, train, val, test, sf, crop_len,
                                                   args, seed=k, device=device)
                n_ep.append(e)
            pred = np.array(labels, dtype=object)[proba.argmax(1)]
            acc = balanced_accuracy(words, pred, labels)
            pv, _, null_mean = permutation_pvalue(words, pred, ses, n_perm=args.n_perm)
            counts = np.unique(words, return_counts=True)[1]
            rows.append(dict(condition=condition, modele=name, schema="loso", contraste="mots15",
                             essais=len(y), equilibree=acc, hasard_eq=1 / len(labels),
                             seuil_eq=minimum_detectable_balanced_accuracy(counts),
                             p_perm=pv, nulle_moy=null_mean, passes_moy=float(np.mean(n_ep))))
            diff, pp, pn = pc.paired_permutation(words, pred.astype(str), ref_pred, ses,
                                                 n_perm=args.n_perm)
            pairs.append(dict(condition=condition, configuration_a=name,
                              configuration_b="bankcov_tangent_lda", difference=diff,
                              p_apparie=pp, nulle_moy=pn))
            print(f"  {name:15s} {100 * acc:6.2f} % (p = {pv:.4f}) ; contre la configuration fixe "
                  f"({100 * ref.balanced_accuracy:.2f} %) {100 * diff:+.2f} points (p = {pp:.4f}) ; "
                  f"{np.mean(n_ep):.0f} passes en moyenne, {time.time() - t0:.0f} s", flush=True)

    t = pd.DataFrame(rows)
    pt = pd.DataFrame(pairs)
    for df, col in ((t, "p_perm"), (pt, "p_apparie")):
        df["p_holm"] = np.nan
        for c in df.condition.unique():
            sel = df.condition == c
            df.loc[sel, "p_holm"] = holm_bonferroni(df.loc[sel, col].values)
    t["significatif_holm"] = t.p_holm < 0.05
    t.to_csv(out / "reseaux.csv", index=False)
    pt.to_csv(out / "reseaux_apparie.csv", index=False)
    with pd.option_context("display.width", 200):
        print("\n" + t.round(4).to_string(index=False))
        print("\n" + pt.round(4).to_string(index=False))
    print(f"\nsorties : reseaux.csv, reseaux_apparie.csv dans {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
