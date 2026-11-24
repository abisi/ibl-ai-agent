"""034 (Kuma, GPU) -- action segmentation comparison on the 8 pilot sessions, side view (2026-11-22).

Same classes (background = unlabelled / ignored, still, whisking, jaw, lick) and the same threshold (heuristic) labels
as the full lightning-action run (la_data_full/side/labels, pooled thresholds); leave-one-mouse-out over the 8 pilot
sessions (one session per mouse). Inputs (one row per side DLC frame = video frame):
  kp      keypoints minus resting pose + velocities (the lightning-action features of 004)
  fm      Facemap motion SVD, first 50 components (033, whole video)
  beast   BEAST CLS embedding (032), PCA to 64 dims (PCA fit on a frame subsample of all 8 sessions; no labels used)
  kp_fm   kp + fm ;  kp_beast  kp + beast
Heads:
  la            lightning-action dilated TCN, supervised on the heuristic labels (as 004)
  daart_pred    daart dTCN, weak (heuristic) labels + next-step prediction loss (Whiteway et al. 2021 semi-supervision)
  daart_nopred  same without the prediction loss (isolates the self-supervised term)
Outputs: results/full/segcompare/<head>_<input>/<sid>_probs.npz (probs per frame, fold); evaluation against independent
signals (piezo licks, jaw onsets, top-camera whisker / nose movement) is done on haas (035).
Run: python 034_segment_compare.py prep | python 034_segment_compare.py fit <head> <input> <fold>
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("/work/lsens/bisi/behaviour_motifs")
LA = BASE / "la_data_full" / "side"
EXT = BASE / "ext_data"
ROOT = EXT / "segcompare"
OUT = BASE / "results" / "full" / "segcompare"
PILOT = ["AB117_20240723_125437", "AB121_20240813_125401", "AB137_20241116_134442", "AB139_20241119_115957",
         "MH026_20250422_181339", "MH031_20250507_104425", "MH037_20250524_143522", "MH039_20250525_112720"]
CLASSES = ["background", "still", "whisking", "jaw", "lick"]
INPUTS = ["kp", "fm", "beast", "kp_fm", "kp_beast"]
EPOCHS = 30
N_FM, N_BEAST = 50, 64


def fit_to(x, n):
    """rows of x mapped to n frames (video and DLC frame counts may differ by a few frames at the end)"""
    if len(x) >= n:
        return x[:n]
    return np.vstack([x, np.repeat(x[-1:], n - len(x), 0)])


def prep():
    from sklearn.decomposition import PCA
    base = {s: pd.read_csv(LA / "features" / f"{s}.csv", index_col=0).to_numpy(np.float32) for s in PILOT}
    n = {s: len(v) for s, v in base.items()}
    have_fm = all((EXT / "facemap" / "side" / f"{s}_side_facemap.npz").exists() for s in PILOT)
    have_be = all((EXT / "beast" / "side" / "embeddings" / f"{s}_side.npy").exists() for s in PILOT)
    sets = dict(kp=base)
    if have_fm:
        fm = {}
        for s in PILOT:
            d = np.load(EXT / "facemap" / "side" / f"{s}_side_facemap.npz")
            fm[s] = fit_to(d["motSVD"][:, :N_FM].astype(np.float32), n[s])
            print(s, "frames kp", n[s], "facemap", d["motSVD"].shape[0], flush=True)
        sets.update(fm=fm, kp_fm={s: np.hstack([base[s], fm[s]]) for s in PILOT})
    if have_be:
        emb = {s: np.load(EXT / "beast" / "side" / "embeddings" / f"{s}_side.npy", mmap_mode="r") for s in PILOT}
        rng = np.random.default_rng(0)
        sub = np.vstack([np.asarray(emb[s][np.sort(rng.choice(len(emb[s]), 20000, replace=False))], np.float32) for s in PILOT])
        pca = PCA(N_BEAST, random_state=0).fit(sub)
        be = {}
        for s in PILOT:
            z = np.vstack([pca.transform(np.asarray(emb[s][i:i + 100000], np.float32)) for i in range(0, len(emb[s]), 100000)])
            be[s] = fit_to(z.astype(np.float32), n[s])
            print(s, "beast", len(emb[s]), "-> pca", be[s].shape, flush=True)
            np.save(EXT / "beast" / "side" / "embeddings" / f"{s}_side_pca{N_BEAST}.npy", be[s])
        json.dump(dict(explained=pca.explained_variance_ratio_.tolist()), open(EXT / "beast" / "side" / "pca.json", "w"))
        sets.update(beast=be, kp_beast={s: np.hstack([base[s], be[s]]) for s in PILOT})
    only = sys.argv[2:]  # optional subset of input names
    sets = {k: v for k, v in sets.items() if not only or k in only}
    for name, X in sets.items():
        d = ROOT / name
        for sub_ in ("features", "labels", "labels-heuristic"):
            (d / sub_).mkdir(parents=True, exist_ok=True)
        for s in PILOT:
            pd.DataFrame(X[s]).to_csv(d / "features" / f"{s}.csv", float_format="%.4g")  # lightning-action
            np.save(d / "features" / f"{s}.npy", X[s])  # daart
            for sub_ in ("labels", "labels-heuristic"):
                dst = d / sub_ / f"{s}.csv"
                if not dst.exists():
                    dst.symlink_to(LA / "labels" / f"{s}.csv")
        json.dump(dict(nin=int(X[PILOT[0]].shape[1])), open(d / "nin.json", "w"))
        print(name, X[PILOT[0]].shape, flush=True)


def fit_la(inp, fold, out):
    import yaml
    from lightning_action.api import Model
    root = ROOT / inp
    nin = json.load(open(root / "nin.json"))["nin"]
    test = [PILOT[fold]]
    train = [s for s in PILOT if s not in test]
    cfg = dict(
        data=dict(data_path=str(root), input_dir="features", expt_ids=train, seed=42, ignore_index=0, weight_classes=True,
                  transforms=["ZScore"]),
        model=dict(input_size=nin, output_size=len(CLASSES), head="dtcn", num_hid_units=64, num_layers=2, num_lags=8, seed=42),
        optimizer=dict(type="Adam", lr=1e-3, wd=1e-4, scheduler="cosine", T_max=EPOCHS),
        training=dict(device="gpu", num_epochs=EPOCHS, log_every_n_steps=10, check_val_every_n_epoch=5, sequence_length=1000,
                      batch_size=32, num_workers=4, train_probability=0.9, val_probability=0.1, seed=42))
    fdir = out / f"fold{fold}"
    fdir.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(cfg, open(fdir / "config.yaml", "w"))
    model = Model.from_config(str(fdir / "config.yaml"))
    model.train(output_dir=str(fdir / "run"))
    model.predict(data_path=str(root), input_dir="features", output_dir=str(fdir / "pred"), expt_ids=test)
    P = pd.read_csv(list((fdir / "pred").glob(f"{test[0]}*.csv"))[0], index_col=0).to_numpy(float)
    return test[0], P


def fit_daart(inp, fold, out, lambda_pred):
    import torch
    from daart.models import Segmenter
    from daart.train import Trainer
    from daart.utils import build_data_generator
    root = ROOT / inp
    nin = json.load(open(root / "nin.json"))["nin"]
    test = PILOT[fold]
    train = [s for s in PILOT if s != test]
    fdir = out / f"fold{fold}"
    fdir.mkdir(parents=True, exist_ok=True)
    hp = dict(data_dir=str(root), input_type="features", expt_ids=train, ignore_class=0, weight_classes=True,
              model_class="segmenter", backbone="dtcn", n_hid_layers=2, n_hid_units=64, n_lags=8, activation="lrelu",
              bidirectional=True, dropout=0.1, classifier_type="multiclass", lambda_weak=1.0, lambda_strong=0.0,
              lambda_recon=0.0, lambda_pred=lambda_pred, lambda_task=0.0, rng_seed_model=0, variational=False,
              semi_supervised_algo=None, input_size=nin, output_size=len(CLASSES), learning_rate=1e-4, sequence_length=1000,
              batch_size=32, l2_reg=0.0, min_epochs=10, max_epochs=EPOCHS * 4, val_check_interval=1, enable_early_stop=True,
              early_stop_history=5, save_last_model=False, train_frac=1.0, trial_splits="9;1;0;0", rng_seed_train=0,
              device="cuda", tt_version_dir=str(fdir), plot_train_curves=False)
    gen = build_data_generator(hp)
    # daart counts classes from hand labels only; here: the heuristic labels of the training sessions
    totals = sum(pd.read_csv(root / "labels-heuristic" / f"{s}.csv", index_col=0).to_numpy().sum(0) for s in train).astype(float)
    totals[0] = 0
    w = np.max(totals) / (totals + 1e-10)
    w[totals == 0] = 0
    hp["class_weights"] = w.tolist()
    torch.manual_seed(0)
    model = Segmenter(hp).to("cuda")
    Trainer(**hp, callbacks=[]).fit(model, gen, save_path=str(fdir))
    hp_test = dict(hp, expt_ids=[test], trial_splits="1;0;0;0")
    gtest = build_data_generator(hp_test)
    res = model.predict_labels(gtest)
    P = np.vstack([x for x in res["labels"][0] if len(x)])  # softmax of the label head
    return test, P


def main():
    stage = sys.argv[1]
    if stage == "prep":
        prep()
        return
    head, inp, fold = sys.argv[2], sys.argv[3], int(sys.argv[4])
    out = OUT / f"{head}_{inp}"
    out.mkdir(parents=True, exist_ok=True)
    if head == "la":
        sid, P = fit_la(inp, fold, out)
    else:
        sid, P = fit_daart(inp, fold, out, 1.0 if head == "daart_pred" else 0.0)
    lab = pd.read_csv(LA / "labels" / f"{sid}.csv", index_col=0).to_numpy().argmax(1)
    if len(P) < len(lab):  # daart drops the last partial sequence: pad with NaN (lightning-action pads itself)
        P = np.vstack([P, np.full((len(lab) - len(P), P.shape[1]), np.nan)])
    pred = np.nanargmax(np.nan_to_num(P, nan=-1), 1)
    np.savez_compressed(out / f"{sid}_probs.npz", probs=P.astype(np.float32), pred=pred, labels=lab, fold=fold, head=head, input=inp)
    m = lab > 0
    print(head, inp, sid, "agreement with heuristic labels (circular)", float(np.mean(pred[m] == lab[m])), flush=True)


if __name__ == "__main__":
    main()
