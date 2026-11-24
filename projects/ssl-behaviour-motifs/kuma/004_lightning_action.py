"""004 (Kuma, GPU) -- lightning-action (paninski-lab, supervised) trained on the HEURISTIC labels of 002, per view.

v2: 200 Hz, classes background / still / whisking / nose (snout movement) / jaw / lick, sequences of 1000 frames (5 s),
30 epochs. F1 against the heuristic labels is circular (the labels come from the same keypoints / features the model
sees): it measures how well the classifier reproduces the heuristic, not behavioural ground truth.
Inputs per session x view (200 Hz grid of that camera): aligned keypoints x, y (002, centred per session) and their frame-to-frame velocities,
written as lightning-action `features/<expt>.csv`; labels = one-hot `labels/<expt>.csv` (columns background, still,
whisking, jaw, lick; background = class 0 = ignore_index, i.e. not scored). Labels live on the side grid and are mapped to
the top grid by nearest time.
Backbone (head) dtcn = dilated TCN. v3: leave-one-mouse-out (8 folds, one session per mouse; 10 % of the training
sequences for validation); inputs = moving keypoints in mm minus the session's resting pose (002 v3) and velocities;
labels from POOLED thresholds in physical units (tables/label_thresholds.json). Outputs per held-out session x view: class probabilities (npz) and per-class F1 vs
heuristic labels; results/lightning_action/<view>/.
Usage: python 004_lightning_action.py <view> [epochs]
"""
import json
import sys
from pathlib import Path

import os

import numpy as np
import pandas as pd
import yaml

BASE = Path("/work/lsens/bisi/behaviour_motifs")
SET = os.environ.get("BM_SET", "pilot")
INP = BASE / "data" / ("inputs" if SET == "pilot" else "full_inputs")
RES_ROOT = BASE / "results" / ("" if SET == "pilot" else "full")
# full run: the nose class is dropped (pilot leave-one-mouse-out F1 0.24 side / 0.42 top: not separable) -> background
CLASSES = ["background", "still", "whisking", "nose", "jaw", "lick"] if SET == "pilot" else ["background", "still", "whisking", "jaw", "lick"]
LABEL_MAP = None if SET == "pilot" else np.array([0, 1, 2, 0, 3, 4])  # 002 codes -> full-run classes
NFOLD_FULL = 5


def write_expt(root, sid, view):
    k = np.load(INP / f"{sid}_{view}_keypoints.npz")
    s = np.load(INP / f"{sid}_session.npz")
    xy = k["xy"].reshape(len(k["t"]), -1)
    xy = xy - k["rest"].reshape(1, -1)  # deviation from the session's resting pose (mm), as kpms 003 norm=rest
    vel = np.vstack([np.zeros((1, xy.shape[1])), np.diff(xy, axis=0)])
    names = [f"{b}_{c}" for b in k["bodyparts"] for c in "xy"]
    X = pd.DataFrame(np.hstack([xy, vel]), columns=names + [f"{n}_vel" for n in names])
    j = np.clip(np.searchsorted(s["t"], k["t"]), 0, len(s["t"]) - 1)
    lab = s["labels"][j].astype(int)
    if LABEL_MAP is not None:
        lab = LABEL_MAP[lab]
    lab[np.abs(s["t"][j] - k["t"]) > 0.0075] = 0
    X.astype(np.float32).to_csv(root / "features" / f"{sid}.csv", float_format="%.3f")
    pd.DataFrame(np.eye(len(CLASSES), dtype=int)[lab], columns=CLASSES).to_csv(root / "labels" / f"{sid}.csv")
    return X.shape[1], lab


def f1_table(lab, pred):
    rows = []
    m = lab > 0
    for c in range(1, len(CLASSES)):
        tp = np.sum((pred == c) & (lab == c) & m)
        fp = np.sum((pred == c) & (lab != c) & m)
        fn = np.sum((pred != c) & (lab == c) & m)
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        rows.append(dict(cls=CLASSES[c], precision=p, recall=r, f1=2 * p * r / max(p + r, 1e-9), n_label=int((lab == c).sum()),
                         frac_pred_on_background=float(np.mean(pred[~m] == c)) if (~m).any() else np.nan))
    return rows


def main():
    """python 004_lightning_action.py <view> <epochs> prep   -> writes the feature / label CSVs once
       python 004_lightning_action.py <view> <epochs> <fold> -> trains fold <fold> (held-out mouse), parallel jobs"""
    view = sys.argv[1]
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    which = sys.argv[3] if len(sys.argv) > 3 else "all"
    from lightning_action.api import Model
    if SET == "pilot":
        sel = pd.read_csv(BASE / "data" / "001_selected_sessions.csv")
        folds = [[s] for s in sel.session_id]  # leave-one-mouse-out
    else:  # full: 5 folds grouped by mouse (one session per mouse), R+ / R- interleaved; plus a final all-data model
        sel = pd.read_csv(BASE / "data" / "full_sessions.csv").sort_values(["reward_group", "session_id"])
        order = list(sel[sel.reward_group == "R+"].session_id) + list(sel[sel.reward_group == "R-"].session_id)
        folds = [order[i::NFOLD_FULL] for i in range(NFOLD_FULL)]
        if which == "final":
            folds = [[]]
    out = RES_ROOT / "lightning_action" / view
    root = BASE / ("la_data" if SET == "pilot" else "la_data_full") / view
    for d in ("features", "labels"):
        (root / d).mkdir(parents=True, exist_ok=True)
    labs, nin = {}, None
    if which in ("prep", "all"):
        for sid in sel.session_id:
            nin, labs[sid] = write_expt(root, sid, view)
        json.dump(dict(nin=nin), open(root / "nin.json", "w"))
        if which == "prep":
            return
    nin = json.load(open(root / "nin.json"))["nin"]
    for sid in sel.session_id:
        labs[sid] = pd.read_csv(root / "labels" / f"{sid}.csv", index_col=0).to_numpy().argmax(1)
    rows = []
    for fi, test in enumerate(folds):
        if which not in ("all", "final") and fi != int(which):
            continue
        train = [s for s in sel.session_id if s not in test]
        cfg = dict(
            data=dict(data_path=str(root), input_dir="features", expt_ids=train, seed=42, ignore_index=0, weight_classes=True,
                      transforms=["ZScore"]),
            model=dict(input_size=nin, output_size=len(CLASSES), head="dtcn", num_hid_units=64, num_layers=2, num_lags=8, seed=42),
            optimizer=dict(type="Adam", lr=1e-3, wd=1e-4, scheduler="cosine", T_max=epochs),
            training=dict(device="gpu", num_epochs=epochs, log_every_n_steps=10, check_val_every_n_epoch=5, sequence_length=1000,
                          batch_size=32, num_workers=4, train_probability=0.9, val_probability=0.1, seed=42))
        fdir = out / ("final" if which == "final" else f"fold{fi}")
        fdir.mkdir(parents=True, exist_ok=True)
        yaml.safe_dump(cfg, open(fdir / "config.yaml", "w"))
        model = Model.from_config(str(fdir / "config.yaml"))
        model.train(output_dir=str(fdir / "run"))
        if not test:
            continue
        model.predict(data_path=str(root), input_dir="features", output_dir=str(fdir / "pred"), expt_ids=test)
        for sid in test:
            pf = list((fdir / "pred").glob(f"{sid}*.csv"))[0]
            P = pd.read_csv(pf, index_col=0).to_numpy(float)
            pred = np.nanargmax(np.nan_to_num(P, nan=-1), 1)
            np.savez_compressed(out / f"{sid}_probs.npz", probs=P.astype(np.float32), pred=pred, labels=labs[sid], fold=fi)
            for r in f1_table(labs[sid], pred):
                rows.append(dict(session_id=sid, view=view, fold=fi, **r))
    pd.DataFrame(rows).to_csv(out / ("f1_heldout.csv" if which == "all" else f"f1_heldout_fold{which}.csv"), index=False)
    json.dump(dict(view=view, epochs=epochs, folds=folds, classes=CLASSES, head="dtcn", inputs="moving keypoints (mm, minus resting pose) + velocity, z-scored",
                   validation="leave-one-mouse-out", labels="pooled thresholds, tables/label_thresholds.json"),
              open(out / "provenance.json", "w"), indent=1)
    if not rows:
        return
    print(pd.DataFrame(rows).groupby("cls")[["precision", "recall", "f1"]].mean())


if __name__ == "__main__":
    main()
