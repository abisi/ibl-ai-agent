"""Merged DLC model (training_plan.md, approved 2026-11-09): shared 90/10 frame split and one PyTorch shuffle per
architecture, in the Kuma copy of Tracking2D-Axel-2026-11-04 (DLC 3.0.2, venv_dlc).

1. config.yaml of the Kuma copy: project_path -> Kuma path (the NAS original is not touched).
2. Checks every labelled frame is 640 x 480 (no crop sampling: whole frames, batches need one size).
3. Shuffle 1 (resnet_50) with DLC's random split at TrainingFraction 0.9; shuffles 2-7 copy that split
   (create_training_dataset_from_existing_split).
4. pytorch_config.yaml of every shuffle: 500 epochs, batch 8, test evaluation and snapshot every 10 epochs (all kept),
   augmentation as agreed; optimizer and scheduler left at DLC's defaults for the architecture.
5. Writes split.csv (frame, train/test) and setup_summary.json next to config.yaml.

Run on Kuma: /work/lsens/bisi/dlc_project/venv_dlc/bin/python 010_merged_training_setup.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image

PROJ = Path("/work/lsens/bisi/dlc_merged/Tracking2D-Axel-2026-11-04")
CFG = PROJ / "config.yaml"
SHUFFLES = {1: "resnet_50", 2: "resnet_101", 3: "hrnet_w18", 4: "hrnet_w32", 5: "hrnet_w48", 6: "cspnext_m"}
# dlcrnet_stride16_ms5 dropped: DLC 3.0.2 requires part-affinity-field parameters (multi-animal skeleton) for DLCRNet
EPOCHS, EVERY, BATCH = 500, 10, 8
AUGMENT = {"affine": {"p": 0.5, "rotation": 15, "scaling": [0.8, 1.2], "translation": 30},
           "covering": True, "motion_blur": True, "gaussian_noise": 12.75, "hist_eq": True, "normalize_images": True}


def set_project_path():
    text = CFG.read_text()
    # video_sets keys are Windows UNC paths: forward slashes so DLC takes each video's stem (= labeled-data folder)
    lines = [f"project_path: {PROJ}" if l.startswith("project_path:") else
             l.replace("\\", "/") if l.startswith("  ? ") else l for l in text.splitlines()]
    CFG.write_text("\n".join(lines) + "\n")


def check_sizes():
    sizes = {}
    for p in (PROJ / "labeled-data").glob("*/img*.png"):
        with Image.open(p) as im:
            sizes.setdefault(im.size, []).append(str(p.relative_to(PROJ)))
    print("frame sizes:", {k: len(v) for k, v in sizes.items()})
    return sizes


def patch_pytorch_config(path):
    c = yaml.safe_load(path.read_text())
    c["train_settings"].update({"epochs": EPOCHS, "batch_size": BATCH, "dataloader_workers": 8,
                                "dataloader_pin_memory": True, "seed": 42})
    c["runner"]["eval_interval"] = EVERY
    c["runner"]["snapshots"].update({"save_epochs": EVERY, "max_snapshots": EPOCHS // EVERY + 5})
    train = c["data"]["train"]
    train.pop("crop_sampling", None)
    train.pop("hflip", None)
    train.update(AUGMENT)
    path.write_text(yaml.safe_dump(c, sort_keys=False))
    return {"optimizer": c["runner"]["optimizer"], "scheduler": c["runner"]["scheduler"], "data_train": train}


def main():
    import deeplabcut
    from deeplabcut.core.engine import Engine
    set_project_path()
    sizes = check_sizes()
    if len(sizes) != 1:
        raise SystemExit(f"frames of several sizes {list(sizes)}: crop sampling or resize needed, stop")
    deeplabcut.create_training_dataset(str(CFG), Shuffles=[1], net_type=SHUFFLES[1], engine=Engine.PYTORCH,
                                       userfeedback=False)
    for s, net in list(SHUFFLES.items())[1:]:
        deeplabcut.create_training_dataset_from_existing_split(str(CFG), from_shuffle=1, shuffles=[s], net_type=net,
                                                               userfeedback=False, engine=Engine.PYTORCH)
    summary = {}
    for s, net in SHUFFLES.items():
        cfgs = list(PROJ.glob(f"dlc-models-pytorch/iteration-0/*shuffle{s}/train/pytorch_config.yaml"))
        assert len(cfgs) == 1, (s, cfgs)
        summary[s] = {"net_type": net, "pytorch_config": str(cfgs[0]), **patch_pytorch_config(cfgs[0])}
    # the split: indices into the merged CollectedData table of the training dataset (DLC's own loader)
    from deeplabcut.pose_estimation_pytorch.data import DLCLoader
    loader = DLCLoader(str(CFG), shuffle=1, trainset_index=0)
    df = loader.df  # all labelled frames, index = image path parts
    tr, te = loader.split["train"], loader.split["test"]
    img = ["/".join(map(str, i)) if isinstance(i, tuple) else str(i) for i in df.index]
    split = pd.DataFrame({"index": np.r_[tr, te], "image": [img[i] for i in np.r_[tr, te]],
                          "set": ["train"] * len(tr) + ["test"] * len(te)}).sort_values("index")
    split.to_csv(PROJ / "split.csv", index=False)
    same = all(DLCLoader(str(CFG), shuffle=s, trainset_index=0).split["test"] == list(te) for s in list(SHUFFLES)[1:])
    summary["split"] = {"n_train": len(tr), "n_test": len(te), "same_split_all_shuffles": bool(same),
                        "frame_sizes": [list(k) for k in sizes]}
    json.dump(summary, open(PROJ / "setup_summary.json", "w"), indent=1, default=str)
    print(json.dumps({s: (v["net_type"], v["optimizer"], v["scheduler"]) for s, v in summary.items() if s != "split"},
                     indent=1, default=str))
    print(summary["split"])


if __name__ == "__main__":
    main()
