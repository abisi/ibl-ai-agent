"""032 (Kuma) -- BEAST (paninski-lab/beast, ICLR 2026): self-supervised ViT pretrained on our own raw side videos,
then one embedding per video frame for the representation / segmentation comparison (2026-11-22, pilot: 8 sessions).

Stages (python 032_beast.py <stage> [args]):
  links            symlinks ext_data/beast/videos_<view>/<sid>.avi -> raw 200 Hz avi (data/<mouse>/<sid>/Video/*_<view> *.avi)
  extract          N_EXTRACT anchor frames per video, stratified uniformly over the session (one random frame per equal
                   time bin), each saved with its +-1 neighbours (img<frame>.png, the layout beast's contrastive sampler
                   parses). beast's own pca_kmeans selection was not used: on 200 Hz sessions of 0.6-0.9 M frames it
                   needs ~80 GB (frames x PCs x clusters distance array) and was OOM-killed.
  train            beast train: ViT-B/16 initialised from facebook/vit-mae-base, MAE (mask 0.75) + InfoNCE between
                   neighbouring frames (the BEAST recipe), EPOCHS epochs
  embed <sid>      CLS-token embedding (768-d, mask ratio 0) of EVERY frame of the session's video, streamed in batches
                   (beast's own predict_video keeps all per-frame outputs in memory) -> embeddings/<sid>_<view>.npy
                   (float16, frame i = video frame i = DLC frame i)
Paths: /work/lsens/bisi/behaviour_motifs/ext_data/beast/<view>/ ; HF weights cached in behaviour_motifs/hf_cache.
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

BASE = Path("/work/lsens/bisi/behaviour_motifs")
VIEW = os.environ.get("BM_VIEW", "side")
ROOT = BASE / "ext_data" / "beast" / VIEW
VID, FRAMES, MODEL, EMB = ROOT / "videos", ROOT / "frames", ROOT / "model", ROOT / "embeddings"
PILOT = ["AB117_20240723_125437", "AB121_20240813_125401", "AB137_20241116_134442", "AB139_20241119_115957",
         "MH026_20250422_181339", "MH031_20250507_104425", "MH037_20250524_143522", "MH039_20250525_112720"]
N_EXTRACT = 1500
EPOCHS = 300
BEAST = BASE / "venv_beast" / "bin" / "beast"
os.environ.setdefault("HF_HOME", str(BASE / "hf_cache"))


def raw_video(sid):
    d = BASE.parent / "data" / sid.split("_")[0] / sid / "Video"
    return sorted(d.glob(f"*_{VIEW} *.avi"))[0]


def links():
    VID.mkdir(parents=True, exist_ok=True)
    for sid in PILOT:
        dst = VID / f"{sid}.avi"
        if not dst.exists():
            dst.symlink_to(raw_video(sid))
        print(sid, "->", raw_video(sid), flush=True)


def extract():
    import cv2
    rng = np.random.default_rng(0)
    for sid in PILOT:
        d = FRAMES / sid
        d.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(VID / f"{sid}.avi"))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        edges = np.linspace(1, n - 2, N_EXTRACT + 1).astype(int)
        anchors = np.array([rng.integers(a, max(a + 1, b)) for a, b in zip(edges[:-1], edges[1:])])
        idx = np.unique(np.concatenate([anchors - 1, anchors, anchors + 1]))
        for i in idx:
            f = d / f"img{i:08d}.png"
            if f.exists():
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, fr = cap.read()
            if ok:
                cv2.imwrite(str(f), fr)
        np.savetxt(d / "selected_frames.csv", anchors, fmt="%d")
        print(sid, n, "frames;", len(idx), "saved", flush=True)


def config():
    import yaml
    cfg = yaml.safe_load(open(BASE / "ext" / "beast" / "configs" / "vit.yaml"))
    mp = cfg["model"]["model_params"]
    mp.update(backbone="vit-b", use_infoNCE=True, infoNCE_weight=0.03, random_init=False)
    cfg["training"].update(num_epochs=EPOCHS, train_batch_size=256, num_workers=8, check_val_every_n_epoch=10)
    cfg["data"]["data_dir"] = str(FRAMES)
    path = ROOT / "vit_infonce.yaml"
    yaml.safe_dump(cfg, open(path, "w"), sort_keys=False)
    return path


def embed(sid):
    import torch
    from beast.api.model import Model
    from beast.data.video import VideoFrameIterator
    EMB.mkdir(parents=True, exist_ok=True)
    out = EMB / f"{sid}_{VIEW}.npy"
    if out.exists():
        print("exists", out)
        return
    m = Model.from_dir(MODEL)
    net = m.model.cuda().eval()
    it = VideoFrameIterator(VID / f"{sid}.avi", batch_size=512)
    n = it.total_frames
    Z = np.lib.format.open_memmap(str(out) + ".tmp.npy", mode="w+", dtype=np.float16, shape=(n, 768))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for k, b in enumerate(it):
            b["image"] = b["image"].cuda(non_blocking=True)
            z = net.predict_step(b, k)["latents"].float().cpu().numpy()
            Z[b["idx"]] = z.astype(np.float16)
            if k % 200 == 0:
                print(sid, b["idx"][-1], "/", n, flush=True)
    Z.flush()
    del Z
    os.replace(str(out) + ".tmp.npy", out)
    print("done", sid, n, flush=True)


def main():
    stage = sys.argv[1]
    ROOT.mkdir(parents=True, exist_ok=True)
    if stage == "links":
        links()
    elif stage == "extract":
        extract()
    elif stage == "train":
        subprocess.run([str(BEAST), "train", "--config", str(config()), "--data", str(FRAMES), "--output", str(MODEL)], check=True)
    elif stage == "embed":
        embed(sys.argv[2])


if __name__ == "__main__":
    main()
