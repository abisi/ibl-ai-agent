"""031 -- why does applying the side kpms model collapse to one syllable on the unsmoothed inputs (2026-11-22)?

The full fit on 60 s training chunks is healthy (largest syllable 58 %), but kpms.apply_model on whole sessions gives
~100 % one syllable (smoothed inputs: 84 %). Unsmoothed keypoints have more exactly-repeated frames (23-29 % vs 9-13 %).
For a few sessions (training + held-out), model x inputs (new/new, old/old), this records syllable usage of:
  fit       the fit's own training-chunk labels (checkpoint), training sessions only
  ar_only   labels given the initial latent trajectory (PCA projection), no latent / noise resampling
  iters=k   apply_model with k Gibbs iterations (k = 1, 3, 10, 20)  -- where does the collapse appear?
  chunked   apply_model (20 iterations) on 60 s chunks instead of the whole session
plus, from the returned model: median noise scale s and the latent x frame-to-frame step size.
Output: results/full/kpms/side/rest/031_apply_diagnostic.csv
Run (Kuma): sbatch -J diag031 scripts/run.sbatch venv_kpms 031_apply_diagnostic.py   (BM_SET=full exported inside)
"""
import importlib.util
import os
import sys
from pathlib import Path

os.environ["BM_SET"] = "full"
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("k003", HERE / "003_kpms.py")
K = importlib.util.module_from_spec(spec)
spec.loader.exec_module(K)
TAG = "ns25_ka1e+09_kf1e+08_s0_ho0"
SESS = ["AB086_20231015_141742", "MH026_20250422_181339", "AB121_20240813_125401", "MH031_20250507_104425"]
VERSIONS = {"new": ("full_inputs", "full"), "old": ("full_inputs_smoothed_20261120", "full_smoothed_20261120")}


def usage(z):
    z = np.asarray(z).ravel()
    c = np.bincount(z, minlength=25) / len(z)
    return dict(top=float(np.sort(c)[::-1][0]), second=float(np.sort(c)[::-1][1]), n_1pct=int((c > 0.01).sum()),
                switches_per_s=float(np.mean(np.diff(z) != 0) * K.FPS))


def main():
    import jax
    jax.config.update("jax_enable_x64", True)
    import keypoint_moseq as kpms
    rows = []
    for ver, (inp, res) in VERSIONS.items():
        K.INP = K.BASE / "data" / inp
        proj = K.BASE / "results" / res / "kpms" / "side" / "rest"
        model = kpms.load_checkpoint(str(proj), TAG)[0]
        cfg = kpms.load_config(str(proj))
        cfg["trans_hypparams"] = dict(cfg["trans_hypparams"], num_states=25)
        train = open(K.BASE / "data" / "full_train_sessions.txt").read().split()
        coords, confs, ts, bps = K.load("side", "rest", only=sorted(set(train) | set(SESS)))
        # fit labels on the training chunks (same chunking as 003)
        cc, kk = K.chunks(coords, confs, train)
        data, meta = kpms.format_data(cc, kk, **cfg)
        zfit = {k: np.asarray(v["syllable"]) for k, v in kpms.extract_results(model, meta, save_results=False).items()}
        for sid in SESS:
            base = dict(version=ver, session_id=sid, training=sid in train)
            if sid in train:
                rows.append(dict(base, mode="fit (training chunks)", **usage(np.concatenate([v for k, v in zfit.items() if k.startswith(sid)]))))
            d1, m1 = kpms.format_data({sid: coords[sid]}, {sid: confs[sid]}, **cfg)
            r = kpms.apply_model(model, d1, m1, save_results=False, ar_only=True, num_iters=1, **cfg)
            rows.append(dict(base, mode="ar_only (initial latents)", **usage(r[sid]["syllable"])))
            for k in (1, 3, 10, 20):
                r, mm = kpms.apply_model(model, d1, m1, save_results=False, num_iters=k, return_model=True, **cfg)
                x = np.asarray(mm["states"]["x"]); s = np.asarray(mm["states"]["s"])
                rows.append(dict(base, mode=f"iters={k}", noise_s_median=float(np.median(s)),
                                 latent_step_median=float(np.median(np.abs(np.diff(x, axis=1)))), **usage(r[sid]["syllable"])))
                print(rows[-1], flush=True)
            ch, kc = K.chunks({sid: coords[sid]}, {sid: confs[sid]}, [sid])
            d2, m2 = kpms.format_data(ch, kc, **cfg)
            r = kpms.apply_model(model, d2, m2, save_results=False, num_iters=20, **cfg)
            rows.append(dict(base, mode="chunked 60 s, iters=20", **usage(np.concatenate([np.asarray(v["syllable"]) for v in r.values()]))))
            print(rows[-1], flush=True)
            pd.DataFrame(rows).to_csv(K.BASE / "results" / "full" / "kpms" / "side" / "rest" / "031_apply_diagnostic.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
