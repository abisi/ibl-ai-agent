# Merged DLC model: training plan (v2, 2026-11-09, after the user's review)

Project: `Axel_Bisi/DLCProjects/Tracking2D-Axel-2026-11-04` — single animal, PyTorch engine, 13 bodyparts
(pupil x4, nose_tip, nose_base, jaw, jaw_ref, spout, particle, tongue, whisker_base, whisker_tip), side and top views
in one model, project colormap `tab20b_r`, pcutoff 0.4. Labels: 6,276 frames from 99 mice (side 2,967, top 3,309).
Nothing is created or submitted until the user approves this recap.

## 1. Split
- Random frame split, TrainingFraction 0.9 (config): ~5,648 train / ~628 test frames, one split (shuffle 1), seed fixed.
- The same split is copied to every architecture with `create_training_dataset_from_existing_split` (PyTorch engine).
  `create_training_model_comparison` only builds TensorFlow models in DLC 3, so the comparison (training, evaluation,
  figures) is our own code on top of the shared split.

## 2. Architectures (one Kuma job each, ImageNet initialisation)
| shuffle | net_type | note |
|---|---|---|
| 1 | `resnet_50` | DLC default; same architecture as the old shuffles 3/5/6 |
| 2 | `resnet_101` | deeper ResNet |
| 3 | `hrnet_w18` | small HRNet |
| 4 | `hrnet_w32` | DLC docs: slower but typically more accurate than ResNets |
| 5 | `hrnet_w48` | largest HRNet |
| 6 | `cspnext_m` | fast modern backbone (RTMPose family) |
| 7 | `dlcrnet_stride16_ms5` | DLC's own multi-scale network (usable for single animal) |
EfficientNet and MobileNet exist only in the TensorFlow engine (DLC 2.x); the PyTorch engine has none of them.

## 3. Training (identical for all)
- Epoch = one full pass over the training frames (~5,648 frames = ~706 iterations at batch size 8).
- 500 epochs (user 2026-11-09), no early stopping (to see overfitting). Cost measured first with a short benchmark job.
- Test evaluation and a snapshot every 10 epochs, all snapshots kept (50 per model).
- Learning rate: AdamW at each architecture's DLC initial rate (ResNet, CSPNeXt 1e-3; HRNet, DLCRNet 1e-4), divided by
  10 at the SAME epochs for every model: 250 and 400 (DLC defaults put the drops at different epochs per family:
  90/120 for ResNet and CSPNeXt, 160/190 for the others).
- Batch size 8 for all. Hyperparameters (batch size 16, initial rate) only as 2 extra jobs on the best architecture,
  after the comparison.
- Augmentation (DLC's actual defaults have covering, motion blur, histogram equalisation off and translation 0; the
  docs page shows them on): affine p 0.5, rotation ±15°, scaling 0.8-1.2, translation 30 px; covering on; motion blur
  on; gaussian noise 12.75; histogram equalisation on; no crop sampling (all frames 640 × 480: the whole image is used;
  checked at setup); horizontal flip off.
- Same DLC version for every model (Kuma venv_dlc), recorded in each model folder.

- Labelling convention (user 2026-11-09): occluded keypoints (tongue in the mouth, pupil in blinks/squints, jaw_ref
  hidden) are not labelled. DLC's covering (CoarseDropout, up to 10 holes of 1-5% of the image side, p 0.5) sets
  keypoints inside a hole to NaN, i.e. the same convention. Learning rate: DLC defaults per architecture (user).
  No hyperparameter jobs for now (user).

## 4. Evaluation and saved results (every snapshot)
- DLC `evaluate_network(per_keypoint_evaluation=True)` on all snapshots: train and test RMSE (all, above pcutoff),
  per keypoint.
- Our own export per snapshot: predictions (x, y, likelihood) for every keypoint of every train and test frame;
  errors per frame, keypoint, view, mouse, session, cohort (AB/MH); in pixels and in mm.
- mm: ephys_utilities `dlc_helpers` conversion (NWB `conversion` attribute, mm per pixel per camera and session; side
  ~0.030, top ~0.033 mm/px); sessions without NWB DLC get the median of their view.
- Also: PCK curves, error vs likelihood, presence (absent keypoints such as pupil in the top view or a hidden tongue
  should have low likelihood: false-positive rate per keypoint), inference speed.

## 5. Human reference
- New DLC project `Tracking2D-Axel-relabel-2026-11-09` with one folder `relabel_100`: 100 frames drawn at random
  (fixed seed) from the test split, renamed and shuffled so the source is not visible, no labels. The user labels them
  in the DLC GUI as usual. A mapping file (kept outside the project) links them back.
- Human error = distance between the two labellings, per keypoint and view (px, mm); shown as a band in the figures.

## 6. Coarse improvement test (separate from the summary figure)
- New resnet_50 (and the best model) vs shuffles 3, 5, 6 on the frames of the 13 mice never seen by the old models
  (AB151-155, MH015, MH062, MH064-065, MH067-070; 559 frames). The new model trained on part of these frames, so this
  only checks that the merged model is not worse; per keypoint and view.
- Then a list of sessions/mice that could benefit from reinference.

## 7. Summary figure (`training_summary/` in the DLC project, ssl-paper-figures style), caption, Methods paragraph
User's panels: example side and top frames with labelled keypoints (project colormap); dataset size; best model train
vs test learning curves with the best snapshot; side/top breakdown with example predictions (heatmaps, part-affinity /
location-refinement fields, likelihood); test performance per view and keypoint; model comparison across epochs
(global, then per keypoint or view).
Human reference in the summary (user 2026-11-09): one horizontal line = mean distance (px) between the two
labellings over all keypoints labelled in both (spout excluded), the same pooling as DLC's test error; per-view panels
use the view's own mean. Values (2026-11-09): all 15.0 px (n 651), side 15.6, top 14.0.
Style: ssl-paper-figures (paperstyle, Nature rcParams), publication quality (user 2026-11-09).
**Spout excluded from all quantifications** (errors, PCK, human reference, model comparison, improvement test; user
2026-11-09); it may still appear in example images.
Additions agreed: PCK curves; error vs likelihood with the pcutoff; presence false positives; human band on every
error panel; per-mouse test error; jitter of fixed points (spout, jaw_ref) on a video as a noise floor; inference speed
vs accuracy across models. Dropped (user): worst-frames gallery, frames-needed curve.
Human reference: the 100 relabel frames are drawn from the test split (user 2026-11-09). Model list approved.
