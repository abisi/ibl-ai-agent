# AB155 day-0 session merge

**Goal (user, 2026-11-04):** merge the two parts of the AB155 day-0 recording (2025-02-08) into one session, with units
tracked over the whole recording, so that the passive-post trials (recorded in the second part) are usable. Session id
of the merged session: `AB155_20250208_175103` (first part). Keep every existing output until the merged NWB is checked
and the data validated; the ultimate check is whether spike trains are lost / interrupted at the junction.

## Data
- Originals (user): `M:\analysis\Axel_Bisi\Misc\SessionToConcat` - raw SpikeGLX runs `AB155_g0_t0` (2025-02-08 17:50:51,
  3,005.39 s) and `AB155_g1_t0` (18:55:47, 413.48 s; separate run, sample counter restarted; gap ~15 min), behaviour
  per part (`log_continuous.bin`, `results.csv`; part 2 = 71 passive-post trials), videos per part + joined videos.
- Probes imec0 A1, imec1 OFC, imec2 wS2, imec3 ALM, imec4 SC (all valid in joint_probe_insertion_info) + NIDQ
  (0 sync, 1 trial start, 2 auditory, 3 whisker, 4 valve, 5 / 6 cameras, 7 piezo).
- Per-run CatGT (current, same command for both runs: `-ni -ap -lf -prb=0:5 -gblcar`, 8 `-xa` extractors):
  `analysis/Axel_Bisi/data/AB155/AB155_20250208_175103/Ephys/catgt_AB155_g0`, `.../AB155_20250208_185559/Ephys/catgt_AB155_g1`.

## Joined recording: the April 2025 supercat is reused (001, 2026-11-04)
`lsens-data/AB155/Recording/AB155_20250208_175103/Ephys/AB155_g0` (CatGT `-t=cat ... -supercat={g0}{g1}
-supercat_trim_edges`, 2025-04-28): 3,405.087 s = g0 0 .. 2,997.087 s + g1 4.755 .. 412.754 s.
- g0 part byte-identical to the current g0 (all 5 probes); g1 part identical to the current g1 up to +-2 int16 units
  (r = 0.9999, 99% of samples equal; common-average rounding), sync channel identical.
- NIDQ edges: every kept edge identical on every channel; 312 trial starts, 152 whisker, 70 valve present. Trimmed at
  the junction (supercat edge trimming, keeps the sync train regular for TPrime): g0 last 8.3 s (no trial; the last
  trial's post-stimulus window cut to ~0.5 s), g1 first 4.75 s and last 0.72 s (no trial); 39 piezo licks, 2 pulses
  per camera, sync pulses. Not redone without trimming: the trimmed seconds only matter for ephys (no trials), and
  behaviour / video / DLC are timed from each run's own complete pulse trains (below).
- Junction: joined time 2,997.0869 s (NIDQ); g1 time = joined time - 2,997.0869 + 4.7545 (exact, from the 72 trial
  starts). `xa_1_0_original.txt` = backup of `xa_1_0.txt` (identical).
- Stray empty `AB155_g0_tcat.imec5.*.meta` files (from `-prb=0:5`): not copied.

## Status
- 2026-11-04 16:43: copy done and verified (40 files, 398.25 GiB, rclone check --size-only 0 differences); stray old
  SpikeGLX meta `AB155_g0_imec2.imec2.ap.meta` moved out of the run folder to `Ephys/stray_from_supercat_copy/` (the
  artifact step globs `*imec*` in the run folder as probe folders).
- 2026-11-04 16:44: pipeline submitted, SLURM array 4817848 (tasks 2-6 = imec0-4), frozen config
  `config_snapshots/preprocess_config_axel_20261104_164410.yaml` (= `..._work_AB155_joined.yaml`), inputs
  `config_snapshots/inputs_20261104_164410.txt`; logs `singularity/logs/ephys_AB155_20250208_175103_imec<p>_*`.
  All 5 tasks hung at start-up (~35 s CPU in 63 min, python in D state `d_alloc_parallel` on GPFS, no output);
  cancelled 17:56. Likely contention of 5 tasks on the same session folder (user: the original setup runs the probes
  in sequence).
- 2026-11-04 17:56: resubmitted as ONE task, probes in sequence: SLURM 4818643 (`--time=2-00:00:00`), inputs
  `config/inputs_axel_AB155_joined_seq.txt` (frozen `config_snapshots/inputs_20261104_175653.txt`), same config
  (frozen `preprocess_config_axel_20261104_175653.yaml`); log `singularity/logs/ephys_AB155_20250208_175103_4818643_2.*`.
  Hung the same way. Cause found (faulthandler): importing bombcell walks the joblib CCG cache
  `~/.bombcell/bombcell/ccg_fast/ccg` (781,101 entries) on GPFS; moved to `/home/bisi/bombcell_ccg_cache_old_20261104`
  (user OK), import 21.7 s.
- 2026-11-04 20:01: motion step found to process probe 4 only (leftover MH064 / MH069 filter in
  `run_motion_estimation.py`, also ignored EPHYS_PROBES); jobs cancelled after 1 min (empty imec4/dredge removed).
  Filter removed, probe ids from the folder names, `probe_selected` added (Kuma + local repo; Kuma backup
  `run_motion_estimation.py.bak_20261104`).
- 2026-11-04 20:07: per-probe array resubmitted, SLURM 4819418 (tasks 2-6 = imec0-4, `--time=2-00:00:00`, frozen config
  `preprocess_config_axel_20261104_200711.yaml`); each task runs motion estimation on its own probe
  (`AB155_g0_imec<N>/dredge`).

## Steps
1. Copy (2026-11-04 15:02 - 16:32): supercat folder -> Kuma
   `/work/lsens/bisi/data/AB155/AB155_20250208_175103/Ephys/catgt_AB155_g0` (AP, LF, NIDQ, meta, edge files, offsets;
   rclone log `/work/lsens/bisi/rclone_AB155_joined.log`), then `rclone check --size-only`. The existing per-run g0
   work folder was renamed `/work/lsens/bisi/data/AB155/AB155_20250208_175103_pergate` (the pipeline takes the date
   from the session folder name and writes to output_path/<mouse>/<session>, so the joined run needs the plain name).
2. Pipeline (user-confirmed order): config = copy of `preprocess_config_si_hpc_array_axel_full_work.yaml` with
   `motion: do: true` (drift_correction dredge kept): CatGT off -> artifact correction (whisker times via TPrime) ->
   motion estimation -> drift correction + Kilosort 4 -> Bombcell -> TPrime (sync_event_times: trial_start, auditory,
   whisker, valve, cam0, cam1, piezo from xa_1..7_0; spikes to imec0 time) -> C_Waves -> waveform metrics -> IBL
   format (needs LF). One array task per probe (rows `<path> <probe>`), `submit_ephys.sh` (frozen config).
3. Copy back (user 2026-11-04) to the dedicated folder
   `analysis/Axel_Bisi/Misc/SessionToConcat/AB155_20250208_175103_joined/Ephys/catgt_AB155_g0` (rclone from Kuma
   /work) for the user to check; per-run outputs, `Axel_Bisi/data` and NWB_ks4 untouched.
   Check NWB (source of the junction figures): converter run for the joined session with outputs to
   `.../AB155_20250208_175103_joined/NWB/` (not NWB_ks4): ephys from the joined sorting, both behaviour logs and
   results files mapped to one real-time clock, junction recorded.
   Junction figures (`concat_figure`, from the check NWB) in `.../AB155_20250208_175103_joined/concat_figure/`:
   spike rasters / firing rates / amplitudes per unit around the junction, units lost or appearing, TTL streams
   (sync, trial starts, whisker, cameras, piezo) across the junction, motion (DREDge) estimate across it.
4. Junction check: per unit, spikes / firing rate / amplitude / waveform in the last minutes of part 1 vs the first
   minutes of part 2; units lost / appearing / changing at the junction; drift estimate across it.
5. Merged trial table: part-1 trials + 71 passive-post trials; times from each run's TTLs; check counts, perf, context,
   stimulus types against both results files (g1 has 72 trial starts vs 71 rows: resolve).
6. NWB with a real-time clock and the junction recorded: part 1 as recorded; part 2 shifted by the real start of g1
   (joined spikes after the junction: t - 2,997.0869 + 4.7545 + T2). T2 = start of g1 relative to g0 from the SpikeGLX
   fileCreateTime (3,896 s, 1-s resolution: nothing crosses the gap, so only the gap length carries this uncertainty).
   Behaviour logs, licks, camera frames per part on the same clock (frames in the trimmed seconds keep valid times, no
   ephys there). DLC of both parts already exists (user): `analysis/Axel_Bisi/DLCOutput/DLC_Day0/AB155_{side,top}
   25-02-08 {17-50-56|17-50-54, 18-55-50|18-55-49}DLC_Resnet50_Tracking2DAug10shuffle3_snapshot_best-90_filtered.h5`
   (part 1 595,292 rows vs 595,294 g0 camera pulses; part 2 77,288 rows vs 77,290 g1 pulses), same model for both
   parts. Junction time and trimmed spans stored in the NWB (epochs / description).
