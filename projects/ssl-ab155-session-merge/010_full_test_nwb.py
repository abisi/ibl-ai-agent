"""010 -- Full test NWB of the joined AB155 day-0 session (behaviour, licks, cameras, DLC, units), made with
NWB_converter as in the repo (NWB_conversion_parallel.convert_data_to_nwb), from the files at their current locations.
Test only: the repo is not changed, nothing is moved; the output goes to SessionToConcat/..._joined/nwb_test/.

Runtime overrides (joined session, run A = run 4819418, Kilosort 4 drift correction on):
  - ephys folders and sync_event_times -> SessionToConcat/AB155_20250208_175103_joined (run A, TPrime to imec0)
  - behaviour results -> part 1 results_corrected_bis.csv (239 trials) + part 2 results_2.csv (71 passive-post
    trials), concatenated (trial_number renumbered), written to nwb_test/inputs/
  - trial TTLs: the part-1 end TTL (last TTL before the junction) is removed from a test copy of the sync event
    files (nwb_test/inputs/sync_event_times) before the converter reads them; the converter then drops the final
    (part-2 end) TTL as usual, so the 310 results rows pair with TTLs 1-239 (part 1) and 241-311 (part 2) by index
    (checked: TTL vs trial_time intervals within 2 ms)
  - licks: part 1 = the converter's licks (behaviour log, mapped to imec0 through the trial TTLs; the TTL mapping is
    given the part-1 TTL pairs only, since the log covers part 1 only); part 2 = the converter's own detector
    (detect_piezo_lick_times) on the joined NIDQ piezo trace, mapped to imec0 through the NIDQ trial TTLs
    (xa_1_0) -> sync trial_start_times; part-1 agreement of the two sources is printed as a check. Part 2 has no
    licks (passive post: piezo maxima ~500-1000 vs ~3000-4800 for part-1 licks, 3 DLC tongue frames); the TPrime
    piezo_licks events in part 2 are threshold noise
  - DLC -> part 1 + part 2 filtered h5 per camera (DLCOutput/DLC_Day0) concatenated, nwb_test/inputs/Video
  - cameras: TPrime frame times of the joined NIDQ (672,582 pulses per camera); the camera on / off pulses (> 50 ms
    from the neighbouring pulse, converter rule) are removed in each part in the test copy: part 1 on pulse (15.4 s
    before the first frame), part 2 off pulse (2.3 s after the last frame) -> 595,292 + 77,288 = DLC rows per part
  - time base: joined NIDQ/imec0 time (part 2 starts at the junction, 2997.087 s; the 891 s real gap is not
    inserted); the junction is stored as acquisition 'supercat_junction'
Output: SessionToConcat/AB155_20250208_175103_joined/nwb_test/AB155_20250208_175103_joined_full_test.nwb
Run (local, conda env nwb): python 010_full_test_nwb.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

sys.path.insert(0, r'C:\Users\bisi\Github\NWB_converter')
os.chdir(r'C:\Users\bisi\Github\NWB_converter')
import NWB_conversion_parallel as ncp  # noqa: E402
import converters.DLC_to_nwb as dlcmod  # noqa: E402
import converters.ephys_to_nwb_test as eph  # noqa: E402
import utils.ephys_converter_misc as ecm  # noqa: E402
import utils.server_paths as sp  # noqa: E402

S = 'AB155_20250208_175103'
J = r'M:\analysis\Axel_Bisi\Misc\SessionToConcat\AB155_20250208_175103_joined'
RUN = os.path.join(J, 'Ephys', 'catgt_AB155_g0')
SYNC = os.path.join(RUN, 'sync_event_times')
OUT = os.path.join(J, 'nwb_test')
INP = os.path.join(OUT, 'inputs')
CONFIG = rf'M:\analysis\Axel_Bisi\data\AB155\{S}\config_{S}.yaml'
TRAIN = rf'M:\data\AB155\Training\{S}'
DLC = r'M:\analysis\Axel_Bisi\DLCOutput\DLC_Day0'
DLC_SUFFIX = 'DLC_Resnet50_Tracking2DAug10shuffle3_snapshot_best-90_filtered.h5'
DLC_PARTS = {'side': ('AB155_side 25-02-08 17-50-56', 'AB155_side 25-02-08 18-55-50'),
             'top': ('AB155_top 25-02-08 17-50-54', 'AB155_top 25-02-08 18-55-49')}
OFF = {l.split()[0].rstrip(':'): l.split()[1:] for l in open(os.path.join(RUN, 'AB155_g0_sc_offsets.txt'))}
TJ_IMEC = float(OFF['sec_imap0'][1])   # junction, imec0 time (TPrime to-stream)
TJ_NIDQ = float(OFF['sec_nidq'][1])    # junction, NIDQ time
CAPTURE = {}


def drop_camera_onoff_pulses(x, name, thresh=0.05):
    """Converter rule (format_ephys_timestamps: first / last pulse > 50 ms from its neighbour = camera turned on / off)
    applied to each part separately; the converter itself only checks the two ends of the joined pulse train."""
    keep = []
    for part in (x[x < TJ_IMEC], x[x >= TJ_IMEC]):
        d = np.diff(part)
        lo, hi = int(d[0] > thresh), len(part) - int(d[-1] > thresh)
        print(f'{name}: part with {len(part)} pulses, camera on pulse removed {bool(lo)}, off pulse removed '
              f'{hi < len(part)} -> {hi - lo}')
        keep.append(part[lo:hi])
    return np.concatenate(keep)


def prepare_inputs():
    os.makedirs(os.path.join(INP, 'Video'), exist_ok=True)
    r1 = pd.read_csv(os.path.join(TRAIN, 'results_corrected_bis.csv'))
    r2 = pd.read_csv(os.path.join(TRAIN, 'results_2.csv'))
    assert list(r1.columns) == list(r2.columns)
    r = pd.concat([r1, r2], ignore_index=True)
    r['trial_number'] = np.arange(1, len(r) + 1)
    res = os.path.join(INP, 'results_joined.csv')
    r.to_csv(res, index=False)
    print(f'results: {len(r1)} + {len(r2)} = {len(r)} trials -> {res}')
    sync_test = os.path.join(INP, 'sync_event_times')
    os.makedirs(sync_test, exist_ok=True)
    for f in os.listdir(SYNC):
        if f.endswith('.txt'):
            x = np.loadtxt(os.path.join(SYNC, f), ndmin=1)
            if f == 'trial_start_times.txt':
                i_end = int(np.flatnonzero(x < TJ_IMEC)[-1])
                print(f'trial TTLs: part-1 end TTL #{i_end} at {x[i_end]:.3f} s removed ({len(x)} -> {len(x) - 1}); '
                      f'part 2 from {x[i_end + 1]:.3f} s')
                x = np.delete(x, i_end)
            if f.endswith('_frame_times.txt'):
                x = drop_camera_onoff_pulses(x, f)
            np.savetxt(os.path.join(sync_test, f), x, fmt='%.6f')
    dlc = []
    for view, (p1, p2) in DLC_PARTS.items():
        out = os.path.join(INP, 'Video', p1 + DLC_SUFFIX)
        if not os.path.exists(out):
            a = pd.read_hdf(os.path.join(DLC, p1 + DLC_SUFFIX))
            b = pd.read_hdf(os.path.join(DLC, p2 + DLC_SUFFIX))
            assert list(a.columns) == list(b.columns)
            pd.concat([a, b], ignore_index=True).to_hdf(out, key='df_with_missing', mode='w')
            print(f'DLC {view}: {len(a)} + {len(b)} = {len(a) + len(b)} frames')
        dlc.append(out.replace('\\', '/'))
    return res, dlc


def install_overrides(res_file, dlc_files):
    imecs = sorted(os.path.join(RUN, f) for f in os.listdir(RUN) if 'imec' in f and os.path.isdir(os.path.join(RUN, f)))
    sp.get_behavior_results_file = lambda config_file: res_file
    for mod in (sp, ncp, dlcmod):
        mod.get_dlc_file_path = lambda config_file: dlc_files
    for mod in (sp, eph):
        mod.get_imec_probe_folder_list = lambda config_file, experimenter=None: imecs
    # event times (trial TTLs, cameras, valve) from the test copy; spike-time files from the run folder
    sp.get_sync_event_times_folder = lambda config_file, experimenter=None: os.path.join(INP, 'sync_event_times')
    eph.get_sync_event_times_folder = lambda config_file, experimenter=None: SYNC

    # NIDQ lick detections (converter's own detector on the joined piezo trace) are kept for part 2
    detect = ecm.detect_piezo_lick_times

    def detect_keep(*a, **kw):
        out = detect(*a, **kw)
        CAPTURE['nidq_licks'] = np.asarray(out[0], dtype=float).ravel()
        return out
    ecm.detect_piezo_lick_times = detect_keep

    # the behaviour log covers part 1 only: pair its trial TTLs with the first part-1 TTLs of the joined NIDQ
    align = ecm.piecewise_align_timestamps

    def align_first_n(t_ref_source, t_ref_target, t_query, event_name='event', return_shifts=True):
        n = min(len(t_ref_source), len(t_ref_target))
        if len(t_ref_source) != len(t_ref_target):
            print(f'[test override] {event_name}: {len(t_ref_source)} log vs {len(t_ref_target)} joined TTLs, '
                  f'first {n} paired (part 1)')
        return align(np.asarray(t_ref_source)[:n], np.asarray(t_ref_target)[:n], t_query, event_name, return_shifts)
    ecm.piecewise_align_timestamps = align_first_n

    log_fn = ncp.analyze_continuous_log

    def analyze_joined(*a, **kw):
        ts, n_frames, cont = log_fn(*a, **kw)
        print(f'[test override] trial TTLs after the converter: {len(ts["trial_TTL"])}')
        # licks: part 1 from the log (as converted), part 2 from the NIDQ detector mapped to imec0
        nidq_ttl = np.loadtxt(os.path.join(RUN, 'AB155_g0_tcat.nidq.xa_1_0.txt'))
        imec_ttl = np.loadtxt(os.path.join(SYNC, 'trial_start_times.txt'))
        assert len(nidq_ttl) == len(imec_ttl)
        to_imec = interp1d(nidq_ttl, imec_ttl, bounds_error=False, fill_value='extrapolate')
        nl = CAPTURE['nidq_licks']
        nl_imec = to_imec(nl)
        log_l = np.asarray(ts['lick_trace'])
        two_col = log_l.ndim == 2
        log_on = log_l[:, 0] if two_col else log_l
        p1 = log_on < TJ_IMEC
        q = np.sort(nl_imec[nl_imec < TJ_IMEC])
        k = np.clip(np.searchsorted(q, log_on[p1]), 1, len(q) - 1)
        d = np.minimum(np.abs(log_on[p1] - q[k - 1]), np.abs(log_on[p1] - q[k]))
        print(f'[test override] part-1 check: {p1.sum()} log licks, {len(q)} NIDQ licks; log lick to nearest NIDQ '
              f'lick median {np.median(d) * 1e3:.2f} ms, {np.mean(d < 0.01):.3f} within 10 ms')
        p2 = np.sort(nl_imec[nl >= TJ_NIDQ])
        part2 = np.c_[p2, np.full(len(p2), np.nan)] if two_col else p2
        ts['lick_trace'] = np.concatenate([log_l[p1], part2])
        print(f'[test override] licks: {p1.sum()} part 1 (log) + {len(p2)} part 2 (NIDQ) = {len(ts["lick_trace"])}')
        CAPTURE['check'] = (float(np.median(d)), float(np.mean(d < 0.01)))
        return ts, n_frames, cont
    ncp.analyze_continuous_log = analyze_joined

    save_fn = ncp.save_nwb_file

    def save_joined(nwb_file, output_folder, with_time_string=False, **kw):
        from pynwb.misc import AnnotationSeries
        nwb_file.add_acquisition(AnnotationSeries(
            name='supercat_junction', data=np.array(['junction part 1 (g0) / part 2 (g1, passive post)']),
            timestamps=np.array([TJ_IMEC]),
            description=f'TEST joined session (CatGT supercat of runs g0 and g1, -supercat_trim_edges): part 2 starts '
                        f'at {TJ_IMEC:.6f} s (imec0 time; NIDQ {TJ_NIDQ:.6f} s). Times are in the joined clock: the '
                        f'~891 s real gap between the runs is not inserted. Sorting: run 4819418 (DREDge + Kilosort 4 '
                        f'drift correction).'))
        save_fn(nwb_file=nwb_file, output_folder=output_folder, with_time_string=with_time_string, **kw)
        src = os.path.join(output_folder, f'{nwb_file.identifier}.nwb')
        dst = os.path.join(output_folder, f'{S}_joined_full_test.nwb')
        if os.path.exists(dst):
            os.remove(dst)
        os.replace(src, dst)
        print(f'saved {dst}')
    ncp.save_nwb_file = save_joined


def main():
    res, dlc = prepare_inputs()
    install_overrides(res, dlc)
    ncp.convert_data_to_nwb(config_file=CONFIG, output_folder=OUT, with_time_string=False, experimenter='Axel_Bisi',
                            add_ephys_recordings=False, remove_extra_ts=False)


if __name__ == '__main__':
    main()
