"""Read-only inventory of raw videos and DLC h5 outputs for every SSL session (NWB_ks4 list).

Run on haas. Writes:
  <OUT>/inventory.csv        one row per session x view (side/top)
  <OUT>/inventory_files.csv  one row per raw video / DLC h5 file
  <OUT>/inventory_summary.txt
Nothing on the NAS is modified except the output folder.
"""
import glob
import json
import os
import re
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import h5py
import pandas as pd

NWB_DIR = '/mnt/lsens-analysis/Axel_Bisi/NWB_ks4'
RAW_ROOT = '/mnt/lsens-data'
DLC_ROOTS = {'AB': '/mnt/lsens-analysis/Axel_Bisi/data', 'MH': '/mnt/lsens-analysis/Myriam_Hamon/data'}
OUT = [a for a in sys.argv[1:] if not a.startswith('--')][0] if len([a for a in sys.argv[1:] if not a.startswith('--')]) else '/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4/ssl-dlc-kuma'
VIDEO_EXT = ('.avi', '.mp4', '.mov', '.mkv')
DLC_RE = re.compile(r'^(?P<stem>.*?)DLC_(?P<net>[A-Za-z0-9]+)_(?P<task>Tracking2D\w*?)shuffle(?P<shuffle>\d+)_(?P<snap>.*?)(?P<filt>_filtered)?\.h5$')


def view_of(name):
    n = name.lower()
    has_side, has_top = 'side' in n, 'top' in n
    if has_side and not has_top:
        return 'side'
    if has_top and not has_side:
        return 'top'
    return 'unknown'


def ffprobe_frames(path):
    try:
        r = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                            'stream=nb_frames,r_frame_rate,width,height', '-of', 'json', path],
                           capture_output=True, text=True, timeout=120)
        s = json.loads(r.stdout)['streams'][0]
        nf = s.get('nb_frames')
        return dict(n_frames=int(nf) if nf not in (None, 'N/A') else None, fps=s.get('r_frame_rate'),
                    width=s.get('width'), height=s.get('height'))
    except Exception as e:  # noqa
        return dict(n_frames=None, fps=None, width=None, height=None, probe_error=str(e)[:100])


SCORER_RE = re.compile(r'DLC_(?P<net>[A-Za-z0-9]+)_(?P<task>Tracking2D\w*?)shuffle(?P<shuffle>\d+)_(?P<snap>.+)$')
RENAMED_RE = re.compile(r'^(?P<mouse>[A-Z]+\d+)_(?P<view>side|top)_(?P<d>\d\d-\d\d-\d\d)_(?P<t>\d\d-\d\d-\d\d)(?P<filt>_filtered)?\.h5$')


def h5_info(path):
    """Rows and scorer name of a DLC h5 (table or fixed pandas format), header only."""
    out = dict(n_frames=None, scorer=None)
    try:
        with h5py.File(path, 'r') as f:
            key = 'df_with_missing' if 'df_with_missing' in f else list(f.keys())[0]
            g = f[key]
            if 'table' in g:
                out['n_frames'] = int(g['table'].shape[0])
            elif 'axis1' in g:
                out['n_frames'] = int(g['axis1'].shape[0])
            if 'axis0_level0' in g:
                out['scorer'] = '|'.join(x.decode() if isinstance(x, bytes) else str(x) for x in g['axis0_level0'][:])
    except Exception as e:  # noqa
        out['scorer'] = f'ERR {e}'[:80]
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    sessions = sorted(os.path.basename(p)[:-4] for p in glob.glob(f'{NWB_DIR}/*.nwb') if '_backup_' not in p)
    files = []
    sess_info = {}
    for s in sessions:
        mouse = s.split('_')[0]
        raw_dirs = [d for d in (f'{RAW_ROOT}/{mouse}/Recording/{s}/Video', f'{RAW_ROOT}/{mouse}/Training/{s}/Video')
                    if os.path.isdir(d)]
        alt = ''
        if not raw_dirs:  # same-date folder with another timestamp?
            date = s.split('_')[1]
            cand = glob.glob(f'{RAW_ROOT}/{mouse}/*/{mouse}_{date}_*/Video')
            if cand:
                alt = ';'.join(cand)
        dlc_root = DLC_ROOTS.get(mouse[:2])
        dlc_dir = f'{dlc_root}/{mouse}/{s}/Video' if dlc_root else None
        sess_info[s] = dict(mouse=mouse, raw_dirs=raw_dirs, alt_raw=alt,
                            dlc_dir=dlc_dir if dlc_dir and os.path.isdir(dlc_dir) else None)
        for d in raw_dirs:
            for fn in sorted(os.listdir(d)):
                p = os.path.join(d, fn)
                if fn.lower().endswith(VIDEO_EXT) and os.path.isfile(p):
                    files.append(dict(session_id=s, mouse_id=mouse, kind='video', folder=d, file=fn,
                                      view=view_of(fn), stem=os.path.splitext(fn)[0],
                                      size_bytes=os.path.getsize(p), raw_type=d.split('/')[-3]))
        if sess_info[s]['dlc_dir']:
            d = sess_info[s]['dlc_dir']
            for fn in sorted(os.listdir(d)):
                if not fn.endswith('.h5') or '_backup_' in fn:
                    continue
                m = DLC_RE.match(fn)
                r = None if m else RENAMED_RE.match(fn)
                stem = m.group('stem') if m else (f"{r.group('mouse')}_{r.group('view')} {r.group('d')} {r.group('t')}" if r else None)
                files.append(dict(session_id=s, mouse_id=mouse, kind='dlc_h5', folder=d, file=fn,
                                  view=view_of(stem or fn), stem=stem,
                                  name_format='standard' if m else ('renamed_no_scorer' if r else 'unparsed'),
                                  shuffle=int(m.group('shuffle')) if m else None,
                                  snapshot=m.group('snap') if m else None,
                                  net=m.group('net') if m else None,
                                  filtered=bool((m and m.group('filt')) or (r and r.group('filt'))),
                                  size_bytes=os.path.getsize(os.path.join(d, fn))))
    fdf = pd.DataFrame(files)
    reuse = f'{OUT}/inventory_files.csv'
    if '--reuse' in sys.argv and os.path.exists(reuse):  # skip the slow probing step
        fdf = pd.read_csv(reuse)
        return summarise(sessions, sess_info, fdf)

    # frame counts (header only) and h5 row counts, in parallel
    def probe(row):
        p = os.path.join(row['folder'], row['file'])
        return ffprobe_frames(p) if row['kind'] == 'video' else h5_info(p)
    with ThreadPoolExecutor(16) as ex:
        res = list(ex.map(probe, fdf.to_dict('records')))
    fdf = pd.concat([fdf.reset_index(drop=True), pd.DataFrame(res)], axis=1)
    for i in fdf.index[(fdf.kind == 'dlc_h5') & fdf.shuffle.isna() & fdf.scorer.notna()]:
        m = SCORER_RE.search(str(fdf.at[i, 'scorer']).split('|')[0])
        if m:
            fdf.at[i, 'shuffle'], fdf.at[i, 'snapshot'], fdf.at[i, 'net'] = int(m.group('shuffle')), m.group('snap'), m.group('net')
    fdf.insert(2, 'is_short_or_converted', fdf['file'].str.contains('short|convert|labeled|rotat', case=False))
    fdf.to_csv(f'{OUT}/inventory_files.csv', index=False)
    summarise(sessions, sess_info, fdf)


def summarise(sessions, sess_info, fdf):
    rows = []
    for s, info in sess_info.items():
        sub = fdf[fdf.session_id == s] if len(fdf) else fdf
        vids = sub[sub.kind == 'video']
        h5s = sub[sub.kind == 'dlc_h5']
        vid_stems = set(vids['stem'])
        sess_problems = []
        if len(vids) and (vids['view'] == 'unknown').any():
            sess_problems.append('video_name_without_side_or_top')
        main_vids = vids[~vids['is_short_or_converted']]
        conv_only = []
        for view in ('side', 'top'):
            if not (main_vids['view'] == view).any():
                c = vids[(vids['view'] == view) & vids['file'].str.contains('convert', case=False)]
                if len(c):
                    main_vids = pd.concat([main_vids, c])
                    conv_only.append(view)
        if len(main_vids) and (main_vids['view'] == 'side').sum() >= 2 and (main_vids['view'] == 'top').sum() == 0:
            sess_problems.append('two_side_videos_no_top(top_named_side?)')
        if len(h5s) and h5s['shuffle'].nunique() > 1:
            sess_problems.append('mixed_shuffles_in_session')
        for view in ('side', 'top'):
            v = main_vids[main_vids['view'] == view]
            vall = vids[vids['view'] == view]
            h = h5s[h5s['view'] == view]
            hf = h[h['filtered'] == True]  # noqa: E712 (object dtype when empty)
            hu = h[h['filtered'] != True]  # noqa: E712
            probs = list(sess_problems)
            if view in conv_only:
                probs.append('only_converted_mp4_video')
            if len(v) and v['n_frames'].isna().any():
                probs.append('ffprobe_failed')
            if len(h) and (h.get('name_format', pd.Series(dtype=str)) == 'renamed_no_scorer').any():
                probs.append('h5_renamed_without_DLC_scorer')
            if len(h) and (h.get('name_format', pd.Series(dtype=str)) == 'unparsed').any():
                probs.append('h5_unparsed_name')
            if len(v) and not v['file'].str.startswith(info['mouse'] + '_').all():
                probs.append('video_name_wrong_mouse_id')
            if len(v) > 1:
                probs.append('multiple_candidate_videos')
            if len(v) and not len(hf):
                probs.append('video_but_no_filtered_DLC')
            if len(h) and not len(v):
                probs.append('DLC_but_no_raw_video')
            unmatched = [st for st in h['stem'].dropna().unique() if st not in vid_stems]
            if len(v) and unmatched:
                probs.append('h5_stem_not_matching_raw_video')
            if len(hf) > 1:
                probs.append('multiple_filtered_h5')
            vf = v['n_frames'].dropna()
            hfn = hf['n_frames'].dropna()
            if len(vf) == 1 and len(hfn) and (hfn != vf.iloc[0]).any():
                probs.append('frame_count_video_vs_dlc_mismatch')
            rows.append(dict(
                mouse_id=info['mouse'], session_id=s, view=view,
                raw_folder=';'.join(info['raw_dirs']), raw_type=';'.join(d.split('/')[-3] for d in info['raw_dirs']),
                alt_raw_same_date=info['alt_raw'],
                n_videos=len(v), n_videos_incl_short_converted=len(vall),
                video_files='|'.join(v['file']), video_size_gb='|'.join(f'{x / 1e9:.1f}' for x in v['size_bytes']),
                video_n_frames='|'.join(str(int(x)) if pd.notna(x) else 'NA' for x in v['n_frames']),
                video_fps='|'.join(str(x) for x in v['fps']) if len(v) else '',
                dlc_folder=info['dlc_dir'] or '',
                n_dlc_filtered=len(hf), n_dlc_unfiltered=len(hu),
                dlc_filtered_files='|'.join(hf['file']),
                dlc_shuffles='|'.join(sorted({str(int(x)) for x in h['shuffle'].dropna()})),
                dlc_snapshots='|'.join(sorted(set(h['snapshot'].dropna()))),
                dlc_filtered_n_frames='|'.join(str(int(x)) if pd.notna(x) else 'NA' for x in hf['n_frames']),
                h5_unmatched_stems='|'.join(unmatched),
                problems=';'.join(probs)))
    inv = pd.DataFrame(rows)
    inv.to_csv(f'{OUT}/inventory.csv', index=False)

    lines = [f'sessions (NWB_ks4, no _backup_): {len(sessions)}',
             f'sessions with raw Video folder: {sum(bool(i["raw_dirs"]) for i in sess_info.values())}',
             f'sessions with DLC Video folder: {sum(bool(i["dlc_dir"]) for i in sess_info.values())}',
             f'raw video files: {(fdf.kind == "video").sum()}  DLC h5 files: {(fdf.kind == "dlc_h5").sum()}',
             'filtered h5 by shuffle/snapshot: ' + str(Counter(zip(fdf.loc[fdf.filtered == True, 'shuffle'],
                                                                     fdf.loc[fdf.filtered == True, 'snapshot']))),
             '', 'problem counts (session x view rows):']
    pc = Counter(p for ps in inv['problems'] for p in ps.split(';') if p)
    lines += [f'  {k}: {v}' for k, v in pc.most_common()]
    open(f'{OUT}/inventory_summary.txt', 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
