import os, re, h5py
from concurrent.futures import ProcessPoolExecutor
D='/mnt/lsens-analysis/Axel_Bisi/NWB_ks4'; RE=re.compile(r'^[A-Za-z]+\d+_\d{8}_\d{6}\.nwb$')
def one(n):
    out=[]
    try:
        with h5py.File(f'{D}/{n}','r') as f:
            g=f.get('processing/behavior/BehavioralTimeSeries')
            if g is None: return out
            for k,v in g.items():
                if 'data' in v and 'timestamps' in v and v['data'].shape[0]!=v['timestamps'].shape[0]:
                    out.append((n[:-4],k,v['data'].shape[0],v['timestamps'].shape[0]))
    except Exception as e: out.append((n[:-4],'ERROR',str(e)[:60],0))
    return out
with ProcessPoolExecutor(24) as ex:
    res=[r for rr in ex.map(one, sorted(x for x in os.listdir(D) if RE.match(x))) for r in rr]
import collections
by=collections.defaultdict(list)
for s,k,a,b in res: by[s].append((k,a,b))
for s,v in by.items():
    ks=sorted({k for k,_,_ in v}); a,b=v[0][1],v[0][2]
    print(s, len(v), 'series, e.g.', ks[:4], 'data', a, 'timestamps', b)
