"""012 (Kuma, CPU) -- grid movies of the global top states: one tile per session (2 x 4, one bout each, from 011 clips),
-0.25 to +1.0 s around bout onset at 100 Hz, written at 50 fps (half speed); an orange 'STATE ON' bar marks t >= 0 (state onset),
tile label = mouse + cohort (R+ green, R- magenta). Output: results/movies/<method>_<view>_<state>.mp4
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

BASE = Path("/work/lsens/bisi/behaviour_motifs")
CL = BASE / "results" / "video" / "clips"
OUT = BASE / "results" / "movies"
COH = {"R+": (0, 180, 0), "R-": (200, 0, 200)}  # BGR


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sel = pd.read_csv(BASE / "data" / "001_selected_sessions.csv")
    for view in ("side", "top"):
        data = {}
        for sid in sel.session_id:
            f = CL / f"{sid}_{view}.npz"
            if f.exists():
                d = np.load(f)
                data[sid] = {str(k): d["clips"][i] for i, k in enumerate(d["keys"])}
        keys = sorted({k for v in data.values() for k in v})
        for key in keys:
            m, s = key.split("|")
            tiles = []
            n = None
            for sid in sel.session_id:
                c = data.get(sid, {}).get(key)
                if c is not None:
                    n = len(c)
                tiles.append((sid, c))
            h, w = 144, 192
            path = OUT / f"{m}_{view}_{s}.mp4"
            vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 50, (4 * w, 2 * h))
            for j in range(n):
                frame = np.zeros((2 * h, 4 * w, 3), np.uint8)
                for i, (sid, c) in enumerate(tiles):
                    y, x = (i // 4) * h, (i % 4) * w
                    if c is not None:
                        frame[y:y + h, x:x + w] = cv2.cvtColor(c[j], cv2.COLOR_GRAY2BGR)
                    rg = sel.set_index("session_id").reward_group[sid]
                    cv2.putText(frame, f"{sid[:5]} {rg}", (x + 4, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COH[rg], 1)
                    if c is not None and j >= 25:  # v2: explicit "STATE ON" bar instead of a dot
                        cv2.rectangle(frame, (x, y + h - 14), (x + w - 1, y + h - 1), (0, 200, 255), -1)
                        cv2.putText(frame, "STATE ON", (x + 4, y + h - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1)
                cv2.putText(frame, f"{m} {view} state {s}  t={(j - 25) * 10:+d} ms (0.5x)", (4, 2 * h - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                vw.write(frame)
            vw.release()
            print("movie", path.name, flush=True)


if __name__ == "__main__":
    main()
