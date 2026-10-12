"""Segment strokes in Rokoko mocap NPZs: dominant hand, swing peaks, forehand/backhand split."""
import glob
import json
import os
import sys

import numpy as np
from scipy.signal import find_peaks, savgol_filter

d_in = sys.argv[1]
B = None
report = {}
all_strokes = []
for f in sorted(glob.glob(os.path.join(d_in, "Clip*.npz"))):
    d = np.load(f)
    bones = list(d["bones"])
    B = {n: i for i, n in enumerate(bones)}
    pos = d["pos"][1:]  # drop T-pose frame
    fps = float(d["fps"])
    T = len(pos)
    hips, sp4 = pos[:, B["Hips"]], pos[:, B["Spine4"]]
    # body heading: left->right shoulder axis; forward = up x right
    rs, ls = pos[:, B["RightArm"]], pos[:, B["LeftArm"]]
    right = rs - ls
    right[:, 2] = 0
    right /= np.linalg.norm(right, axis=1, keepdims=True) + 1e-9
    up = np.array([0, 0, 1.0])
    fwd = np.cross(up, right)
    stats = {}
    for side in ("Right", "Left"):
        h = pos[:, B[f"{side}Hand"]]
        v = savgol_filter(h, 7, 2, deriv=1, axis=0) * fps
        sp = np.linalg.norm(v, axis=1)
        stats[side] = dict(mean_speed=float(sp.mean()), p99=float(np.percentile(sp, 99)))
    dom = max(stats, key=lambda s: stats[s]["p99"])
    h = pos[:, B[f"{dom}Hand"]]
    v = savgol_filter(h, 7, 2, deriv=1, axis=0) * fps
    sp = np.linalg.norm(v, axis=1)
    peaks, props = find_peaks(sp, height=3.0, distance=int(0.6 * fps), prominence=1.5)
    strokes = []
    for p in peaks:
        rel = h[p] - sp4[p]
        lat = float(np.dot(rel, right[p]))  # + => racket on dominant side (forehand for right hander)
        vlat = float(np.dot(v[p], right[p]))
        kind = "forehand" if (lat > 0) == (dom == "Right") else "backhand"
        strokes.append(dict(clip=os.path.basename(f), frame=int(p + 1), t=float(p / fps), speed=float(sp[p]),
                            kind=kind, lat=lat, fwd=float(np.dot(rel, fwd[p])), height=float(h[p, 2]),
                            hip_h=float(hips[p, 2]), vlat=vlat))
    report[os.path.basename(f)] = dict(frames=T, seconds=T / fps, fps=fps, dominant=dom, hands=stats,
                                       n_strokes=len(strokes),
                                       n_forehand=sum(s["kind"] == "forehand" for s in strokes),
                                       n_backhand=sum(s["kind"] == "backhand" for s in strokes),
                                       peak_speed_median=float(np.median([s["speed"] for s in strokes])) if strokes else None,
                                       hip_travel_m=float(np.ptp(hips[:, 0]) + np.ptp(hips[:, 1])))
    all_strokes += strokes

json.dump(dict(clips=report, strokes=all_strokes), open(os.path.join(d_in, "strokes.json"), "w"), indent=1)
for k, v in report.items():
    print(k, {a: (round(b, 2) if isinstance(b, float) else b) for a, b in v.items() if a != "hands"})
sp = np.array([s["speed"] for s in all_strokes])
print("TOTAL strokes", len(all_strokes), "forehand", sum(s["kind"] == "forehand" for s in all_strokes),
      "peak hand speed m/s: median %.2f p10 %.2f p90 %.2f" % (np.median(sp), np.percentile(sp, 10), np.percentile(sp, 90)))
