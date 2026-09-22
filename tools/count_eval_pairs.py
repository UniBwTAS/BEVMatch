"""Zaehlt die evaluierten Frame-Paare je Bin direkt aus den Dataset-Infos (R1.2).

Bildet die Auswahl aus *_dataset_keypoints.py::_find_random_sample_in_range exakt nach:
  - Suche NUR vorwaerts, offset 1..min(150, len-idx)
  - Zeitfenster |dt| <= 20 s (Stempel > 1e10 gelten als Mikrosekunden)
  - Relativpose inv(A) @ B, auf 2D projiziert (z-Zeile genullt)
  - gueltig, wenn min_dist <= |t| <= max_dist und |yaw| <= max_angle
  - kein Kandidat -> kein gueltiges Paar fuer diesen Frame

Die Zufallswahl unter den Kandidaten beeinflusst nur, WELCHER Partner genommen wird,
nicht OB einer existiert -- die Zaehlung ist daher deterministisch.

Aufruf: python tools/count_eval_pairs.py <infos.pkl> [<Name>]
"""
import pickle
import sys

import numpy as np

RANGES = [
    ('close', 0.0, 5.0, 10.0),      # 5m10d
    ('mid', 5.0, 10.0, 30.0),       # 10m30d
    ('far', 10.0, 20.0, 50.0),      # 20m50d
]
MAX_OFFSET = 150
MAX_DT_S = 20.0


def load_frames(path):
    with open(path, 'rb') as fh:
        info = pickle.load(fh)
    dl = info.get('data_list', info) if isinstance(info, dict) else info
    poses, stamps = [], []
    for it in dl:
        e = it.get('ego2global')
        if e is None and 'images' in it:                     # Waymo/KITTI-Format
            e = it['images'].get('ego2global')
        if e is None:
            continue
        ts = it.get('timestamp')
        if ts is None:
            ts = it.get('timestamp_micros', len(poses))
        poses.append(np.asarray(e, np.float64).reshape(4, 4))
        stamps.append(float(ts))
    return np.array(poses), np.array(stamps)


def count(poses, stamps):
    n = len(poses)
    sec = np.where(stamps > 1e10, stamps / 1e6, stamps)
    xy = poses[:, :2, 3]
    yaw = np.arctan2(poses[:, 1, 0], poses[:, 0, 0])
    found = {name: 0 for name, _, _, _ in RANGES}
    for i in range(n):
        hi = min(i + MAX_OFFSET, n - 1)
        if hi <= i:
            continue
        j = np.arange(i + 1, hi + 1)
        ok_t = np.abs(sec[j] - sec[i]) <= MAX_DT_S
        if not ok_t.any():
            continue
        j = j[ok_t]
        # Relativpose im Frame i: Translation rotiert in i's Koordinaten, z faellt weg
        d = xy[j] - xy[i]
        c, s = np.cos(-yaw[i]), np.sin(-yaw[i])
        tx = d[:, 0] * c - d[:, 1] * s
        ty = d[:, 0] * s + d[:, 1] * c
        dist = np.hypot(tx, ty)
        da = np.degrees(np.arctan2(np.sin(yaw[j] - yaw[i]), np.cos(yaw[j] - yaw[i])))
        for name, lo, hi_d, amax in RANGES:
            if ((dist >= lo) & (dist <= hi_d) & (np.abs(da) <= amax)).any():
                found[name] += 1
    return found, n


if __name__ == '__main__':
    path = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 else path
    poses, stamps = load_frames(path)
    found, n = count(poses, stamps)
    print(f'{label}: {n} Frames')
    for name, lo, hi_d, amax in RANGES:
        print(f'   {name:6s} ({lo:.0f}-{hi_d:.0f} m, <={amax:.0f} deg): {found[name]:7d} Paare')
