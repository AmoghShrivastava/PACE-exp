"""Build a T1 forehand stroke library from retargeted HIM mocap, train a SMASH-style
conditional motion VAE on it, and augment the library with FK-filtered VAE samples.

Outputs <out>/stroke_lib_mocap.npz and <out>/stroke_lib_vae.npz with
  ref_q   [M, H, J]  upper-body joint trajectories at 50 Hz (contact at index C)
  key     [M, 3]     paddle point at contact, relative to trunk, in trunk-yaw frame
  joint_names, contact_idx, dt
"""
import argparse
import glob
import json
import math
import os

import mujoco as mj
import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, savgol_filter

UPPER = ["Right_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw", "Waist",
         "Left_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw"]
PADDLE_OFFSET = np.array([0.0, -0.345, 0.0])  # PACE paddle touch point in right_hand_link frame
DT = 0.02
PRE, POST = 0.54, 0.52  # SMASH uses 0.54 s either side of contact
H = int(round((PRE + POST) / DT)) + 1  # 54
C = int(round(PRE / DT))  # 27


class Kin:
    def __init__(self, xml):
        self.m = mj.MjModel.from_xml_path(xml)
        self.d = mj.MjData(self.m)
        self.jn = [mj.mj_id2name(self.m, mj.mjtObj.mjOBJ_JOINT, self.m.dof_jntid[i]) for i in range(6, self.m.nv)]
        self.qadr = {n: self.m.jnt_qposadr[mj.mj_name2id(self.m, mj.mjtObj.mjOBJ_JOINT, n)] for n in self.jn}
        self.hand = mj.mj_name2id(self.m, mj.mjtObj.mjOBJ_BODY, "right_hand_link")
        self.trunk = mj.mj_name2id(self.m, mj.mjtObj.mjOBJ_BODY, "Trunk")
        self.lim = np.array([self.m.jnt_range[mj.mj_name2id(self.m, mj.mjtObj.mjOBJ_JOINT, n)] for n in UPPER])

    def key(self, qpos):
        """Paddle point relative to trunk in trunk-yaw frame."""
        self.d.qpos[:] = qpos
        mj.mj_kinematics(self.m, self.d)
        R = self.d.xmat[self.hand].reshape(3, 3)
        p = self.d.xpos[self.hand] + R @ PADDLE_OFFSET
        rel = p - self.d.xpos[self.trunk]
        Rt = self.d.xmat[self.trunk].reshape(3, 3)
        yaw = math.atan2(Rt[1, 0], Rt[0, 0])
        c, s = math.cos(-yaw), math.sin(-yaw)
        return np.array([c * rel[0] - s * rel[1], s * rel[0] + c * rel[1], rel[2]])

    def key_upper(self, q_upper, base_qpos):
        q = base_qpos.copy()
        for j, n in enumerate(UPPER):
            q[self.qadr[n]] = q_upper[j]
        return self.key(q)


def extract(human_dir, t1_dir, kin, speed_th, prom):
    clips, keys, meta = [], [], []
    for f in sorted(glob.glob(os.path.join(t1_dir, "Clip*_t1.npz"))):
        stem = os.path.basename(f).replace("_t1.npz", "")
        r = np.load(f)
        h = np.load(os.path.join(human_dir, stem + ".npz"))
        b = list(h["bones"])
        pos = h["pos"][1:]
        fps = float(r["fps"])
        jn = list(r["joint_names"])
        qpos = r["qpos"]
        idx = [7 + jn.index(n) for n in UPPER]
        hand = pos[:, b.index("RightHand")]
        sp = np.linalg.norm(savgol_filter(hand, 7, 2, deriv=1, axis=0) * fps, axis=1)
        right = pos[:, b.index("RightArm")] - pos[:, b.index("LeftArm")]
        peaks, _ = find_peaks(sp, height=speed_th, distance=int(0.4 * fps), prominence=prom)
        t_src = np.arange(len(qpos)) / fps
        f_q = interp1d(t_src, qpos[:, idx], axis=0)
        for p in peaks:
            tc = p / fps
            if tc - PRE < 0 or tc + POST > t_src[-1] or r["ik_err"][p] > 1.5:
                continue
            rel = hand[p] - pos[p, b.index("Spine4")]
            if np.dot(rel[:2], right[p, :2]) <= 0:  # backhand side
                continue
            ts = tc + (np.arange(H) - C) * DT
            clips.append(f_q(ts).astype(np.float32))
            keys.append(kin.key(qpos[p]))
            meta.append((stem, int(p), float(sp[p])))
    return np.array(clips), np.array(keys, np.float32), meta


class CVAE(nn.Module):
    """q(z | X, c), p(X | z, c): c = [contact key (3), initial pose (J)]."""

    def __init__(self, J, zdim=16, hid=256):
        super().__init__()
        self.J = J
        cdim = 3 + J
        self.enc = nn.Sequential(nn.Linear(H * J + cdim, hid), nn.ELU(), nn.Dropout(0.1), nn.Linear(hid, hid), nn.ELU(),
                                 nn.Linear(hid, 2 * zdim))
        self.dec = nn.Sequential(nn.Linear(zdim + cdim, hid), nn.ELU(), nn.Dropout(0.1), nn.Linear(hid, hid), nn.ELU(),
                                 nn.Linear(hid, H * J))
        self.zdim = zdim

    def forward(self, x, c):
        mu, logv = self.enc(torch.cat([x.flatten(1), c], 1)).chunk(2, 1)
        z = mu + torch.randn_like(mu) * (0.5 * logv).exp()
        return self.decode(z, c), mu, logv

    def decode(self, z, c):
        return self.dec(torch.cat([z, c], 1)).view(-1, H, self.J)


def train_vae(X, K, iters, device, log):
    J = X.shape[2]
    xm, xs = X.reshape(-1, J).mean(0), X.reshape(-1, J).std(0) + 1e-3
    km, ks = K.mean(0), K.std(0) + 1e-3
    Xn = torch.tensor((X - xm) / xs, device=device)
    Kn = torch.tensor((K - km) / ks, device=device)
    model = CVAE(J).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    n = len(Xn)
    perm = torch.randperm(n, device=device)
    nval = max(8, n // 10)
    va, tr = perm[:nval], perm[nval:]
    cycles = 4
    for it in range(iters):
        # cyclical KL annealing (SMASH: prevents posterior collapse)
        beta = 1e-2 * min(1.0, ((it % (iters // cycles)) / (iters // cycles)) * 2)
        bi = tr[torch.randint(0, len(tr), (256,), device=device)]
        x = Xn[bi]
        c = torch.cat([Kn[bi], x[:, 0]], 1)
        xh, mu, logv = model(x, c)
        rec = ((xh - x) ** 2).mean()
        kl = (-0.5 * (1 + logv - mu ** 2 - logv.exp())).sum(1).mean()
        smooth = ((xh[:, 2:] - 2 * xh[:, 1:-1] + xh[:, :-2]) - (x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2])).pow(2).mean()
        loss = rec + beta * kl + 10.0 * smooth
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 1000 == 0 or it == iters - 1:
            model.eval()
            with torch.no_grad():
                xv = Xn[va]; cv = torch.cat([Kn[va], xv[:, 0]], 1)
                mu_v, _ = model.enc(torch.cat([xv.flatten(1), cv], 1)).chunk(2, 1)
                rec_v = ((model.decode(mu_v, cv) - xv) ** 2).mean().item()
                rec_prior = ((model.decode(torch.randn_like(mu_v), cv) - xv) ** 2).mean().item()
            model.train()
            log(f"vae it {it} rec {rec.item():.4f} kl {kl.item():.2f} beta {beta:.4f} val_rec {rec_v:.4f} val_rec_prior {rec_prior:.4f}")
    model.eval()
    return model, (xm, xs, km, ks)


def augment(model, norm, X, K, kin, base_qpos, n_gen, device, key_tol, rng):
    xm, xs, km, ks = norm
    # broaden the contact-key distribution: jitter real keys (KDE-style) and widen by 1.5x around the mean
    src = rng.integers(0, len(K), n_gen)
    kc = K.mean(0)
    keys_req = kc + 1.5 * (K[src] - kc) + rng.normal(0, 1, (n_gen, 3)) * K.std(0) * 0.3
    s0 = X[rng.integers(0, len(X), n_gen), 0]
    with torch.no_grad():
        c = torch.cat([torch.tensor((keys_req - km) / ks, device=device, dtype=torch.float32),
                       torch.tensor((s0 - xm) / xs, device=device, dtype=torch.float32)], 1)
        z = torch.randn(n_gen, model.zdim, device=device)
        G = model.decode(z, c).cpu().numpy() * xs + xm
    keep, gkeys, stats = [], [], dict(limit=0, jerk=0, key=0)
    jerk_ref = np.percentile(np.abs(np.diff(X, 3, axis=1)).max(axis=(1, 2)), 99)
    lo, hi = kin.lim[:, 0], kin.lim[:, 1]
    for i, g in enumerate(G):
        if (g < lo - 0.15).any() or (g > hi + 0.15).any():
            stats["limit"] += 1; continue
        if np.abs(np.diff(g, 3, axis=0)).max() > 1.5 * jerk_ref:
            stats["jerk"] += 1; continue
        g = np.clip(g, lo, hi)
        # SMASH keys each generated clip by its *actual* contact racket position (FK), not the request
        k = kin.key_upper(g[C], base_qpos)
        if np.linalg.norm(k - keys_req[i]) > key_tol:  # decoder ignored its condition -> implausible sample
            stats["key"] += 1; continue
        keep.append(g); gkeys.append(k)
    return np.array(keep, np.float32), np.array(gkeys, np.float32), stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", required=True)
    ap.add_argument("--t1", required=True)
    ap.add_argument("--xml", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--speed_th", type=float, default=1.5)
    ap.add_argument("--prom", type=float, default=0.8)
    ap.add_argument("--vae_iters", type=int, default=20000)
    ap.add_argument("--n_gen", type=int, default=12000)
    ap.add_argument("--key_tol", type=float, default=0.15)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, "library_log.txt"), "w")

    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()

    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    kin = Kin(a.xml)
    X, K, meta = extract(a.human, a.t1, kin, a.speed_th, a.prom)
    log(f"mocap forehand strokes: {len(X)}  window H={H} contact={C} joints={UPPER}")
    log(f"key mean {K.mean(0).round(3)} std {K.std(0).round(3)} min {K.min(0).round(3)} max {K.max(0).round(3)}")
    np.savez_compressed(os.path.join(a.out, "stroke_lib_mocap.npz"), ref_q=X, key=K, joint_names=np.array(UPPER),
                        contact_idx=C, dt=DT, meta=np.array([f"{m[0]}:{m[1]}:{m[2]:.2f}" for m in meta]))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, norm = train_vae(X, K, a.vae_iters, dev, log)
    torch.save(dict(model=model.state_dict(), norm=norm), os.path.join(a.out, "motion_vae.pt"))
    base = np.zeros(kin.m.nq); base[2] = 0.7; base[3] = 1.0
    G, GK, st = augment(model, norm, X, K, kin, base, a.n_gen, dev, a.key_tol, rng)
    log(f"VAE generated {a.n_gen}, kept {len(G)} after filtering; rejects {st}")
    log(f"aug key mean {GK.mean(0).round(3)} std {GK.std(0).round(3)}")
    XA, KA = np.concatenate([X, G]), np.concatenate([K, GK])
    np.savez_compressed(os.path.join(a.out, "stroke_lib_vae.npz"), ref_q=XA, key=KA, joint_names=np.array(UPPER),
                        contact_idx=C, dt=DT, n_mocap=len(X))
    # coverage: fraction of a grid of plausible contact points within 5 cm of a library key
    grid = np.stack(np.meshgrid(np.linspace(K[:, 0].min() - .1, K[:, 0].max() + .1, 12),
                                np.linspace(K[:, 1].min() - .1, K[:, 1].max() + .1, 12),
                                np.linspace(K[:, 2].min() - .1, K[:, 2].max() + .1, 12)), -1).reshape(-1, 3)
    for name, kk in (("mocap", K), ("mocap+vae", KA)):
        dmin = np.min(np.linalg.norm(grid[:, None] - kk[None], axis=-1), axis=1)
        log(f"coverage[{name}] grid within 5cm: {(dmin < 0.05).mean():.3f}, within 10cm: {(dmin < 0.10).mean():.3f}, median nn dist {np.median(dmin):.3f}")
    json.dump(dict(n_mocap=len(X), n_aug=len(G), rejects=st), open(os.path.join(a.out, "library_summary.json"), "w"))
