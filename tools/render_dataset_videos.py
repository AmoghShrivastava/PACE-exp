"""Videos of the extra training data: HIM Rokoko mocap (left) next to its Booster T1 retarget (right).

  rally   : a continuous excerpt of a clip, real time, strokes that entered the library flagged
  strokes : several library strokes, 1.2 s each, half speed, with their contact keys
  vae     : library stroke (mocap) next to the nearest VAE-generated stroke, upper body only, half speed

MUJOCO_GL=osmesa python render_dataset_videos.py <kind> --out x.mp4 [...]
"""
import argparse

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser()
ap.add_argument("kind", choices=["rally", "strokes", "vae"])
ap.add_argument("--out", required=True)
ap.add_argument("--xml", default="/root/GMR/assets/booster_t1/T1_serial.xml")
ap.add_argument("--human_dir", default="/root/mocap/npz")
ap.add_argument("--t1_dir", default="/root/mocap/t1")
ap.add_argument("--lib", default="/root/mocap/t1/stroke_lib_mocap.npz")
ap.add_argument("--vae", default="/root/mocap/t1/stroke_lib_vae.npz")
ap.add_argument("--clip", default="Clip_4")
ap.add_argument("--start", type=float, default=-1, help="rally start time (s); -1 = densest 12 s of strokes")
ap.add_argument("--seconds", type=float, default=12.0)
ap.add_argument("--n", type=int, default=4)
ap.add_argument("--w", type=int, default=480)
ap.add_argument("--h", type=int, default=400)
a = ap.parse_args()

G = mujoco.mjtGeom
spec = mujoco.MjSpec.from_file(a.xml)
hand = spec.body("right_hand_link")
hand.add_geom(type=G.mjGEOM_CAPSULE, fromto=[0, -0.06, 0, 0, -0.27, 0], size=[0.012, 0, 0], rgba=[0.55, 0.38, 0.2, 1])
hand.add_geom(type=G.mjGEOM_CYLINDER, size=[0.075, 0.006, 0], pos=[0, -0.345, 0], euler=[0, 90, 0], rgba=[0.8, 0.1, 0.1, 1])
spec.worldbody.add_light(pos=[0, -2, 4], dir=[0, 0.4, -1], diffuse=[0.7, 0.7, 0.7])
for g in spec.geoms:
    g.contype, g.conaffinity = 0, 0
m = spec.compile()
d = mujoco.MjData(m)
ren = mujoco.Renderer(m, a.h, a.w)
font = ImageFont.load_default()
jn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.dof_jntid[i]) for i in range(6, m.nv)]
qadr = {n: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in jn}

MAIN = ["Hips", "Spine1", "Spine2", "Spine3", "Spine4", "Neck", "Head", "RightShoulder", "RightArm", "RightForeArm",
        "RightHand", "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand", "RightThigh", "RightShin", "RightFoot",
        "RightToe", "LeftThigh", "LeftShin", "LeftFoot", "LeftToe"]


def cam(look, dist, az=125, el=-12):
    c = mujoco.MjvCamera()
    c.lookat[:] = look; c.distance = dist; c.azimuth = az; c.elevation = el
    return c


def flags():
    ren.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
    ren.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0


def add_geom(gtype, size, pos, rgba, mat=None):
    if ren.scene.ngeom >= ren.scene.maxgeom:
        return None
    g = ren.scene.geoms[ren.scene.ngeom]
    mujoco.mjv_initGeom(g, gtype, np.array(size, float), np.array(pos, float),
                        (np.eye(3) if mat is None else mat).flatten(), np.array(rgba, np.float32))
    ren.scene.ngeom += 1
    return g


def human_tile(P, bones, parents):
    """Render the mocap skeleton (robot hidden below the floor)."""
    d.qpos[:] = 0; d.qpos[2] = -5; d.qpos[3] = 1
    mujoco.mj_forward(m, d)
    hips = P[bones.index("Hips")]
    ren.update_scene(d, cam([hips[0], hips[1], 0.95], 3.0)); flags()
    keep = set(bones.index(n) for n in MAIN)
    for j in keep:
        pj = parents[j]
        if pj < 0 or pj not in keep:
            continue
        g = add_geom(G.mjGEOM_CAPSULE, [0.03, 0, 0], [0, 0, 0], [0.92, 0.62, 0.35, 1])
        if g is not None:
            mujoco.mjv_connector(g, G.mjGEOM_CAPSULE, 0.03, P[pj].astype(float), P[j].astype(float))
    add_geom(G.mjGEOM_SPHERE, [0.09, 0, 0], P[bones.index("Head")] + [0, 0, 0.06], [0.92, 0.62, 0.35, 1])
    h, f = P[bones.index("RightHand")], P[bones.index("RightForeArm")]
    u = (h - f) / (np.linalg.norm(h - f) + 1e-9)
    add_geom(G.mjGEOM_SPHERE, [0.07, 0, 0], h + 0.12 * u, [0.8, 0.1, 0.1, 0.9])  # approximate paddle
    return ren.render()


def robot_tile(q, look=None, dist=2.4):
    d.qpos[:] = q
    mujoco.mj_forward(m, d)
    lk = look if look is not None else [q[0], q[1], 0.65]
    ren.update_scene(d, cam(lk, dist)); flags()
    return ren.render()


def caption(tiles, top, left_lab, right_lab):
    img = Image.fromarray(np.concatenate(tiles, axis=1))
    dr = ImageDraw.Draw(img)
    dr.rectangle([0, 0, img.width, 20], fill=(10, 12, 16))
    dr.text((8, 4), top, fill=(255, 255, 255), font=font)
    dr.text((8, 26), left_lab, fill=(20, 20, 20), font=font)
    dr.text((a.w + 8, 26), right_lab, fill=(20, 20, 20), font=font)
    return np.asarray(img)


L = np.load(a.lib)
meta = [str(x).split(":") for x in L["meta"]]

if a.kind in ("rally", "strokes"):
    H = {}
    for clip in sorted(set([a.clip] + [mm[0] for mm in meta])):
        try:
            H[clip] = (np.load(f"{a.human_dir}/{clip}.npz"), np.load(f"{a.t1_dir}/{clip}_t1.npz"))
        except FileNotFoundError:
            pass

if a.kind == "rally":
    hum, rob = H[a.clip]
    bones, parents = [str(b) for b in hum["bones"]], hum["parents"]
    fps = float(hum["fps"])
    peaks = sorted(int(mm[1]) for mm in meta if mm[0] == a.clip)
    span = int(a.seconds * fps)
    if a.start < 0:
        best = max(range(0, len(rob["qpos"]) - span, int(fps)), key=lambda s: sum(s <= p < s + span for p in peaks))
    else:
        best = int(a.start * fps)
    wr = imageio.get_writer(a.out, fps=int(fps), codec="libx264", quality=7, macro_block_size=8)
    n_seen = 0
    for k in range(best, best + span):
        near = any(abs(k - p) <= 3 for p in peaks)
        n_seen = sum(best <= p <= k for p in peaks)
        tiles = [human_tile(hum["pos"][k + 1], bones, parents), robot_tile(rob["qpos"][k])]
        tag = "  STROKE IN LIBRARY" if near else ""
        wr.append_data(caption(tiles, f"HIM Arena mocap {a.clip}, t={k / fps:6.2f}s   real time   library strokes so far: {n_seen}{tag}",
                               "Human (Rokoko, 30 fps)", "Retargeted Booster T1 (GMR IK)"))
    wr.close()
    print(f"[ds] rally {a.clip} start {best / fps:.1f}s strokes {n_seen}")

elif a.kind == "strokes":
    rng = np.random.default_rng(5)
    cand = [i for i, mm in enumerate(meta) if mm[0] in H and float(mm[2]) > 2.2]
    pick = sorted(rng.choice(cand, min(a.n, len(cand)), replace=False).tolist())
    wr = imageio.get_writer(a.out, fps=15, codec="libx264", quality=7, macro_block_size=8)
    for i in pick:
        clip, p, sp = meta[i][0], int(meta[i][1]), float(meta[i][2])
        hum, rob = H[clip]
        bones, parents = [str(b) for b in hum["bones"]], hum["parents"]
        key = L["key"][i]
        for k in range(p - 18, p + 18):
            tiles = [human_tile(hum["pos"][k + 1], bones, parents), robot_tile(rob["qpos"][k])]
            dt = (k - p) / 30.0
            wr.append_data(caption(tiles, f"Library stroke #{i}  ({clip} frame {p})  peak hand speed {sp:.2f} m/s   t-contact={dt:+.2f}s   0.5x speed",
                                   "Human", f"T1   contact key x={key[0]:.2f} y={key[1]:.2f} z={key[2]:.2f} m"))
    wr.close()
    print(f"[ds] strokes {pick}")

else:  # vae: mocap library stroke vs nearest VAE stroke, upper body on a fixed standing base
    V = np.load(a.vae)
    nm = int(V["n_mocap"])
    names = [str(n) for n in L["joint_names"]]
    base = np.zeros(m.nq); base[2] = 0.68; base[3] = 1.0
    rng = np.random.default_rng(9)
    vk = V["key"][nm:]
    pick = rng.choice(len(vk), a.n, replace=False)
    wr = imageio.get_writer(a.out, fps=25, codec="libx264", quality=7, macro_block_size=8)
    for vi in pick:
        mi = int(np.argmin(np.linalg.norm(L["key"] - vk[vi], axis=1)))
        for k in range(V["ref_q"].shape[1]):
            tiles = []
            for traj in (L["ref_q"][mi], V["ref_q"][nm + vi]):
                q = base.copy()
                for j, n in enumerate(names):
                    q[qadr[n]] = traj[k, j]
                tiles.append(robot_tile(q, look=[0, 0, 0.75], dist=2.2))
            t = (k - int(L["contact_idx"])) * float(L["dt"])
            wr.append_data(caption(tiles, f"Upper-body references as the reward sees them   t-contact={t:+.2f}s   0.5x speed",
                                   "Mocap stroke #%d  key %s m" % (mi, ", ".join(f"{x:.2f}" for x in L["key"][mi])),
                                   "VAE stroke #%d  key %s m" % (vi, ", ".join(f"{x:.2f}" for x in vk[vi]))))
    wr.close()
    print(f"[ds] vae {pick.tolist()}")
