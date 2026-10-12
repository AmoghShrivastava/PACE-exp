"""Replay an Isaac Sim rollout (record_rollout.py) in MuJoCo and write an MP4.

The robot is the Booster T1 MJCF from GMR; joint angles and root pose come from Isaac each step.
Table, net and ball are added to the scene; the paddle is drawn approximately (handle plus blade
around PACE's paddle touch point, 0.345 m along the hand link's -y axis).

MUJOCO_GL=osmesa python render_rollout.py rollout.npz T1_serial.xml out.mp4 --label "..." [--env 0]
"""
import argparse

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser()
ap.add_argument("rollout")
ap.add_argument("xml")
ap.add_argument("out")
ap.add_argument("--label", default="")
ap.add_argument("--env", type=int, default=-1, help="-1: the env with the most hits (same rule for every policy)")
ap.add_argument("--w", type=int, default=640)
ap.add_argument("--h", type=int, default=400)
ap.add_argument("--fps", type=int, default=25)  # rollout is 50 Hz -> 0.5x speed
ap.add_argument("--t0", type=int, default=0)
ap.add_argument("--t1", type=int, default=-1)
ap.add_argument("--png", default="", help="write the last frame as a PNG (preview)")
a = ap.parse_args()

R = np.load(a.rollout)
def _hits_per_env(R):
    h = np.zeros(R["hit"].shape[1], int); cur = np.zeros_like(h, dtype=bool)
    for t in range(len(R["hit"])):
        cur |= R["hit"][t]; h += (R["reset"][t] & cur); cur &= ~R["reset"][t]
    return h
HPE = _hits_per_env(R)
e = int(np.argmax(HPE)) if a.env < 0 else a.env
spec = mujoco.MjSpec.from_file(a.xml)
wb = spec.worldbody
G = mujoco.mjtGeom
# the GMR model already has a floor plane; adding a second one z-fights
wb.add_light(pos=[-1, -2, 4], dir=[0.2, 0.4, -1], diffuse=[0.8, 0.8, 0.8])
wb.add_light(pos=[1, 2, 4], dir=[-0.2, -0.4, -1], diffuse=[0.4, 0.4, 0.4])
top = 0.76
wb.add_geom(type=G.mjGEOM_BOX, size=[1.37, 0.7625, 0.015], pos=[0, 0, top - 0.015], rgba=[0.12, 0.28, 0.55, 1])
wb.add_geom(type=G.mjGEOM_BOX, size=[1.37, 0.01, 0.0005], pos=[0, 0, top + 0.0005], rgba=[1, 1, 1, 1])
for y in (-0.7525, 0.7525):
    wb.add_geom(type=G.mjGEOM_BOX, size=[1.37, 0.01, 0.0006], pos=[0, y, top + 0.0006], rgba=[1, 1, 1, 1])
for x in (-1.36, 1.36):
    wb.add_geom(type=G.mjGEOM_BOX, size=[0.01, 0.7625, 0.0006], pos=[x, 0, top + 0.0006], rgba=[1, 1, 1, 1])
for x in (-1.2, 1.2):
    for y in (-0.65, 0.65):
        wb.add_geom(type=G.mjGEOM_BOX, size=[0.03, 0.03, (top - 0.03) / 2], pos=[x, y, (top - 0.03) / 2], rgba=[0.2, 0.2, 0.22, 1])
wb.add_geom(type=G.mjGEOM_BOX, size=[0.005, 0.915, 0.0763], pos=[0, 0, top + 0.0763], rgba=[0.95, 0.95, 0.95, 0.55])
ball = wb.add_body(name="ball", mocap=True)
ball.add_geom(type=G.mjGEOM_SPHERE, size=[0.02, 0, 0], rgba=[1.0, 0.55, 0.1, 1], contype=0, conaffinity=0)
hand = spec.body("right_hand_link")
hand.add_geom(type=G.mjGEOM_CAPSULE, fromto=[0, -0.06, 0, 0, -0.27, 0], size=[0.012, 0, 0], rgba=[0.55, 0.38, 0.2, 1],
              contype=0, conaffinity=0)
hand.add_geom(type=G.mjGEOM_CYLINDER, size=[0.075, 0.006, 0], pos=[0, -0.345, 0], euler=[0, 90, 0],
              rgba=[0.8, 0.1, 0.1, 1], contype=0, conaffinity=0)
for g in spec.geoms:  # replay is kinematic
    g.contype, g.conaffinity = 0, 0
m = spec.compile()
d = mujoco.MjData(m)

names = [str(n) for n in R["joint_names"]]
qadr = []
for n in names:
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
    qadr.append(m.jnt_qposadr[jid] if jid >= 0 else -1)
free = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE][0]
fadr = m.jnt_qposadr[free]
bid = m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ball")]

ren = mujoco.Renderer(m, a.h, a.w)
cams = []
root0 = R["root_pos"][0, e]
for look, dist, az, el in (([-1.5, root0[1] * 0.5, 0.9], 3.3, 90, -10), ([root0[0] + 0.3, root0[1], 0.95], 2.6, 180 + 15, -12)):
    c = mujoco.MjvCamera()
    c.lookat[:] = look; c.distance = dist; c.azimuth = az; c.elevation = el
    cams.append(c)
font = ImageFont.load_default()
wr = imageio.get_writer(a.out, fps=a.fps, codec="libx264", quality=7, macro_block_size=8)
T = len(R["q"])
serves = hits = rets = 0
hf = rf = False
trail = []
t1 = T if a.t1 < 0 else min(a.t1, T)
for t in range(0, t1):
    d.qpos[fadr:fadr + 3] = R["root_pos"][t, e]
    d.qpos[fadr + 3:fadr + 7] = R["root_quat"][t, e]
    for k, ad in enumerate(qadr):
        if ad >= 0:
            d.qpos[ad] = R["q"][t, e, k]
    d.mocap_pos[bid] = R["ball"][t, e]
    mujoco.mj_forward(m, d)
    hf |= bool(R["hit"][t, e]); rf |= bool(R["ret"][t, e])
    if R["reset"][t, e]:
        serves += 1; hits += int(hf); rets += int(rf); hf = rf = False; trail = []
    trail = (trail + [R["ball"][t, e].copy()])[-12:]
    if t < a.t0:  # counters still accumulate from t=0 so chunks agree
        continue
    tiles = []
    for c in cams:
        ren.update_scene(d, c)
        ren.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0  # software GL: reflections/shadows alias badly
        ren.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        for i, p in enumerate(trail[:-1]):  # faint ball trail
            if ren.scene.ngeom >= ren.scene.maxgeom:
                break
            gg = ren.scene.geoms[ren.scene.ngeom]
            mujoco.mjv_initGeom(gg, G.mjGEOM_SPHERE, [0.008, 0, 0], p, np.eye(3).flatten(),
                                np.array([1, 0.55, 0.1, 0.15 + 0.5 * i / len(trail)], np.float32))
            ren.scene.ngeom += 1
        tiles.append(ren.render())
    img = Image.fromarray(np.concatenate(tiles, axis=1))
    dr = ImageDraw.Draw(img)
    dr.rectangle([0, 0, img.width, 20], fill=(10, 12, 16))
    flash = "  HIT" if hf else ""
    dr.text((8, 4), f"{a.label}   t={t * float(R['dt']):5.2f}s   serves {serves}  hits {hits}  returns {rets}{flash}   0.5x speed",
            fill=(255, 255, 255), font=font)
    wr.append_data(np.asarray(img))
wr.close()
if a.png:
    img.save(a.png)
tot = R["reset"].sum(0)
print(f"[render] {a.out}: env{e} serves {serves} hits {hits} returns {rets}; hits per env {HPE.tolist()}; all envs serves {int(tot.sum())}")
