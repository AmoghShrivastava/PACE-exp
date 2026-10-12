"""Broadcast-style replay of an Isaac Sim rollout in MuJoCo (1280x720).

Main view: behind and above the robot, looking down the table at the incoming ball.
Inset: side close-up that follows the robot. Exact joint angles, base pose and ball
position come from record_rollout.py; table, net and paddle are drawn around them.

MUJOCO_GL=osmesa python render_v2.py rollout.npz T1_serial.xml out.mp4 --title "..." [--t0 --t1]
"""
import argparse

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser()
ap.add_argument("rollout"); ap.add_argument("xml"); ap.add_argument("out")
ap.add_argument("--title", default=""); ap.add_argument("--sub", default="")
ap.add_argument("--env", type=int, default=-1)
ap.add_argument("--t0", type=int, default=0); ap.add_argument("--t1", type=int, default=-1)
ap.add_argument("--fps", type=int, default=25)
a = ap.parse_args()
W, H, IW, IH = 1280, 720, 448, 252

R = np.load(a.rollout)


def hits_per_env(R):
    h = np.zeros(R["hit"].shape[1], int); cur = np.zeros_like(h, dtype=bool)
    for t in range(len(R["hit"])):
        cur |= R["hit"][t]; h += (R["reset"][t] & cur); cur &= ~R["reset"][t]
    return h


HPE = hits_per_env(R)
e = int(np.argmax(HPE)) if a.env < 0 else a.env

G = mujoco.mjtGeom
spec = mujoco.MjSpec.from_file(a.xml)
for g in spec.geoms:  # drop the model's own floor; we draw a nicer one
    if g.type == G.mjGEOM_PLANE:
        g.pos = [0, 0, -0.02]; g.rgba = [0, 0, 0, 0]
    g.contype, g.conaffinity = 0, 0
_sky = [t for t in spec.textures if t.type == mujoco.mjtTexture.mjTEXTURE_SKYBOX]
if _sky:  # the model ships its own skybox; recolour it instead of adding a second one
    for t in _sky:
        t.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT; t.rgb1 = [0.42, 0.58, 0.80]; t.rgb2 = [0.90, 0.94, 0.98]; t.file = ""
else:
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.42, 0.58, 0.80], rgb2=[0.90, 0.94, 0.98], width=256, height=256)
spec.add_texture(name="floor", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                 rgb1=[0.30, 0.33, 0.38], rgb2=[0.26, 0.29, 0.34], width=512, height=512)
spec.add_material(name="floor", textures=["", "floor"], texrepeat=[12, 12])
wb = spec.worldbody
wb.add_geom(type=G.mjGEOM_PLANE, size=[14, 14, 0.1], material="floor", contype=0, conaffinity=0)
wb.add_light(pos=[-2, -3, 5], dir=[0.3, 0.5, -1], diffuse=[0.75, 0.75, 0.75], specular=[0.2, 0.2, 0.2])
wb.add_light(pos=[2, 3, 5], dir=[-0.3, -0.5, -1], diffuse=[0.45, 0.45, 0.45])
top = 0.76
blue, white, dark = [0.07, 0.24, 0.52, 1], [0.97, 0.97, 0.97, 1], [0.12, 0.13, 0.15, 1]
geoms = [
    (G.mjGEOM_BOX, [1.37, 0.7625, 0.015], [0, 0, top - 0.015], blue),
    (G.mjGEOM_BOX, [1.37, 0.006, 0.0006], [0, 0, top + 0.0006], white),
    (G.mjGEOM_BOX, [1.37, 0.01, 0.0006], [0, 0.7525, top + 0.0006], white),
    (G.mjGEOM_BOX, [1.37, 0.01, 0.0006], [0, -0.7525, top + 0.0006], white),
    (G.mjGEOM_BOX, [0.01, 0.7625, 0.0006], [1.36, 0, top + 0.0006], white),
    (G.mjGEOM_BOX, [0.01, 0.7625, 0.0006], [-1.36, 0, top + 0.0006], white),
    (G.mjGEOM_BOX, [0.004, 0.915, 0.0763], [0, 0, top + 0.0763], [0.15, 0.17, 0.2, 0.55]),
    (G.mjGEOM_BOX, [0.006, 0.915, 0.008], [0, 0, top + 0.1525], white),
    (G.mjGEOM_CYLINDER, [0.012, 0.085, 0], [0, 0.93, top + 0.085], dark),
    (G.mjGEOM_CYLINDER, [0.012, 0.085, 0], [0, -0.93, top + 0.085], dark),
]
for x in (-1.15, 1.15):
    for y in (-0.62, 0.62):
        geoms.append((G.mjGEOM_BOX, [0.035, 0.035, (top - 0.03) / 2], [x, y, (top - 0.03) / 2], dark))
for gt, sz, ps, rgba in geoms:
    wb.add_geom(type=gt, size=sz, pos=ps, rgba=rgba, contype=0, conaffinity=0)
ball = wb.add_body(name="ball", mocap=True)
ball.add_geom(type=G.mjGEOM_SPHERE, size=[0.025, 0, 0], rgba=[1.0, 0.5, 0.05, 1], contype=0, conaffinity=0)
hand = spec.body("right_hand_link")
hand.add_geom(type=G.mjGEOM_CAPSULE, fromto=[0, -0.05, 0, 0, -0.26, 0], size=[0.013, 0, 0], rgba=[0.62, 0.43, 0.24, 1], contype=0, conaffinity=0)
hand.add_geom(type=G.mjGEOM_CYLINDER, size=[0.078, 0.004, 0], pos=[0.004, -0.345, 0], euler=[0, 90, 0], rgba=[0.82, 0.08, 0.08, 1], contype=0, conaffinity=0)
hand.add_geom(type=G.mjGEOM_CYLINDER, size=[0.078, 0.004, 0], pos=[-0.004, -0.345, 0], euler=[0, 90, 0], rgba=[0.08, 0.08, 0.09, 1], contype=0, conaffinity=0)
m = spec.compile()
m.vis.global_.offwidth, m.vis.global_.offheight = W, H
m.vis.headlight.ambient[:] = [0.35, 0.35, 0.35]
d = mujoco.MjData(m)

names = [str(n) for n in R["joint_names"]]
qadr = [m.jnt_qposadr[j] if (j := mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)) >= 0 else -1 for n in names]
free = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE][0]
fa = m.jnt_qposadr[free]
bid = m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ball")]
ren = mujoco.Renderer(m, H, W)
ren_in = mujoco.Renderer(m, IH, IW)

main = mujoco.MjvCamera(); main.lookat[:] = [-0.8, 0.0, 0.85]; main.distance = 3.7; main.azimuth = 14; main.elevation = -16
side = mujoco.MjvCamera(); side.distance = 2.1; side.azimuth = 100; side.elevation = -6

try:
    F = ImageFont.truetype("DejaVuSans-Bold.ttf", 26); F2 = ImageFont.truetype("DejaVuSans.ttf", 18); F3 = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
except OSError:
    F = F2 = F3 = ImageFont.load_default()

T = len(R["q"])
t1 = T if a.t1 < 0 else min(a.t1, T)
wr = imageio.get_writer(a.out, fps=a.fps, codec="libx264", quality=8, macro_block_size=8)
serves = hits = rets = 0
hf = rf = False
flash = 0
trail = []
look = R["root_pos"][0, e].copy()


def flags(r):
    r.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
    r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0


for t in range(0, t1):
    newhit = bool(R["hit"][t, e]) and not hf
    hf |= bool(R["hit"][t, e]); rf |= bool(R["ret"][t, e])
    if newhit:
        flash = 12
    if R["reset"][t, e]:
        serves += 1; hits += int(hf); rets += int(rf); hf = rf = False; trail = []
    trail = (trail + [R["ball"][t, e].copy()])[-14:]
    look = 0.85 * look + 0.15 * R["root_pos"][t, e]
    flash = max(0, flash - 1)
    if t < a.t0:
        continue
    d.qpos[fa:fa + 3] = R["root_pos"][t, e]; d.qpos[fa + 3:fa + 7] = R["root_quat"][t, e]
    for k, ad in enumerate(qadr):
        if ad >= 0:
            d.qpos[ad] = R["q"][t, e, k]
    d.mocap_pos[bid] = R["ball"][t, e]
    mujoco.mj_forward(m, d)
    ren.update_scene(d, main); flags(ren)
    for i, pt in enumerate(trail[:-1]):
        if ren.scene.ngeom < ren.scene.maxgeom:
            mujoco.mjv_initGeom(ren.scene.geoms[ren.scene.ngeom], G.mjGEOM_SPHERE, [0.011, 0, 0], pt, np.eye(3).flatten(),
                                np.array([1, 0.55, 0.1, 0.12 + 0.55 * i / len(trail)], np.float32))
            ren.scene.ngeom += 1
    img = Image.fromarray(ren.render())
    side.lookat[:] = [look[0] + 0.25, look[1], 0.85]
    ren_in.update_scene(d, side); flags(ren_in)
    inset = Image.fromarray(ren_in.render())
    img.paste(Image.new("RGB", (IW + 6, IH + 6), (255, 255, 255)), (W - IW - 26, H - IH - 26))
    img.paste(inset, (W - IW - 23, H - IH - 23))
    dr = ImageDraw.Draw(img, "RGBA")
    dr.rounded_rectangle([20, 18, 20 + 640, 18 + 92], 12, fill=(12, 22, 40, 200))
    dr.text((36, 28), a.title, font=F, fill=(255, 255, 255))
    dr.text((36, 66), a.sub, font=F2, fill=(200, 215, 235))
    sb = f"SERVES {serves}   HITS {hits}   RETURNS {rets}     0.5× speed"
    dr.rounded_rectangle([20, H - 70, 20 + 610, H - 22], 10, fill=(12, 22, 40, 200))
    dr.text((36, H - 60), sb, font=F3, fill=(255, 255, 255))
    if flash:
        dr.rounded_rectangle([W - 200, 22, W - 24, 70], 10, fill=(240, 125, 34, 230))
        dr.text((W - 172, 32), "HIT", font=F, fill=(255, 255, 255))
    dr.rounded_rectangle([W - IW - 26, H - IH - 58, W - IW + 86, H - IH - 30], 8, fill=(12, 22, 40, 200))
    dr.text((W - IW - 16, H - IH - 55), "side view", font=F2, fill=(255, 255, 255))
    wr.append_data(np.asarray(img))
wr.close()
print(f"[render2] {a.out}: env{e} serves {serves} hits {hits} returns {rets}; hits per env {HPE.tolist()}")
