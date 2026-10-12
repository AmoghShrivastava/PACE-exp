"""Render a strip: human skeleton (matplotlib) vs retargeted T1 (MuJoCo) at given frames."""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco as mj
import numpy as np
from general_motion_retargeting.params import ROBOT_XML_DICT

human = np.load(sys.argv[1])
robot = np.load(sys.argv[2])
frames = [int(x) for x in sys.argv[3].split(",")]
out = sys.argv[4]

model = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT["booster_t1"]))
data = mj.MjData(model)
renderer = mj.Renderer(model, 320, 320)
cam = mj.MjvCamera()
cam.distance, cam.elevation, cam.azimuth = 2.2, -10, 90+180

pos, par = human["pos"], human["parents"]
fig, ax = plt.subplots(2, len(frames), figsize=(3 * len(frames), 6))
for i, f in enumerate(frames):
    p = pos[f + 1]
    a = ax[0, i]
    for j, pj in enumerate(par):
        if pj >= 0 and j < 61:
            a.plot([p[j, 0], p[pj, 0]], [p[j, 2], p[pj, 2]], "k-", lw=1)
    a.set_aspect("equal"); a.set_xlim(p[0, 0] - 1, p[0, 0] + 1); a.set_ylim(0, 2); a.set_title(f"human f{f}")
    a.axis("off")
    data.qpos[:] = robot["qpos"][f]
    mj.mj_forward(model, data)
    cam.lookat[:] = data.qpos[:3]
    renderer.update_scene(data, cam)
    ax[1, i].imshow(renderer.render()); ax[1, i].axis("off"); ax[1, i].set_title("T1")
plt.tight_layout(); plt.savefig(out, dpi=70)
