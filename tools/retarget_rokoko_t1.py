"""Retarget HIM Rokoko mocap (NPZ from fbx_to_npz.py) to Booster T1 with GMR (mink IK).

Rokoko globals are converted to the SMPL-X convention GMR expects: z-up, facing -y,
rest orientation R_x(90deg). Frame 0 of every clip is the T-pose, so per-bone
deltas G_t G_0^-1 are applied to the SMPL-X rest orientation.

python retarget_rokoko_t1.py <npz_dir> <out_dir> [--height 1.75]
"""
import argparse
import copy
import glob
import json
import os
from multiprocessing import Pool

import numpy as np
from scipy.spatial.transform import Rotation as R

MAP = {  # GMR smplx name -> Rokoko bone
    "pelvis": "Hips", "spine3": "Spine4",
    "left_hip": "LeftThigh", "right_hip": "RightThigh",
    "left_knee": "LeftShin", "right_knee": "RightShin",
    "left_foot": "LeftFoot", "right_foot": "RightFoot",
    "left_shoulder": "LeftArm", "right_shoulder": "RightArm",
    "left_elbow": "LeftForeArm", "right_elbow": "RightForeArm",
    "left_wrist": "LeftHand", "right_wrist": "RightHand",
}
REST = R.from_matrix(np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]))


def make_config(out_path):
    from general_motion_retargeting.params import IK_CONFIG_DICT
    cfg = json.load(open(IK_CONFIG_DICT["smplx"]["booster_t1"]))
    # Paddle hand must follow the human wrist (not the elbow) so the stroke shape survives.
    for tbl, (pw, rw) in (("ik_match_table1", (20, 2)), ("ik_match_table2", (40, 2))):
        for side in ("left", "right"):
            e = cfg[tbl][f"{side}_hand_link"]
            e[0] = f"{side}_wrist"
            e[1], e[2] = pw, rw
    for side in ("left", "right"):
        cfg["human_scale_table"][f"{side}_wrist"] = cfg["human_scale_table"].pop(f"{side}_elbow")
    json.dump(cfg, open(out_path, "w"), indent=1)
    return out_path


def frames_to_human(d):
    bones = list(d["bones"])
    pos, quat = d["pos"], d["quat"]  # quat wxyz
    out = []
    g0 = {k: R.from_quat(quat[0, bones.index(v)], scalar_first=True) for k, v in MAP.items()}
    for t in range(1, len(pos)):
        fr = {}
        for k, v in MAP.items():
            j = bones.index(v)
            g = R.from_quat(quat[t, j], scalar_first=True)
            fr[k] = (pos[t, j].astype(np.float64), ((g * g0[k].inv()) * REST).as_quat(scalar_first=True))
        out.append(fr)
    return out


def run(args):
    path, out_dir, cfg_path, height = args
    import general_motion_retargeting.params as P
    from general_motion_retargeting import GeneralMotionRetargeting
    P.IK_CONFIG_DICT["smplx"]["booster_t1"] = cfg_path
    d = np.load(path)
    fps = float(d["fps"])
    frames = frames_to_human(d)
    gmr = GeneralMotionRetargeting("smplx", "booster_t1", actual_human_height=height, verbose=False)
    import mujoco as mj
    model = gmr.model
    jnames = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, model.dof_jntid[i]) for i in range(6, model.nv)]
    qs, err, hand = [], [], []
    data = mj.MjData(model)
    rh = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "right_hand_link")
    tr = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "Trunk")
    for fr in frames:
        q = gmr.retarget(copy.deepcopy(fr))
        qs.append(q.copy())
        err.append(gmr.error2())
        data.qpos[:] = q
        mj.mj_kinematics(model, data)
        hand.append(np.concatenate([data.xpos[rh], data.xpos[tr], data.xquat[tr]]))
    qs = np.array(qs, np.float32)
    stem = os.path.splitext(os.path.basename(path))[0]
    np.savez_compressed(os.path.join(out_dir, stem + "_t1.npz"), qpos=qs, fps=fps, joint_names=np.array(jnames),
                        ik_err=np.array(err, np.float32), hand_trunk=np.array(hand, np.float32))
    return stem, len(qs), float(np.mean(err)), float(np.percentile(err, 95))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("npz_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--height", type=float, default=1.75)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    cfg_path = make_config(os.path.join(a.out_dir, "rokoko_to_t1.json"))
    files = sorted(glob.glob(os.path.join(a.npz_dir, "Clip*.npz")))
    if a.limit:
        files = files[: a.limit]
    with Pool(len(files)) as p:
        for r in p.imap_unordered(run, [(f, a.out_dir, cfg_path, a.height) for f in files]):
            print("RETARGETED", r, flush=True)
