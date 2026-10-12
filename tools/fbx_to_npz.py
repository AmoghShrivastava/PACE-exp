"""Headless Blender: dump each FBX armature to global joint positions/quaternions.

blender -b --python fbx_to_npz.py -- <in_dir> <out_dir>
"""
import json
import os
import sys

import bpy
import numpy as np

argv = sys.argv[sys.argv.index("--") + 1:]
in_dir, out_dir = argv[0], argv[1]
os.makedirs(out_dir, exist_ok=True)

summary = {}
for name in sorted(os.listdir(in_dir)):
    if not name.lower().endswith(".fbx"):
        continue
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=os.path.join(in_dir, name), automatic_bone_orientation=False)
    scene = bpy.context.scene
    arms = [o for o in scene.objects if o.type == "ARMATURE"]
    if not arms:
        summary[name] = {"error": "no armature", "objects": [o.name for o in scene.objects]}
        continue
    arm = max(arms, key=lambda a: len(a.pose.bones))
    action = arm.animation_data.action if arm.animation_data else None
    f0, f1 = (int(action.frame_range[0]), int(action.frame_range[1])) if action else (scene.frame_start, scene.frame_end)
    fps = scene.render.fps / scene.render.fps_base
    bones = [b.name for b in arm.pose.bones]
    parents = [bones.index(b.parent.name) if b.parent else -1 for b in arm.pose.bones]
    T = f1 - f0 + 1
    pos = np.zeros((T, len(bones), 3), np.float32)
    quat = np.zeros((T, len(bones), 4), np.float32)  # wxyz, world frame
    mw = arm.matrix_world
    for t, f in enumerate(range(f0, f1 + 1)):
        scene.frame_set(f)
        for j, pb in enumerate(arm.pose.bones):
            m = mw @ pb.matrix
            pos[t, j] = m.translation
            q = m.to_quaternion()
            quat[t, j] = (q.w, q.x, q.y, q.z)
    stem = os.path.splitext(name)[0]
    np.savez_compressed(os.path.join(out_dir, stem + ".npz"), pos=pos, quat=quat, fps=fps,
                        bones=np.array(bones), parents=np.array(parents))
    summary[name] = {"armature": arm.name, "frames": T, "fps": fps, "n_bones": len(bones),
                     "bones": bones, "scale": list(arm.scale),
                     "other_objects": [o.name for o in scene.objects if o != arm][:20]}
    print("DONE", name, T, fps, len(bones), flush=True)

with open(os.path.join(out_dir, "summary.json"), "w") as fh:
    json.dump(summary, fh, indent=1)
