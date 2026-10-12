"""Record a policy rollout in Isaac Sim (headless, RTX offscreen) from two cameras.

Writes an MP4 (side view | behind-the-player view) at 0.5x speed with a live counter of
serves / hits / returns for the filmed environment.
"""
import argparse
import os

import numpy as np
import torch
from isaaclab.app import AppLauncher

import legged_lab.utils.cli_args as cli_args  # isort: skip

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="t1_tt_eval")
parser.add_argument("--seconds", type=float, default=16.0)
parser.add_argument("--label", type=str, default="")
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--width", type=int, default=640)
parser.add_argument("--height", type=int, default=400)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import imageio.v2 as imageio  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from legged_lab.envs import *  # noqa: F401,F403,E402
from legged_lab.utils import task_registry  # noqa: E402
from legged_lab.utils.cli_args import update_rsl_rl_cfg  # noqa: E402
from rsl_rl.rsl_rl.runners import OnPolicyPredictorRegressionRunner  # noqa: E402


def main():
    env_cfg, agent_cfg = task_registry.get_cfgs(args_cli.task)
    env_cfg.scene.num_envs = 4
    env_cfg.scene.env_spacing = 6
    env_cfg.domain_rand.events.push_robot = None
    env_cfg.noise.add_noise = True
    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.seed = 7
    env = task_registry.get_task_class(args_cli.task)(env_cfg, True)

    log_root = os.path.abspath(os.path.join("logs", agent_cfg.experiment_name))
    resume_path = get_checkpoint_path(log_root, args_cli.load_run, args_cli.checkpoint)
    runner = OnPolicyPredictorRegressionRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path, load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device)

    o = env.scene.env_origins[0].cpu().numpy()
    cams = [
        rep.create.camera(position=tuple(o + [-1.1, -3.4, 1.55]), look_at=tuple(o + [-1.3, 0.1, 0.85])),
        rep.create.camera(position=tuple(o + [-4.4, 0.9, 1.75]), look_at=tuple(o + [0.4, -0.1, 0.8])),
    ]
    annots = []
    for c in cams:
        rp = rep.create.render_product(c, (args_cli.width, args_cli.height))
        a = rep.AnnotatorRegistry.get_annotator("rgb")
        a.attach([rp])
        annots.append(a)
    for _ in range(10):  # warm up the renderer
        env.sim.render()

    font = ImageFont.load_default()
    writer = imageio.get_writer(args_cli.out, fps=25, codec="libx264", quality=7, macro_block_size=8)
    serves = hits = rets = 0
    hit_flag = ret_flag = False
    obs, _ = env.get_observations()
    n_steps = int(args_cli.seconds / env.step_dt)
    for t in range(n_steps):
        with torch.inference_mode():
            obs, _, _, _ = env.step(policy(obs))
            runner._record_ball_positions()
            runner._maybe_predict_and_update_env()
        hit_flag |= bool(env.ball_contact_rew[0] > 0)
        ret_flag |= bool(env.has_touch_opponent_table_just_now[0] & env.has_touch_paddle[0])
        if (env.ball_reset_ids == 0).any():
            serves += 1; hits += int(hit_flag); rets += int(ret_flag)
            hit_flag = ret_flag = False
        env.sim.render()
        frames = []
        for a in annots:
            d = a.get_data()
            if d is None or len(d) == 0:
                rep.orchestrator.step(rt_subframes=1)
                d = a.get_data()
            frames.append(np.asarray(d)[..., :3])
        img = Image.fromarray(np.concatenate(frames, axis=1).astype(np.uint8))
        dr = ImageDraw.Draw(img)
        dr.rectangle([0, 0, img.width, 22], fill=(0, 0, 0))
        dr.text((8, 5), f"{args_cli.label}   t={t * env.step_dt:5.2f}s   serves {serves}   hits {hits}   returns {rets}   (0.5x speed)",
                fill=(255, 255, 255), font=font)
        writer.append_data(np.asarray(img))
    writer.close()
    print(f"[video] wrote {args_cli.out}: serves {serves} hits {hits} returns {rets}", flush=True)


if __name__ == "__main__":
    main()
    os._exit(0)
