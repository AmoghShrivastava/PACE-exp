"""Controlled paddle-collision test: does the ball physically bounce off the paddle?

The robot holds its default pose (zero actions). Every 0.4 s each env's ball is placed 0.25 m in
front of the paddle touch point (+x, toward the opponent) with a small random offset and fired at
it along -x. 0.12 s later we check whether the ball's x-velocity reversed (a real collision) or it
kept going (passed through / missed). Run once with PACE_FAST_SIM=1 (5 ms x 4) and once without
(PACE's 2 ms x 10).
"""
import argparse
import json
import os

import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--speed", type=float, default=6.0)
parser.add_argument("--offset", type=float, default=0.02, help="max lateral offset from paddle centre (m)")
parser.add_argument("--shots", type=int, default=6)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--out", type=str, required=True)
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True
app = AppLauncher(args).app

import isaaclab.utils.math as math_utils  # noqa: E402

from legged_lab.envs import *  # noqa: F401,F403,E402
from legged_lab.utils import task_registry  # noqa: E402


def main():
    env_cfg, _ = task_registry.get_cfgs("t1_tt_eval")
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.domain_rand.events.push_robot = None
    env_cfg.noise.add_noise = True  # PACE crashes with noise off; actions are zero so it does not matter
    env_cfg.scene.seed = 11
    env = task_registry.get_task_class("t1_tt_eval")(env_cfg, True)
    N, dev = env.num_envs, env.device
    act = torch.zeros(N, env.num_actions, device=dev)
    for _ in range(50):  # settle
        env.step(act)
    shot_steps, check_steps = int(0.4 / env.step_dt), int(0.12 / env.step_dt)
    res = []
    ids = torch.arange(N, device=dev)
    for s in range(args.shots):
        tp = env.paddle_touch_point.clone()
        off = (torch.rand(N, 3, device=dev) * 2 - 1) * args.offset
        off[:, 0] = 0.0
        pos = tp + off + torch.tensor([0.25, 0.0, 0.0], device=dev)
        st = torch.zeros(N, 13, device=dev)
        st[:, :3] = pos
        st[:, 3] = 1.0
        st[:, 7] = -args.speed
        env.ball.write_root_pose_to_sim(st[:, :7], ids)
        env.ball.write_root_velocity_to_sim(st[:, 7:], ids)
        env.ball_episode_length_buf[:] = 0
        minds = torch.full((N,), 9.0, device=dev)
        for k in range(shot_steps):
            env.step(act)
            minds = torch.minimum(minds, (env.ball.data.root_pos_w - env.paddle_touch_point).norm(dim=-1))
            if k == check_steps:
                vx = env.ball.data.root_lin_vel_w[:, 0].clone()
        res.append({"bounced": (vx > 0).float().mean().item(), "reached": (minds < 0.08).float().mean().item(),
                    "vx_after_mean": vx.mean().item()})
        print(f"[phys] shot {s}: bounced {res[-1]['bounced']:.2f} reached-within-8cm {res[-1]['reached']:.2f}", flush=True)
    out = {"fast_sim": os.environ.get("PACE_FAST_SIM", "0") == "1", "dt": env.cfg.sim.dt, "decimation": env.cfg.sim.decimation,
           "speed": args.speed, "bounced": sum(r["bounced"] for r in res) / len(res),
           "reached": sum(r["reached"] for r in res) / len(res), "vx_after": sum(r["vx_after_mean"] for r in res) / len(res)}
    json.dump(out, open(args.out, "w"), indent=1)
    print("[phys] RESULT", json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
    os._exit(0)
