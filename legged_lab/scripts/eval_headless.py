"""Headless, bounded evaluation for PACE / PACE-MP checkpoints.

Counts serves, hits and successful returns exactly like scripts/eval.py, and additionally
logs motion-quality metrics (SMASH-style): arm jerk, action rate, mechanical power, paddle
speed at contact, outgoing ball speed, and distance of the arm trajectory to the human
stroke library ("human-likeness", evaluated for every policy with the same library).
"""
import argparse
import json
import os

import numpy as np
import torch
from isaaclab.app import AppLauncher

import legged_lab.utils.cli_args as cli_args  # isort: skip

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="t1_tt_eval")
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--serves", type=int, default=4000)
parser.add_argument("--serve_range", type=str, default="mixed", choices=["mixed", "train"])
parser.add_argument("--human_lib", type=str, default="")
parser.add_argument("--aug_lib", type=str, default="", help="mocap+VAE library, for contact-point coverage")
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--seed", type=int, default=None)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402

from legged_lab.envs import *  # noqa: F401,F403,E402
from legged_lab.utils import task_registry  # noqa: E402
from legged_lab.utils.cli_args import update_rsl_rl_cfg  # noqa: E402
from rsl_rl.rsl_rl.runners import OnPolicyPredictorRegressionRunner  # noqa: E402

SERVE_RANGES = {  # "mixed" = PACE's t1_tt_eval serve box; "train" = the wider training box
    "mixed": ((-6.5, -5.2), (-0.6, 0.2), (1.5, 1.9)),
    "train": ((-6.5, -5.0), (-0.8, 0.4), (1.5, 2.0)),
}
UPPER = ["Right_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw", "Waist",
         "Left_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw"]


def main():
    env_cfg, agent_cfg = task_registry.get_cfgs(args_cli.task)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.env_spacing = 5
    env_cfg.domain_rand.events.push_robot = None
    env_cfg.noise.add_noise = True
    vx, vy, vz = SERVE_RANGES[args_cli.serve_range]
    env_cfg.ball.ball_speed_x_range, env_cfg.ball.ball_speed_y_range, env_cfg.ball.ball_speed_z_range = vx, vy, vz
    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.seed = 1234
    env = task_registry.get_task_class(args_cli.task)(env_cfg, True)

    log_root = os.path.abspath(os.path.join("logs", agent_cfg.experiment_name))
    resume_path = get_checkpoint_path(log_root, args_cli.load_run, args_cli.checkpoint)
    runner = OnPolicyPredictorRegressionRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path, load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device)

    N, dev, dt = env.num_envs, env.device, env.step_dt
    up_ids, _ = env.robot.find_joints(UPPER, preserve_order=True)
    lib = None
    if args_cli.human_lib:
        L = np.load(args_cli.human_lib)
        lib = torch.tensor(L["ref_q"], device=dev, dtype=torch.float)  # [M,H,J]
        C = int(L["contact_idx"])
        lib_key = torch.tensor(L["key"], device=dev, dtype=torch.float)
    aug_key = None
    if args_cli.aug_lib:
        aug_key = torch.tensor(np.load(args_cli.aug_lib)["key"], device=dev, dtype=torch.float)
    H = 0 if lib is None else lib.shape[1]
    hist = torch.zeros(N, max(H, 1), len(UPPER), device=dev)  # rolling arm history for human-likeness

    succ = torch.zeros(N, dtype=torch.bool, device=dev)
    hit = torch.zeros(N, dtype=torch.bool, device=dev)
    prev_touch = torch.zeros(N, dtype=torch.bool, device=dev)
    pending = []  # (env_id, steps_after_contact) for human-likeness windows
    m = dict(serves=0, hits=0, succ=0, falls=0, jerk=[], act_rate=[], power=[], paddle_speed=[], ball_out_speed=[],
             human_dist=[], land_err=[], cover_mocap=[], cover_aug=[])
    prev_paddle = env.paddle_touch_point.clone()
    prev_act = torch.zeros(N, env.num_actions, device=dev)
    qv_hist = [env.robot.data.joint_vel[:, up_ids].clone() for _ in range(2)]
    warm = 2 * N
    obs, _ = env.get_observations()
    steps = 0
    while m["serves"] < args_cli.serves and simulation_app.is_running():
        with torch.inference_mode():
            act = policy(obs)
            obs, _, dones, _ = env.step(act)
            runner._record_ball_positions()
            runner._maybe_predict_and_update_env()
            steps += 1
            qv = env.robot.data.joint_vel[:, up_ids]
            # jerk of arm joints from finite differences of velocity (rad/s^3)
            jerk = ((qv - 2 * qv_hist[-1] + qv_hist[-2]) / dt ** 2).abs().mean(-1)
            qv_hist = [qv_hist[-1], qv.clone()]
            m["jerk"].append(jerk.mean().item())
            m["act_rate"].append(((act - prev_act) ** 2).sum(-1).mean().item())
            prev_act = act.clone()
            tau = env.robot.data.applied_torque[:, env.action_joint_ids]
            m["power"].append((tau * env.robot.data.joint_vel[:, env.action_joint_ids]).abs().sum(-1).mean().item())
            hist = torch.roll(hist, -1, dims=1)
            hist[:, -1] = env.robot.data.joint_pos[:, up_ids]
            pspeed = (env.paddle_touch_point - prev_paddle).norm(dim=-1) / dt
            prev_paddle = env.paddle_touch_point.clone()

            contact_now = env.has_touch_paddle & ~prev_touch
            prev_touch = env.has_touch_paddle.clone()
            if contact_now.any():
                ids = contact_now.nonzero().squeeze(-1)
                m["paddle_speed"] += pspeed[ids].tolist()
                # actual contact point in the trunk-yaw frame vs. nearest stroke-library key
                rel = env.paddle_touch_point[ids] - env.robot.data.root_link_pos_w[ids]
                yaw = env.robot.data.heading_w[ids]
                c, s_ = torch.cos(-yaw), torch.sin(-yaw)
                key = torch.stack([c * rel[:, 0] - s_ * rel[:, 1], s_ * rel[:, 0] + c * rel[:, 1], rel[:, 2]], -1)
                if lib is not None:
                    m["cover_mocap"] += torch.cdist(key, lib_key).min(1)[0].tolist()
                if aug_key is not None:
                    m["cover_aug"] += torch.cdist(key, aug_key).min(1)[0].tolist()
                pending += [[int(i), 0] for i in ids]
            # ball outgoing speed 3 steps after contact; human-likeness once the follow-through window is full
            nxt = []
            for p in pending:
                p[1] += 1
                if p[1] == 3:
                    m["ball_out_speed"].append(env.ball_linvel[p[0]].norm().item())
                if lib is not None and p[1] == H - 1 - C:
                    w = hist[p[0]]  # contact aligned to index C
                    m["human_dist"].append(((lib - w[None]) ** 2).mean(dim=(1, 2)).min().sqrt().item())
                if p[1] < max(3, H - C):
                    nxt.append(p)
            pending = nxt[-20000:]

            succ |= env.has_touch_opponent_table_just_now & env.has_touch_paddle
            hit |= env.ball_contact_rew > 0.0
            ids = env.ball_reset_ids
            if ids.numel() > 0:
                skip = min(warm, ids.numel())
                warm -= skip
                take = ids[skip:]
                m["serves"] += int(take.numel())
                m["hits"] += int(hit[take].sum())
                m["succ"] += int(succ[take].sum())
                succ[ids] = False
                hit[ids] = False
            fell = dones & ~env.time_out_buf
            m["falls"] += int(fell.sum())
        if steps % 500 == 0:
            print(f"[eval] steps {steps} serves {m['serves']} hit {m['hits']/max(1,m['serves']):.3f} succ {m['succ']/max(1,m['serves']):.3f}", flush=True)

    res = dict(checkpoint=resume_path, task=args_cli.task, serve_range=args_cli.serve_range, serves=m["serves"],
               hit_rate=m["hits"] / max(1, m["serves"]), success_rate=m["succ"] / max(1, m["serves"]),
               falls_per_1k_steps=1000 * m["falls"] / (steps * N))
    for k in ("jerk", "act_rate", "power", "paddle_speed", "ball_out_speed", "human_dist", "cover_mocap", "cover_aug"):
        v = np.array(m[k]) if len(m[k]) else np.array([np.nan])
        res[k + "_mean"] = float(np.nanmean(v))
        res[k + "_median"] = float(np.nanmedian(v))
    os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
    json.dump(res, open(args_cli.out, "w"), indent=1)
    print("[eval] RESULT", json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
    # Isaac Sim 4.5 can hang in simulation_app.close() after headless runs; results are already on disk.
    os._exit(0)
