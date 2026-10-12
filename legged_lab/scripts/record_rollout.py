"""Run a policy in Isaac Sim (headless) and log the full simulation state at 50 Hz for replay.

The pod has no graphics capability for RTX cameras, so the video is made by replaying these
exact states in MuJoCo (scripts/render_rollout.py). Logged per step and per env: robot root pose,
every joint angle (by name), ball position, paddle touch point, hit / return / serve events.
"""
import argparse
import os

import numpy as np
import torch
from isaaclab.app import AppLauncher

import legged_lab.utils.cli_args as cli_args  # isort: skip

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="t1_tt_eval")
parser.add_argument("--seconds", type=float, default=24.0)
parser.add_argument("--num_envs", type=int, default=8)
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


def main():
    env_cfg, agent_cfg = task_registry.get_cfgs(args_cli.task)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.env_spacing = 6
    env_cfg.domain_rand.events.push_robot = None
    env_cfg.noise.add_noise = True
    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.seed = 7  # identical serve sequence for every policy
    env = task_registry.get_task_class(args_cli.task)(env_cfg, True)
    resume_path = get_checkpoint_path(os.path.abspath(os.path.join("logs", agent_cfg.experiment_name)),
                                      args_cli.load_run, args_cli.checkpoint)
    runner = OnPolicyPredictorRegressionRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path, load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device)

    org = env.scene.env_origins
    log = {k: [] for k in ("root_pos", "root_quat", "q", "ball", "paddle", "hit", "ret", "reset")}
    obs, _ = env.get_observations()
    for _ in range(int(args_cli.seconds / env.step_dt)):
        with torch.inference_mode():
            obs, _, _, _ = env.step(policy(obs))
            runner._record_ball_positions()
            runner._maybe_predict_and_update_env()
            reset = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            reset[env.ball_reset_ids] = True
            log["root_pos"].append((env.robot.data.root_link_pos_w - org).cpu().numpy())
            log["root_quat"].append(env.robot.data.root_link_quat_w.cpu().numpy())
            log["q"].append(env.robot.data.joint_pos.cpu().numpy())
            log["ball"].append((env.ball.data.root_pos_w - org).cpu().numpy())
            log["paddle"].append((env.paddle_touch_point - org).cpu().numpy())
            log["hit"].append((env.ball_contact_rew > 0).cpu().numpy())
            log["ret"].append((env.has_touch_opponent_table_just_now & env.has_touch_paddle).cpu().numpy())
            log["reset"].append(reset.cpu().numpy())
    np.savez_compressed(args_cli.out, dt=env.step_dt, joint_names=np.array(env.robot.joint_names),
                        checkpoint=resume_path, **{k: np.array(v) for k, v in log.items()})
    print(f"[rollout] wrote {args_cli.out}", flush=True)


if __name__ == "__main__":
    main()
    os._exit(0)
