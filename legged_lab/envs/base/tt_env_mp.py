"""PACE + human motion prior (PACE-MP).

Adds a library of human forehand strokes (HIM Rokoko mocap retargeted to T1, optionally
augmented with a SMASH-style motion VAE). Each step, for the incoming ball, the stroke whose
contact-time paddle position (relative to the trunk, trunk-yaw frame) best matches the
physics oracle's hitting point is selected (SMASH motion matching), and its reference is
phase-aligned to the oracle's time-to-strike (HITTER-style t_strike alignment). After contact
the reference plays out the follow-through.

The reference is used only in the reward and the critic observation, so the deployed actor
keeps PACE's original observation space.
"""

import os

import numpy as np
import torch

from legged_lab.envs.base.tt_env import TTEnv


class TTEnvMP(TTEnv):
    def __init__(self, cfg, headless):
        lib_path = os.environ.get("PACE_MP_LIB", getattr(cfg, "motion_lib_path", ""))
        lib = np.load(lib_path)
        self.mp_names = [str(n) for n in lib["joint_names"]]
        self._mp_ref = lib["ref_q"]
        self._mp_key = lib["key"]
        self.mp_C = int(lib["contact_idx"])
        self.mp_H = self._mp_ref.shape[1]
        self.mp_lib_size = self._mp_ref.shape[0]
        print(f"[TTEnvMP] motion library {lib_path}: {self.mp_lib_size} strokes, H={self.mp_H}, contact={self.mp_C}")
        super().__init__(cfg, headless)

    def init_buffers(self):
        dev = self.device
        self.mp_ref = torch.tensor(self._mp_ref, device=dev, dtype=torch.float)  # [M,H,J]
        self.mp_key = torch.tensor(self._mp_key, device=dev, dtype=torch.float)  # [M,3]
        n = self.cfg.scene.num_envs
        self.mp_clip = torch.zeros(n, dtype=torch.long, device=dev)
        self.mp_idx = torch.zeros(n, dtype=torch.long, device=dev)
        self.mp_active = torch.zeros(n, dtype=torch.bool, device=dev)
        self.mp_post = torch.zeros(n, dtype=torch.bool, device=dev)
        self.mp_key_err = torch.zeros(n, device=dev)
        self.mp_tau = torch.zeros(n, device=dev)  # signed time to contact (s): <0 before the strike, >0 follow-through
        self.mp_ref_now = torch.zeros(n, len(self.mp_names), device=dev)
        self.mp_ref_next = torch.zeros(n, len(self.mp_names), device=dev)
        super().init_buffers()
        self.mp_joint_ids, _ = self.robot.find_joints(self.mp_names, preserve_order=True)
        self.mp_ref_now[:] = self.robot.data.default_joint_pos[:, self.mp_joint_ids]
        self.mp_ref_next[:] = self.mp_ref_now

    def reset_ball(self, env_ids):
        super().reset_ball(env_ids)
        if len(env_ids) == 0 or not hasattr(self, "mp_active"):
            return
        self.mp_active[env_ids] = False
        self.mp_post[env_ids] = False
        self.mp_idx[env_ids] = 0

    def _trunk_frame_key(self, target_local, ids):
        trunk = self.robot.data.root_link_pos_w[ids] - self.scene.env_origins[ids]
        rel = target_local - trunk
        yaw = self.robot.data.heading_w[ids]
        c, s = torch.cos(-yaw), torch.sin(-yaw)
        return torch.stack([c * rel[:, 0] - s * rel[:, 1], s * rel[:, 0] + c * rel[:, 1], rel[:, 2]], dim=-1)

    def compute_intermediate_values(self):
        touched_before = self.has_touch_paddle_rew.clone()
        super().compute_intermediate_values()
        pre = ~self.mask_invalid  # ball incoming, oracle valid, not yet hit
        contact_now = self.has_touch_paddle & ~touched_before

        if pre.any():
            ids = pre.nonzero(as_tuple=False).squeeze(-1)
            key_des = self._trunk_frame_key(self.ball_future_pose[ids], ids)
            d = torch.cdist(key_des, self.mp_key)  # [n, M]
            err, clip = d.min(dim=1)
            self.mp_clip[ids] = clip
            self.mp_key_err[ids] = err
            steps_to_hit = torch.round(self.ball_future_t[ids, 0] / self.step_dt).long()
            self.mp_idx[ids] = torch.clamp(self.mp_C - steps_to_hit, 0, self.mp_C)
            self.mp_tau[ids] = -steps_to_hit.float() * self.step_dt
            self.mp_active[ids] = True

        # follow-through: continue the matched stroke after contact
        self.mp_post |= contact_now & self.mp_active
        self.mp_idx[contact_now] = self.mp_C
        adv = self.mp_post & ~contact_now
        self.mp_idx[adv] = torch.clamp(self.mp_idx[adv] + 1, max=self.mp_H - 1)
        post = self.mp_post
        self.mp_tau[post] = (self.mp_idx[post] - self.mp_C).float() * self.step_dt
        # whiffed/expired strokes stop being imitated once the ball is gone
        self.mp_active &= pre | self.mp_post
        self.mp_active &= ~(self.mp_post & (self.mp_idx >= self.mp_H - 1))

        nxt = torch.clamp(self.mp_idx + 5, max=self.mp_H - 1)
        self.mp_ref_now = self.mp_ref[self.mp_clip, self.mp_idx]
        self.mp_ref_next = self.mp_ref[self.mp_clip, nxt]

    def compute_current_observations_perception(self):
        actor_obs, critic_obs = super().compute_current_observations_perception()
        q = self.robot.data.joint_pos[:, self.mp_joint_ids]
        act = self.mp_active.float().unsqueeze(-1)
        extra = torch.cat([(self.mp_ref_now - q) * act, (self.mp_ref_next - q) * act, act], dim=-1)
        return actor_obs, torch.cat([critic_obs, extra], dim=-1)
