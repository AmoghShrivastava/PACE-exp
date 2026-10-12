# PACE + HIM Arena table-tennis data — fine-tuning report (2026-10-05)

## 1. What HIM Arena has (release `him-table-tennis-v1`, captured 2026-08-15)

| Asset | Size | Content | Usefulness for PACE |
|---|---|---|---|
| Rokoko mocap (FBX, 8 clips) | 194 MB | 61-bone full-body human, 30 fps, ~33 min of right-handed rally play. Frame 0 = T-pose. | **High.** PACE uses no human motion. Gave **1,040 forehand strokes** (+1 backhand-side outliers filtered). |
| WitMotion paddle IMU (10 sessions) | 10 MB | accel / gyro / euler / mag at **~11 Hz**, no quaternions, no chip time | Low. Too slow for a ~150 ms stroke. Used only as a cross-check: 887 swings over 34 min (25–30/min), consistent with mocap. |
| Orbbec RGB-D (8 sessions, 2 cams) | 19.4 GB | 640×480 colour + aligned depth (mm), 30/60 fps, two overhead cameras (one per table half), fiducials on table corners | Medium (future). Calibratable, so it could give real ball trajectories for PACE's drag/restitution and serve distribution. Not extracted here (needs ball detection and triangulation). |

The IMU and RGB-D session timestamps line up (e.g. IMU `214836` ↔ RGB-D `214831`), so this is one synchronised multi-sensor capture.

## 2. What was built ("PACE-MP" = PACE + human motion prior)

**Pipeline:** `pace_ft/tools/`
1. `fbx_to_npz.py`: headless Blender 4.2 dumps global joint positions and quaternions.
2. `retarget_rokoko_t1.py`: Rokoko → SMPL-X convention (T-pose deltas × R_x(90°)), then **GMR / mink IK onto the Booster T1**. The paddle hand tracks the human wrist. All 8 clips were retargeted (65k frames).
3. `build_stroke_library.py` builds two libraries:
   - **Stroke library:** 1.08 s windows (±0.54 s around peak hand speed, as in SMASH) resampled to 50 Hz, 9 upper-body joints. Each stroke is keyed by the T1 paddle contact point in the trunk-yaw frame, computed with FK.
   - **SMASH-style conditional motion VAE:**
     - Encoder q(z | X, c); decoder p(X | z, c), with z of dimension 16 and c = [contact key, initial pose].
     - Cyclic KL annealing plus a smoothness loss.
     - 15k samples drawn from a broadened key distribution, then filtered for joint limits, jerk and FK key-consistency: **3,404 kept**, giving a library of **4,444 strokes** (SMASH used about 5.4k).
     - Grid coverage within 5 cm: 6.4% → 10.5%.
     - The VAE fills **higher contact heights**, which is where a 1.2 m T1 meets a post-bounce apex (see `stroke_library.png`).
4. `pace_repo/legged_lab/envs/base/tt_env_mp.py` and `mdp.reward_motion_prior`:
   - **Motion matching** (SMASH): each step, pick the stroke whose key is nearest the physics oracle's hitting point.
   - **Phase alignment** (HITTER t_strike): reference index = contact − time-to-apex/dt; the follow-through plays out after contact.
   - **Reward:** exp(−Σ w_j (q − q_ref)² / σ²) with weight 3.0. **Elbow-yaw is excluded**, the analogue of SMASH freeing the wrist.
   - **The reference goes to the critic only**, so the deployed actor and predictor have exactly PACE's inputs. No sim-to-real change.

**Fixes needed to run PACE at all:**
- The code needs **IsaacLab 2.2.0** (it uses `quat_apply_inverse` and `is_global`), not the 2.1.0 the README says. It ran on IsaacSim 4.5 with torch 2.5.1/cu121.
- `aerodynamics.py` got a fallback for older APIs.

## 3. Experiments (HIM A40 48 GB, one seed each)

| Run | Description |
|---|---|
| A | PACE baseline (predictor runner) |
| B | PACE-MP, mocap-only library (1,040 strokes) |
| C | PACE-MP, mocap + VAE library (4,444 strokes) |
| C2 | C resumed at iteration 300 with imitation weight 0.75 (instead of 3.0) |

**Setup:**
- 4,096 envs and PACE's PPO settings.
- **PACE's own "fast validation" sim** (dt 0.005 × 4, still a 50 Hz policy) instead of 0.002 × 10, to reach about 1,000 iterations in the time window. PACE trains 10,000.
- Eval:
  - **Conditions:** PACE's `t1_tt_eval` serve box, observation noise on, 1,000 serves (2,500 at iteration 1,000).
  - **Hit and return:** the same hit/return logic as PACE's `eval.py`.
  - **Human-likeness:** RMS distance of the ±0.54 s arm window around contact to the nearest human stroke, measured against the same mocap library for every run.

| iter | run | hit rate | return success | human-likeness ↓ (rad) | arm jerk ↓ | power (W) | contact→nearest key (mocap / +VAE) |
|---|---|---|---|---|---|---|---|
| 400 | A | 3.5% | 0 | 0.483 | 1061 | 21.8 | 12.0 / 3.0 cm |
| 400 | B | **12.1%** | 0 | 0.417 | 1055 | 21.5 | 8.1 / 3.5 cm |
| 400 | C | 10.8% | 0 | 0.423 | 1114 | 26.4 | 7.5 / 4.0 cm |
| 800 | A | 1.8% | 0 | 0.452 | 1055 | 24.2 | 9.4 / 3.6 cm |
| 800 | B | 0.4% | 0 | 0.317 | 945 | 24.5 | 4.3 / 1.4 cm |
| 800 | C | 2.5% | 0 | **0.306** | 954 | 24.3 | 5.1 / 1.6 cm |
| 1000 | A | 0.3% | 0 | 0.497 | 988 | 23.9 | 10.9 / 5.6 cm |
| 1000 | B | 0.4% | 0 | 0.309 | 928 | 24.6 | 8.7 / 2.5 cm |
| 1000 | C | **1.4%** | 0 | **0.304** | **926** | 25.7 | 8.3 / 2.1 cm |
| 500 | C (w=3) | 5.3% | 0 | 0.402 | 1060 | 21.1 | — |
| 500 | C2 (w=0.75) | 5.8% | 0 | 0.430 | 1049 | 28.4 | — |

Training curves are in `artifacts/training_curves.png` and `artifacts/training_curves.csv`.

## 4. Assessment — what improved, what didn't

**Robust improvements:**
- **Much more human-like strokes.** The arm trajectory around contact is **39% closer** to real human forehands (0.30 vs 0.50 rad RMS at iteration 1,000). The gain grows with training, and mechanical power is unchanged.
- **Smoother arm motion.** Arm jerk is 6–10% lower at iterations 800–1,000.
- **VAE augmentation fixes library coverage**, as in SMASH. The contact points the robot actually needs are 2–4 cm from a VAE-augmented stroke versus 8–12 cm from raw mocap. Human casual play hits low; the T1 needs higher contacts.

**Suggestive, single seed:**
- **Faster early hitting.** Both prior runs peak higher in training contact reward (about 0.024 vs 0.020) and are 3× better in eval hits at iteration 400 (11–12% vs 3.5%).
  - Caveat: A's iteration-200 checkpoint scored 13.5% in a smaller spot check (401 serves), and A's curve dips right after its resume. Its 400 number is probably depressed, so treat the 3× as an upper bound.
- **At iteration 1,000, C (with VAE) hits more than A or B:** 35/2,501 vs 8–10/2,501.

**Not achieved / negative:**
- **No policy returns balls yet.** All success rates are 0, and hit rates are low. At about 10% of PACE's training budget this is expected. The paper's 96% hit / 92% success at 10k iterations was not reproduced or compared against.
- **Reward competition after iteration ~600.** The imitation reward grows super-linearly while contact and return reward fall faster than the baseline's. The baseline keeps a small return signal through iteration 1,000; B and C lose it. Cutting the weight to 0.75 (C2) made no measurable difference by iteration 500 (5.8% vs 5.3%, within noise), so the right fix is still open.
- **Caveats:**
  - One seed per run.
  - Eval hit rates fluctuate a lot between checkpoints this early.
  - A and B were resumed from iteration 200 after a network-filesystem I/O crash, at equal-iteration comparison.
  - Fast-sim physics was used.

## 5. Recommended next steps

1. **Gate the imitation reward** instead of running it as a dense, always-on term:
   - apply it only in a window around t_strike, as SMASH does (±0.1 s);
   - or multiply it by the hit reward;
   - or anneal its weight once contact rate exceeds about 50%.
2. **Fine-tune from a converged PACE checkpoint** (the 10k-iteration baseline) with a low imitation weight, rather than training from scratch. Use the full 0.002 × 10 sim and at least 3 seeds. The code supports this: `--resume True --load_run … --task t1_tt_mp`.
3. **Use the RGB-D sessions** to extract real ball trajectories:
   - calibrate the cameras from the table fiducials, detect and triangulate the ball;
   - feed the result into PACE's drag/restitution fit and its serve distribution, and fine-tune the learned predictor on real arcs (the predictor's real-world error is unmeasured in the paper).
4. **Backhand:** the mocap has almost none (one stroke), so a SMASH-style backhand prior would need new captures.

## Files
- `pace_ft/tools/*.py`: data and library pipeline.
- `pace_repo/`: patched PACE (`git diff`), with tasks `t1_tt_mp` and `t1_tt_mp_eval`, and `scripts/eval_headless.py`.
- `pace_ft/artifacts/`:
  - `stroke_lib/` (libraries + `motion_vae.pt`), `t1_retarget/` (retargeted T1 motions);
  - `ckpt/` (A/B/C at iteration 1,000, C2 at 500), `results/` (eval JSONs and logs), `tb/` (TensorBoard), `training_curves.*`.
- Reproduce a run (needs a 4.5 + 2.2 env, `PACE_FAST_SIM=1` for the fast setting):
  `PACE_MP_LIB=…/stroke_lib_vae.npz python -m legged_lab.scripts.train --task=t1_tt_mp --predictor --headless --num_envs=4096`
