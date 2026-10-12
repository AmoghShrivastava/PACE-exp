# Results

| Folder / file | Contents |
|---|---|
| `official_eval/` | Logs from PACE's unmodified `eval.py` (final policy on long / mid / short / mixed serves; plain PACE at iteration 3,300 on mixed). `official_eval.txt` summarises them. |
| `evals/` | Per-checkpoint test results from `legged_lab/scripts/eval_headless.py` (2,500 serves each): hit rate, return rate, human-likeness, jerk, power. |
| `training_curves.json` | Training hit / return rates logged every 100 iterations. |
| `early_experiments/` | The first round of runs with PACE's 5 ms "fast" physics, see `docs/early_report_2026-10-05.md`. |

Run names:

| Name | Run |
|---|---|
| `Aauth2_*` | Plain PACE on our build, 2 ms physics, seed 42 |
| `Dauth_*`, `Dres_*` | PACE + swing-gated stroke bonus, 2 ms physics, seed 42 (`Dres` = resumed stretch to 7,800; `Dauth_10000` = final checkpoint) |
| `D7auth_*` | Same as `Dauth`, seed 7 |
| `D42_400`, `D7_400` | Early 400-iteration checks of the two seeds |
| `phys_*` | Physics checks comparing PACE's 2 ms default and its 5 ms "fast" setting |
| `A_*`, `B_*`, `C_*`, `C2_*` (early) | Plain PACE / always-on mocap prior / mocap + VAE prior / VAE prior at weight 0.75, fast physics |
