"""Draw media/img/curves.png (training hit rate vs the PACE paper, test return rate over iterations)."""
import json, glob, os, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
A = "results/evals"; OUT = "media/img/"  # run from the repo root
os.makedirs(OUT, exist_ok=True)
L = json.load(open("results/training_curves.json"))
B, O, Q, G, T1, T2 = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": "#c9c8c3", "axes.labelcolor": T2,
                     "xtick.color": T2, "ytick.color": T2, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4), facecolor="#fcfcfb")
for a in ax: a.set_facecolor("#fcfcfb"); a.grid(axis="y", color="#e6e5e0", lw=0.8); a.set_axisbelow(True)
# left: training hit rate vs paper
d = [q for q in L["Dauth"]["hit"] if q[0] <= 3500] + [q for q in L["Dauth_resume"]["hit"] if q[0] > 3500]
paper = [[0, 0], [500, 0.02], [1000, 0.04], [1500, 0.07], [2000, 0.2], [2500, 0.38], [3000, 0.5], [3500, 0.58]]
series = [("PACE + human strokes (ours)", d, B), ("Plain PACE (our build)", L["Aauth2"]["hit"], O), ("PACE paper, Fig. 5", paper, G)]
for n, s, c in series:
    s = [q for q in s if q[0] <= 3300]
    ax[0].plot([q[0] for q in s], [100 * q[1] for q in s], color=c, lw=2, ls="--" if c == G else "-", label=n)
ax[0].set_title("Training hit rate, first 3,300 iterations", loc="left", color=T1, fontsize=12.5)
ax[0].set_xlabel("PPO iteration"); ax[0].set_ylabel("hit rate (%)"); ax[0].set_ylim(0, 100); ax[0].set_xlim(0, 3400)
ax[0].legend(frameon=False, loc="upper left", labelcolor=T2)
# right: test return rate (2,500 serves) for D*
pts = []
for f in glob.glob(A + "/Dauth_*.json") + glob.glob(A + "/Dres_*.json"):
    it = int(os.path.basename(f).split("_")[1][:-5]); r = json.load(open(f)); pts.append((it, r["hit_rate"], r["success_rate"]))
pts.sort()
a2 = sorted((int(os.path.basename(f).split("_")[1][:-5]), json.load(open(f))["success_rate"]) for f in glob.glob(A + "/Aauth2_*.json"))
ax[1].plot([p[0] for p in pts], [100 * p[2] for p in pts], color=B, lw=2, marker="o", ms=6, mec="#fcfcfb", mew=2, label="PACE + human strokes (ours)")
ax[1].plot([p[0] for p in a2], [100 * p[1] for p in a2], color=O, lw=2, marker="o", ms=6, mec="#fcfcfb", mew=2, label="Plain PACE (our build)")
ax[1].axhline(94.1, color=G, ls="--", lw=1.5); ax[1].text(150, 96.2, "PACE paper at 10k: 94.1%", color=T2, fontsize=10)
ax[1].annotate(f"{100*pts[-1][2]:.1f}%", (pts[-1][0], 100 * pts[-1][2]), xytext=(-8, -20), textcoords="offset points", ha="right", color=T1, fontsize=10.5)
ax[1].set_title("Test return rate (ball lands on the far side)", loc="left", color=T1, fontsize=12.5)
ax[1].set_xlabel("PPO iteration"); ax[1].set_ylabel("returned (%)"); ax[1].set_ylim(0, 100); ax[1].set_xlim(0, 10400)
ax[1].legend(frameon=False, loc="lower right", labelcolor=T2)
fig.tight_layout(); fig.savefig(OUT + "curves.png", dpi=150, facecolor=fig.get_facecolor()); print(pts, a2)
