"""Turn an Orbbec RGB-D session from the HIM release into two MP4s (both colour cams; colour + depth)."""
import json
import sys

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

S = sys.argv[1]
out_dir = sys.argv[2]
C0, C1 = f"{S}/cam0_CP0E8530008E", f"{S}/cam1_CP0E8530008F"
f0 = [json.loads(line) for line in open(f"{C0}/frames.jsonl")]
f1 = {r["idx"]: r for r in (json.loads(line) for line in open(f"{C1}/frames.jsonl"))}
font = ImageFont.load_default()


def depth_rgb(path):
    dd = np.asarray(Image.open(path)).astype(float)  # millimetres
    v = np.clip((dd - 500) / 2500, 0, 1)
    rgb = np.stack([255 * (1 - v), 180 * (1 - np.abs(v - 0.5) * 2), 255 * v], -1)
    rgb[dd == 0] = 0
    return Image.fromarray(rgb.astype(np.uint8)).resize((640, 480))


def bar(img, text):
    out = Image.new("RGB", (img.width, img.height + 20))
    out.paste(img, (0, 20))
    d = ImageDraw.Draw(out)
    d.rectangle([0, 0, out.width, 20], fill=(10, 12, 16))
    d.text((8, 4), text, fill=(255, 255, 255), font=font)
    return np.asarray(out)


w1 = imageio.get_writer(f"{out_dir}/rgbd_session.mp4", fps=15, codec="libx264", quality=7, macro_block_size=8)
w2 = imageio.get_writer(f"{out_dir}/rgbd_depth.mp4", fps=15, codec="libx264", quality=7, macro_block_size=8)
n = 0
for r in f0:
    if not r.get("color_file"):
        continue
    c0 = Image.open(f"{C0}/{r['color_file']}").convert("RGB")
    r1 = f1.get(r["idx"])
    if r1 and r1.get("color_file"):
        c1 = Image.open(f"{C1}/{r1['color_file']}").convert("RGB")
        both = Image.new("RGB", (1280, 480)); both.paste(c0, (0, 0)); both.paste(c1, (640, 0))
        w1.append_data(bar(both, f"Orbbec RGB-D session 214619  frame {r['idx']}  t={r['elapsed_s']:.2f}s   left cam CP0E8530008E | right cam CP0E8530008F"))
    cd = Image.new("RGB", (1280, 480)); cd.paste(c0, (0, 0)); cd.paste(depth_rgb(f"{C0}/{r['depth_file']}"), (640, 0))
    w2.append_data(bar(cd, f"cam CP0E8530008E  frame {r['idx']}   colour | aligned depth (0.5 m red -> 3 m blue, black = no return)"))
    n += 1
w1.close(); w2.close()
print("frames", n)
