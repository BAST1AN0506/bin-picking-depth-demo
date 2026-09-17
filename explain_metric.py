# -*- coding: utf-8 -*-
"""
explain_metric.py —— 说明"为什么现在的判定标准是错的"的一张图
用真实的深度数据标注：接触点上方那些像素，显示的物体其实在**更远处**，不是障碍。
"""
import os
import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont
from bin_picking_demo import make_xml, object_geom_ids, grasp_for_object, nms, W, H

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

FONT = "C:/Windows/Fonts/msyh.ttc"
f_big = ImageFont.truetype(FONT, 20)
f_mid = ImageFont.truetype(FONT, 16)
f_sm = ImageFont.truetype(FONT, 14)

m = mujoco.MjModel.from_xml_string(make_xml(0))
d = mujoco.MjData(m)
for _ in range(700):
    mujoco.mj_step(m, d)
r = mujoco.Renderer(m, height=H, width=W)
r.update_scene(d, camera="cam"); rgb = np.asarray(r.render()).copy()
r.enable_depth_rendering(); r.update_scene(d, camera="cam"); dep = np.asarray(r.render()).copy()
r.enable_segmentation_rendering(); r.update_scene(d, camera="cam"); sg = np.asarray(r.render()).copy()[:, :, 0]
r.close()

val = (dep > 0.05) & (dep < 2.0)
gids = object_geom_ids(m)
G = []
for i, gid in gids.items():
    mk = (sg == gid) & val
    g = grasp_for_object(mk, dep)
    if g:
        G.append(g)
top = nms(G)
_, cu, cv, deg, wm, wpx, z, p1, p2 = top[0]
g1 = sg[p1[1], p1[0]]
z_contact = dep[p1[1], p1[0]]

img = Image.fromarray(rgb).convert("RGBA")
ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
dr = ImageDraw.Draw(ov)

# 接触点与夹爪
dr.line([p1[0], p1[1], p2[0], p2[1]], fill=(40, 226, 130, 255), width=3)
for p in (p1, p2):
    dr.ellipse([p[0] - 5, p[1] - 5, p[0] + 5, p[1] + 5], outline=(40, 226, 130, 255), width=3)

# 采样点：正上方 10 / 20 / 30 像素
for dy in (10, 20, 30):
    yy = p1[1] - dy
    zz = dep[yy, p1[0]]
    oid = sg[yy, p1[0]]
    closer = zz < z_contact - 0.005
    col = (255, 80, 80, 255) if closer else (255, 215, 90, 255)
    dr.line([p1[0] - 26, yy, p1[0], yy], fill=col, width=2)
    dr.ellipse([p1[0] - 4, yy - 4, p1[0] + 4, yy + 4], fill=col)
    txt = f"上方{dy}px：物体{oid}，深度 {zz:.2f} m"
    tw = dr.textlength(txt, font=f_sm)
    dr.rectangle([p1[0] - 30 - tw - 8, yy - 11, p1[0] - 30, yy + 11], fill=(12, 16, 26, 235))
    dr.text((p1[0] - 30 - tw - 4, yy - 8), txt, font=f_sm, fill=col)

# 顶部说明（自动缩小字号以保证不越界）
def fit_text(dr_, text, x, y, size, fill, max_w, min_size=12):
    while size > min_size:
        f = ImageFont.truetype(FONT, size)
        if dr_.textlength(text, font=f) <= max_w - x:
            break
        size -= 1
    dr_.text((x, y), text, font=ImageFont.truetype(FONT, size), fill=fill)

lines = [
    ("这条判定标准是错的：把“画面更高处”当成了“挡路”", (255, 235, 120), 19),
    (f"绿线处深度 {z_contact:.2f} m；上方各点深度都更大 → 更远，不是障碍", (205, 218, 238), 15),
    ("深度越大＝离相机越远；“屏幕上更高” ≠ “空间上更高”", (205, 218, 238), 15),
]
dr.rectangle([0, 0, W - 1, 70], fill=(10, 14, 24, 245))
yy = 6
for t, col, sz in lines:
    fit_text(dr, t, 8, yy, sz, col, W - 12)
    yy += 22

out = Image.alpha_composite(img, ov).convert("RGB")
out.save(os.path.join(OUT_DIR, "explain_metric.png"))
print("saved explain_metric.png  接触点深度 %.3f m" % z_contact)
