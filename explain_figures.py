# -*- coding: utf-8 -*-
"""
explain_figures.py —— 生成两张"看懂那张表"的示意图
  1) explain_failure.png  ：一个真实的失败案例，标出"夹爪下不去的通道"和被谁挡住
  2) explain_viewpoint.png：30°/45°/60° 三种视角为什么成功率不同（侧视示意）
"""

import os
import numpy as np
import mujoco
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

from bin_picking_demo import (make_xml, depth_to_points, object_geom_ids,
                              grasp_for_object, nms, W, H, F)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

OUT = OUT_DIR + "/"


# ---------------------------------------------------------------- 图 1：失败案例
def render_case(seed, elev, density="dense"):
    model = mujoco.MjModel.from_xml_string(make_xml(seed, elev_deg=elev, density=density))
    data = mujoco.MjData(model)
    for _ in range(700):
        mujoco.mj_step(model, data)
    r = mujoco.Renderer(model, height=H, width=W)
    r.update_scene(data, camera="cam"); rgb = np.asarray(r.render()).copy()
    r.enable_depth_rendering()
    r.update_scene(data, camera="cam"); depth = np.asarray(r.render()).copy()
    r.enable_segmentation_rendering()
    r.update_scene(data, camera="cam"); seg = np.asarray(r.render()).copy()
    r.close()
    return rgb, depth, seg[:, :, 0]


def find_failing_case(elev=60, density="dense", max_seed=40):
    for seed in range(max_seed):
        rgb, depth, seg = render_case(seed, elev, density)
        valid = (depth > 0.05) & (depth < 2.0)
        gids = {i: g for i, g in
                [(i, g) for i, g in zip(range(12), [])] }  # placeholder
        # 直接复用主程序的 id 表：重新读一次模型
        model = mujoco.MjModel.from_xml_string(make_xml(seed, elev_deg=elev, density=density))
        gids = object_geom_ids(model)
        grasps = []
        for i, gid in gids.items():
            mask = (seg == gid) & valid
            if mask.sum() < 60:
                continue
            g = grasp_for_object(mask, depth)
            if g:
                grasps.append(g)
        top = nms(grasps)
        if not top:
            continue
        _, cu, cv, deg, wm, wpx, z, p1, p2 = top[0]
        g1 = seg[p1[1], p1[0]]
        blocker = None
        for (px, py) in (p1, p2):
            for dy in range(1, 41, 3):
                yy = py - dy
                if yy < 0:
                    break
                s = seg[yy, px]
                if (s != g1) and (s in gids.values()):
                    blocker = s
                    break
            if blocker is not None:
                break
        if blocker is not None:
            return seed, rgb, seg, gids, top[0], blocker
    return None


def draw_failure():
    res = find_failing_case()
    if res is None:
        print("没找到失败案例（可以换角度/密度）")
        return
    seed, rgb, seg, gids, grasp, blocker = res
    _, cu, cv, deg, wm, wpx, z, p1, p2 = grasp
    img = Image.fromarray(rgb).convert("RGB")
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)

    # 被挡住的那个物体：红描边
    m = (seg == blocker)
    edge = m & ~(np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1))
    for y, x in zip(*np.nonzero(edge)):
        d.point((int(x), int(y)), fill=(255, 70, 70, 255))
    # 两个接触点上方的"下手通道"（40px 高）
    for (px, py) in (p1, p2):
        d.rectangle([px - 5, max(py - 40, 0), px + 5, py], fill=(255, 210, 0, 70),
                    outline=(255, 210, 0, 220))
    # 夹爪位置（红=失败）
    d.line([p1[0], p1[1], p2[0], p2[1]], fill=(255, 70, 70, 255), width=3)
    for p in (p1, p2):
        d.ellipse([p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4], outline=(255, 70, 70, 255), width=2)

    out = Image.alpha_composite(img.convert("RGBA"), ov).convert("RGB")
    dr = ImageDraw.Draw(out)
    dr.rectangle([0, 0, W - 1, 46], fill=(10, 14, 24))
    dr.text((8, 6), "失败案例：夹爪要夹的东西没被挡，但“从上方下手的通道”被红色物体挡了",
            fill=(255, 235, 120))
    dr.text((8, 24), f"黄框 = 接触点上方 40px 的通道   |   场景：堆叠 12 个物体，相机俯角 60°",
            fill=(180, 200, 230))
    out.save(OUT + "explain_failure.png")
    print("saved explain_failure.png  (seed=%d)" % seed)


# ------------------------------------------------------- 图 2：三种视角的侧视示意
def draw_viewpoint():
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, ang in zip(axes, (30, 45, 60)):
        th = np.deg2rad(ang)
        # 料箱剖面
        ax.plot([-1.2, 1.2], [0, 0], color="#8899aa", lw=3)
        ax.plot([-1.2, -1.0], [0, 0.9], color="#8899aa", lw=3)
        ax.plot([1.0, 1.2], [0, 0.9], color="#8899aa", lw=3)
        # 两个物体（前面一个会挡视线）
        ax.add_patch(plt.Rectangle((-0.95, 0), 0.5, 0.55, color="#5aa9f8"))
        ax.add_patch(plt.Rectangle((0.1, 0), 0.5, 0.75, color="#e95c7a"))
        # 相机与视线
        cx, cz = 0.0 - 2.3 * np.cos(th), 0.2 + 2.3 * np.sin(th)
        ax.plot([cx], [cz], marker="o", color="#f6c454", ms=9)
        ax.annotate("相机", (cx, cz), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
        # 想看的是蓝色物体（左边那个）——画两条视线
        tgt = (-0.7, 0.4)
        ax.plot([cx, tgt[0]], [cz, tgt[1]], "--", color="#2b2b2b", lw=1.4)
        ax.plot([cx, tgt[0]], [cz, tgt[1]], "--", color="#2b2b2b", lw=1.4)
        # 标出视线是否被红色物体挡住
        if ang <= 30:
            note = "太斜：前面物体挡住后面\n看不到 → 抓取点选不出来"
            col = "#c0392b"
        elif ang >= 60:
            note = "太陡：物体在图像上挤在一起\n“上方通道”容易撞到邻居"
            col = "#c0392b"
        else:
            note = "中间角度：既看得到、\n又有空间从上方下去"
            col = "#1e8449"
        ax.text(0, 1.55, f"相机俯角 {ang}°", ha="center", fontsize=11)
        ax.text(0, 1.15, note, ha="center", fontsize=9.5, color=col)
        ax.set_xlim(-2.6, 2.2)
        ax.set_ylim(-0.15, 1.9)
        ax.axis("off")
    fig.suptitle("为什么不是越陡越好、也不是越斜越好", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT + "explain_viewpoint.png", dpi=160)
    print("saved explain_viewpoint.png")


if __name__ == "__main__":
    draw_failure()
    draw_viewpoint()
