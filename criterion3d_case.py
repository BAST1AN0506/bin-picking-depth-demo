# -*- coding: utf-8 -*-
"""
criterion3d_case.py —— 挑一个"旧判据判挡、真实几何判能抓"的场景，把两种判据画在同一张图上
========================================================================================
这张图是给"你这判据到底错在哪"这类追问准备的：一眼能看出旧判据往上找的那条线
在图像里是竖的、在世界里是斜的，而夹爪真正要占的那两块空间在哪儿。

跑法：
  ./.venv/Scripts/python.exe criterion3d_case.py            # 自动找第一个翻转场景
  ./.venv/Scripts/python.exe criterion3d_case.py 30 12 dense  # 指定 俯角 种子 场景密度
"""

import os
import sys

import numpy as np
import mujoco
from PIL import Image, ImageDraw

from bin_picking_demo import (make_xml, object_geom_ids, grasp_for_object, nms, W, H)
from gripper_model import GRIPPER, finger_boxes
from criterion3d import (xml_with_fingers, camera_pose, cloud_from_depth, pixel_to_world,
                         world_to_pixel, admissible, old_criterion_2d, new_criterion_cloud,
                         new_criterion_exact, exact_min_clearance, N_STEPS)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def find_flip(elev=30, density="dense", seeds=range(30)):
    for seed in seeds:
        rec = evaluate(seed, elev, density)
        if rec and rec["A"] == 0 and rec["D"] == 1:
            return rec
    return None


def evaluate(seed, elev, density):
    model = mujoco.MjModel.from_xml_string(xml_with_fingers(seed, elev, density))
    data = mujoco.MjData(model)
    for _ in range(N_STEPS):
        mujoco.mj_step(model, data)

    r = mujoco.Renderer(model, height=H, width=W)
    r.update_scene(data, camera="cam")
    rgb = np.asarray(r.render()).copy()
    r.enable_depth_rendering()
    r.update_scene(data, camera="cam")
    depth = np.asarray(r.render()).copy()
    r.enable_segmentation_rendering()
    r.update_scene(data, camera="cam")
    seg = np.asarray(r.render()).copy()[:, :, 0]
    r.close()

    valid = (depth > 0.05) & (depth < 2.0)
    gids = object_geom_ids(model)
    R, t = camera_pose(data)

    grasps = []
    for i, gid in gids.items():
        mask = (seg == gid) & valid
        if mask.sum() < 60:
            continue
        g = grasp_for_object(mask, depth, seg=seg, gid=gid)
        if g:
            grasps.append(g)
    top = nms(grasps)
    if not top:
        return None
    _, cu, cv, deg, width_m, wpx, z0, p1, p2 = top[0]
    ok_adm, target_gid, why = admissible(seg, gids, p1, p2)
    if not ok_adm:
        return None

    c1 = pixel_to_world(p1[0], p1[1], float(depth[p1[1], p1[0]]), R, t)
    c2 = pixel_to_world(p2[0], p2[1], float(depth[p2[1], p2[0]]), R, t)
    boxes = finger_boxes(c1, c2, GRIPPER)
    pts, labels = cloud_from_depth(depth, seg, R, t, valid)
    return dict(seed=seed, elev=elev, density=density, rgb=rgb, seg=seg, depth=depth,
                R=R, t=t, p1=p1, p2=p2, c1=c1, c2=c2, boxes=boxes, target_gid=target_gid,
                A=int(old_criterion_2d(seg, gids, target_gid, p1, p2)),
                B=int(new_criterion_cloud(pts, labels, boxes, target_gid, GRIPPER["tol"])),
                D=int(new_criterion_exact(model, data, boxes, target_gid, GRIPPER["tol"])),
                clearance=exact_min_clearance(model, data, boxes, target_gid),
                width_cm=width_m * 100, opening_cm=float(np.linalg.norm(c2 - c1)) * 100)


def box_outline(box, R, t):
    """盒子的 8 个角投到像素，返回轮廓点（按投影后的凸包顺序画不出来就按角顺序画）。"""
    _, center, axes, half = box
    pts = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                P = center + axes @ np.array([sx * half[0], sy * half[1], sz * half[2]])
                u, v, z = world_to_pixel(P, R, t)
                pts.append((u, v, z))
    return pts


def draw(rec, path):
    img = Image.fromarray(rec["rgb"])
    dr = ImageDraw.Draw(img)
    R, t = rec["R"], rec["t"]
    p1, p2 = rec["p1"], rec["p2"]

    # 目标物体描边
    m = (rec["seg"] == rec["target_gid"])
    edge = m & ~(np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1))
    for y, x in zip(*np.nonzero(edge)):
        dr.point((x, y), fill=(0, 210, 90))

    # 抓取：两接触点 + 连线
    dr.line([p1[0], p1[1], p2[0], p2[1]], fill=(0, 210, 90), width=2)
    for p in (p1, p2):
        dr.ellipse([p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4], outline=(0, 210, 90), width=2)

    # 旧判据：图像里往上 40 px 的通道
    for p in (p1, p2):
        dr.line([p[0], p[1], p[0], max(0, p[1] - 40)], fill=(210, 40, 40), width=1)
        dr.point((p[0], max(0, p[1] - 40)), fill=(210, 40, 40))

    # 新判据：两个手指的扫过空间（投影轮廓）
    cols = [(0, 120, 255), (255, 150, 0)]
    for k, box in enumerate(rec["boxes"]):
        if box[0] == "body":
            continue                                  # 只画指体，图干净一点
        corners = box_outline(box, R, t)
        # 用 8 个角在图像里的最小外接框近似（盒子在图上基本是条带）
        us = [c[0] for c in corners]
        vs = [c[1] for c in corners]
        dr.rectangle([min(us), min(vs), max(us), max(vs)], outline=cols[k // 2 % 2], width=2)

    dr.text((6, 6), "seed %d  elev %d  %s" % (rec["seed"], rec["elev"], rec["density"]),
            fill=(255, 255, 255))
    dr.text((6, 20), "red = old 2D corridor   blue = finger swept box (tip)",
            fill=(255, 255, 255))
    dr.text((6, 34), "A=%d  B=%d  D=%d   opening %.1f cm   clearance %+.1f cm" %
            (rec["A"], rec["B"], rec["D"], rec["opening_cm"], rec["clearance"] * 100),
            fill=(255, 255, 255))
    img.save(path)
    return path


if __name__ == "__main__":
    if len(sys.argv) >= 4:
        rec = evaluate(int(sys.argv[2]), int(sys.argv[1]), sys.argv[3])
    else:
        rec = find_flip()
    if rec is None:
        print("没有找到符合条件的场景")
        sys.exit(1)
    out = os.path.join(OUT_DIR, "criterion3d_case.png")
    draw(rec, out)
    print("seed=%d elev=%d %s  宽 %.2f cm  张开 %.2f cm  A=%d B=%d D=%d  最小间隙 %+.1f cm -> %s"
          % (rec["seed"], rec["elev"], rec["density"], rec["width_cm"], rec["opening_cm"],
             rec["A"], rec["B"], rec["D"], rec["clearance"] * 100, out))
