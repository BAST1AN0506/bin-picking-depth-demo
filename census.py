# -*- coding: utf-8 -*-
"""
census.py —— 失败原因普查：把"没成功"拆开，看到底卡在哪一步
输出每个相机俯角下，30 个场景的失败原因分布：
  选不出候选 / 两点不在同一物体 / 闭合路径被挡 / 上方通道被挡 / 成功
"""
import os
import sys
import numpy as np
import mujoco
from bin_picking_demo import (make_xml, depth_to_points, object_geom_ids,
                              grasp_for_object, nms, W, H)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def classify(seed, elev, density="dense", n_steps=700):
    model = mujoco.MjModel.from_xml_string(make_xml(seed, elev_deg=elev, density=density))
    data = mujoco.MjData(model)
    for _ in range(n_steps):
        mujoco.mj_step(model, data)
    r = mujoco.Renderer(model, height=H, width=W)
    r.enable_depth_rendering()
    r.update_scene(data, camera="cam"); depth = np.asarray(r.render()).copy()
    r.enable_segmentation_rendering()
    r.update_scene(data, camera="cam"); seg = np.asarray(r.render()).copy()[:, :, 0]
    r.close()
    valid = (depth > 0.05) & (depth < 2.0)
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
        return "选不出候选"
    _, cu, cv, deg, wm, wpx, z, p1, p2 = top[0]
    g1, g2 = seg[p1[1], p1[0]], seg[p2[1], p2[0]]
    if not ((g1 == g2) and (g1 in gids.values())):
        return "两点不在同一物体"
    ts = np.linspace(0, 1, 21)
    for t in ts:
        s = seg[int(round(p1[1] + t * (p2[1] - p1[1]))), int(round(p1[0] + t * (p2[0] - p1[0])))]
        if not ((s == g1) or (s == 0) or (s not in gids.values())):
            return "闭合路径被挡"
    for (px, py) in (p1, p2):
        for dy in range(1, 41, 3):
            yy = py - dy
            if yy < 0:
                break
            s = seg[yy, px]
            if (s != g1) and (s in gids.values()):
                return "上方通道被挡"
    return "成功"


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    cats = ["选不出候选", "两点不在同一物体", "闭合路径被挡", "上方通道被挡", "成功"]
    print(f"{'俯角':<6}" + "".join(f"{c:<16}" for c in cats))
    rows = []
    for elev in (30, 45, 60):
        cnt = {c: 0 for c in cats}
        for i in range(n):
            cnt[classify(i, elev)] += 1
        rows.append((elev, cnt))
        print(f"{elev}°{'':<3}" + "".join(f"{cnt[c]:<16}" for c in cats))
    with open(os.path.join(OUT_DIR, "failure_census.txt"), "w", encoding="utf-8") as f:
        f.write(f"每格 {n} 个场景（堆叠 12 个物体）\n")
        f.write(f"{'俯角':<6}" + "".join(f"{c:<16}" for c in cats) + "\n")
        for elev, cnt in rows:
            f.write(f"{elev}°{'':<3}" + "".join(f"{cnt[c]:<16}" for c in cats) + "\n")
    print("\n已存 failure_census.txt")
