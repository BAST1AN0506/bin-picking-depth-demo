# -*- coding: utf-8 -*-
"""
clearance_check.py —— 关键一问：这只夹爪到底下不下得去？
用"排除抓取目标自身"的竖直柱，按不同半径（= 手指/夹爪需要的空间）逐物体量距离。
半径扫到手指粗细，是为了把两件事分开：
  · 抓取目标自身必然与柱相交（柱就长在它的表面上）→ 必须排除
  · 别的物体是否侵入柱内 → 这才是物理意义上的"下不去手"
"""
import numpy as np
import mujoco
from bin_picking_demo import make_xml, object_geom_ids, grasp_for_object, nms, W, H

ELEV, DENS, SEED = 30.0, "dense", 14
BASE = make_xml(SEED, elev_deg=ELEV, density=DENS)

m = mujoco.MjModel.from_xml_string(BASE)
d = mujoco.MjData(m)
for _ in range(700):
    mujoco.mj_step(m, d)
r = mujoco.Renderer(m, height=H, width=W)
r.enable_depth_rendering(); r.update_scene(d, camera="cam"); dep = np.asarray(r.render()).copy()
r.enable_segmentation_rendering(); r.update_scene(d, camera="cam"); sg = np.asarray(r.render()).copy()[:, :, 0]
r.close()
QPOS = d.qpos.copy()
cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "cam")
fovy = float(m.cam_fovy[cam_id]); F = (H / 2) / np.tan(np.deg2rad(fovy) / 2)
cx, cy = W / 2, H / 2
R = d.cam_xmat[cam_id].reshape(3, 3); t = d.cam_xpos[cam_id]
valid = (dep > 0.05) & (dep < 2.0)


def to_world(u, v, dd):
    return R @ np.array([(u - cx) / F * dd, -(v - cy) / F * dd, -dd]) + t


def probe_model(extra):
    mm = mujoco.MjModel.from_xml_string(BASE.replace("</worldbody>", extra + "\n  </worldbody>"))
    dd_ = mujoco.MjData(mm)
    dd_.qpos[:] = QPOS
    mujoco.mj_forward(mm, dd_)
    return mm, dd_


def geom_aabb(model, data, gid):
    p = data.geom_xpos[gid]; Rg = data.geom_xmat[gid].reshape(3, 3); s = model.geom_size[gid]
    gt = model.geom_type[gid]
    if gt == mujoco.mjtGeom.mjGEOM_SPHERE:
        hx = hy = hz = s[0]
    elif gt == mujoco.mjtGeom.mjGEOM_CYLINDER:
        hx, hy, hz = s[0], s[0], s[1]
    else:
        hx, hy, hz = s[0], s[1], s[2]
    c = np.array([[sx * hx, sy * hy, sz * hz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    w = (Rg @ c.T).T + p
    return w.min(0), w.max(0)


objs = object_geom_ids(m)
G = [grasp_for_object((sg == gg) & valid, dep) for gg in objs.values() if ((sg == gg) & valid).sum() >= 60]
G = [g for g in G if g]
_, cu, cv, deg, wm, wpx, z0, p1, p2 = nms(G)[0]
g1 = int(sg[p1[1], p1[0]])
oi = [k for k, v in objs.items() if v == g1][0]
print(f"抓取目标 obj{oi}（geom id {g1}）；接触点 p1=({p1[0]},{p1[1]}) d={dep[p1[1],p1[0]]:.3f}m，p2=({p2[0]},{p2[1]}) d={dep[p2[1],p2[0]]:.3f}m")
print(f"真实几何中心 obj{oi} = ({d.geom_xpos[g1][0]:+.4f},{d.geom_xpos[g1][1]:+.4f},{d.geom_xpos[g1][2]:+.4f})，size = {np.round(m.geom_size[g1],4)}")
print()

def clearance(u, v, rad):
    w = to_world(u, v, dep[v, u])
    probe = (f'<body name="probe" pos="{w[0]:.5f} {w[1]:.5f} {w[2]+0.025:.5f}">'
             f'<geom name="g_probe" type="cylinder" size="{rad} 0.025" rgba="0 1 1 1"/></body>')
    m2, d2 = probe_model(probe)
    pid = mujoco.mj_name2id(m2, mujoco.mjtObj.mjOBJ_GEOM, "g_probe")
    out = {}
    for i, gg in object_geom_ids(m2).items():
        ft = np.zeros(6)
        out[i] = float(mujoco.mj_geomDistance(m2, d2, pid, gg, 2.0, ft))
    lo, hi = geom_aabb(m2, d2, pid)
    return out, w, (lo, hi)


print("## A. 每次张开的竖直柱（半径 3cm）逐物体距离 —— 看清楚谁在柱里")
for tag, (u, v) in (("p1", p1), ("p2", p2)):
    out, w, _ = clearance(u, v, 0.03)
    print(f"\n接触点 {tag} = ({u},{v})，重建世界 ({w[0]:+.4f},{w[1]:+.4f},{w[2]:+.4f})，柱底在 z={w[2]:.4f}")
    print("| 物体 | 与柱面距离 |")
    print("|---|---|")
    for i, val in sorted(out.items(), key=lambda kv: kv[1]):
        star = "  ← 抓取目标自身（预期相交）" if i == oi else ("  ← **别的物体**" if val < 0 else "")
        print(f"| obj{i} | {val*100:+.2f} cm{star} |")
print()

print("## B. 半径扫描（只看『别的物体』，目标自身已排除）")
print()
print("| 柱半径 | p1 柱内最近的别的物体 | p2 柱内最近的别的物体 |")
print("|---|---|---|")
for rad in (0.005, 0.010, 0.020, 0.030, 0.040):
    cells = []
    for (u, v) in (p1, p2):
        out, w, _ = clearance(u, v, rad)
        others = {i: s for i, s in out.items() if i != oi}
        j = min(others, key=lambda k: others[k])
        cells.append(f"obj{j}: {others[j]*100:+.2f} cm")
    print(f"| {rad*100:.1f} cm | {cells[0]} | {cells[1]} |")
print()
print("解读：")
print("- 半径 ≥ 2cm 时，邻物都还只是『擦到柱面/轻微侵入』；半径 ≥ 3cm 时才明显侵入。")
print("- 这只夹爪真正需要多少空间，取决于手指厚度与张开方式 —— 所以『能不能下得去』")
print("  不是一条通用判据能回答的，必须先定夹爪模型。")
