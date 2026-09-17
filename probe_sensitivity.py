# -*- coding: utf-8 -*-
"""
probe_sensitivity.py —— 两个稳健性测试（v2：修正 probe 模型的位姿）
  1) 反投影的像素级误差：探针球放在 obj1 掩码的"边缘像素"和"内部像素"上，
     量它到 obj1 真实表面的带符号距离
  2) 把"夹爪下行柱"半径从 3cm 放大到 6cm，看"柱内没有别的物体"还成立吗

⚠️ v1 的坑：探针模型只跑了 mj_forward、没跑沉降，物体还停在初始堆叠位置，
   量出来的距离全部无效。v2 把已沉降的 qpos 原样搬进探针模型后再算。
"""
import numpy as np
import mujoco
from bin_picking_demo import make_xml, object_geom_ids, grasp_for_object, nms, W, H

ELEV, DENS, SEED = 30.0, "dense", 14
BASE = make_xml(SEED, elev_deg=ELEV, density=DENS)

# ---- 基准场景：真的沉降 700 步
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
    """加入探针几何，并把已沉降的位姿搬进来（关键修正）"""
    mm = mujoco.MjModel.from_xml_string(BASE.replace("</worldbody>", extra + "\n  </worldbody>"))
    dd_ = mujoco.MjData(mm)
    assert mm.nq == m.nq, f"nq 不一致 {mm.nq} vs {m.nq}"
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


def gap(lo1, hi1, lo2, hi2):
    s = np.maximum(np.maximum(lo1 - hi2, lo2 - hi1), 0.0)
    return float(s.max()), bool((s > 0).any())


objs = object_geom_ids(m)
G = []
for gg in objs.values():
    if ((sg == gg) & valid).sum() >= 60:
        g = grasp_for_object((sg == gg) & valid, dep)
        if g:
            G.append(g)
top = nms(G)
_, cu, cv, deg, wm, wpx, z0, p1, p2 = top[0]
g1 = int(sg[p1[1], p1[0]])
oi = [k for k, v in objs.items() if v == g1][0]
mask1 = (sg == g1) & valid

print("## 0. 修正校验：探针模型里的物体位姿必须和基准场景一致")
mm0, dd0 = probe_model('<body pos="0 0 0.5"><geom name="g_dummy" type="sphere" size="0.001"/></body>')
mx = max(np.abs(dd0.geom_xpos[gg] - d.geom_xpos[gg]).max() for gg in objs.values())
print(f"- 12 个物体位姿的最大偏差 = {mx:.2e} m  →", "PASS" if mx < 1e-9 else "FAIL")
print()


def edge_dist(px, py, maxr=8):
    for rad in range(0, maxr + 1):
        for dy in range(-rad, rad + 1):
            for dx in range(-rad, rad + 1):
                if max(abs(dx), abs(dy)) != rad:
                    continue
                yy, xx = py + dy, px + dx
                if not (0 <= yy < H and 0 <= xx < W) or sg[yy, xx] != g1:
                    return rad
    return maxr


ys, xs = np.nonzero(mask1)
ed = np.array([edge_dist(x, y) for x, y in zip(xs, ys)])
print(f"## 1. 掩码结构：obj{oi} 共 {mask1.sum()} px，其中距边界 ≤1px 的有 {(ed<=1).sum()} px，≥6px 的有 {(ed>=6).sum()} px")
print()

rng = np.random.default_rng(0)
edge_pts = list(zip(xs[ed <= 1], ys[ed <= 1]))
inner_pts = list(zip(xs[ed >= 6], ys[ed >= 6]))
edge_sel = [edge_pts[i] for i in rng.choice(len(edge_pts), size=min(8, len(edge_pts)), replace=False)]
inner_sel = [inner_pts[i] for i in rng.choice(len(inner_pts), size=min(8, len(inner_pts)), replace=False)]
pts = [(x, y, "边缘(≤1px)") for x, y in edge_sel] + [(x, y, "内部(≥6px)") for x, y in inner_sel]

bodies = "".join(
    f'<body pos="{to_world(px,py,dep[py,px])[0]:.5f} {to_world(px,py,dep[py,px])[1]:.5f} {to_world(px,py,dep[py,px])[2]:.5f}">'
    f'<geom name="gp{k}" type="sphere" size="0.0005"/></body>'
    for k, (px, py, _) in enumerate(pts))
mm, dd_ = probe_model(bodies)
gid_new = object_geom_ids(mm)[oi]

print(f"## 2. 单像素反投影误差（重建点到 obj{oi} 真实表面的带符号距离；负=穿进物体里）")
print()
print(f"| 位置 | 像素 (u,v) | 深度(m) | 到 obj{oi} 表面距离 |")
print("|---|---|---|---|")
errs = {"边缘(≤1px)": [], "内部(≥6px)": []}
for k, (px, py, tag) in enumerate(pts):
    pid = mujoco.mj_name2id(mm, mujoco.mjtObj.mjOBJ_GEOM, f"gp{k}")
    ft = np.zeros(6)
    v = float(mujoco.mj_geomDistance(mm, dd_, pid, gid_new, 1.0, ft)) - 0.0005
    errs[tag].append(v)
    print(f"| {tag} | ({px},{py}) | {dep[py,px]:.4f} | **{v*100:+.2f} cm** |")
print()
for tag, vv in errs.items():
    vv = np.array(vv)
    print(f"- {tag}：中位 {np.median(vv)*100:+.2f} cm，范围 [{vv.min()*100:+.2f}, {vv.max()*100:+.2f}] cm（n={len(vv)}）")
print()

# ================================================ 3. 柱半径敏感性
w1 = to_world(p2[0], p2[1], dep[p2[1], p2[0]])
print("## 3. 柱半径敏感性（把重建误差计进去）")
print()
print(f"抓取点重建世界坐标 = ({w1[0]:+.4f}, {w1[1]:+.4f}, {w1[2]:+.4f}) m；"
      f"目标 obj{oi} 真实几何中心 = ({d.geom_xpos[g1][0]:+.4f}, {d.geom_xpos[g1][1]:+.4f}, {d.geom_xpos[g1][2]:+.4f}) m")
print()
print("| 柱半径 | 柱高 | 别的物体:AABB 最近间隙 | 全部别的物体 AABB 分离 | 别的物体:mj_geomDistance 最近 | 与目标自身 |")
print("|---|---|---|---|---|---|")
for rad in (0.03, 0.04, 0.05, 0.06):
    probe = (f'<body name="probe" pos="{w1[0]:.5f} {w1[1]:.5f} {w1[2]+0.025:.5f}">'
             f'<geom name="g_probe" type="cylinder" size="{rad} 0.025" rgba="0 1 1 1"/></body>')
    m2, d2 = probe_model(probe)
    pid = mujoco.mj_name2id(m2, mujoco.mjtObj.mjOBJ_GEOM, "g_probe")
    plo, phi = geom_aabb(m2, d2, pid)
    gs = object_geom_ids(m2)
    aabbs, apis, sep_all, self_api = [], [], True, None
    for i, gg in gs.items():
        lo, hi = geom_aabb(m2, d2, gg)
        gp, sp = gap(plo, phi, lo, hi)
        ft = np.zeros(6)
        api = float(mujoco.mj_geomDistance(m2, d2, pid, gg, 1.0, ft))
        if i == oi:
            self_api = api
            continue
        aabbs.append(gp); sep_all &= sp; apis.append(api)
    print(f"| {rad*100:.0f} cm | 5 cm | {min(aabbs)*100:.2f} cm | {'是' if sep_all else '否'} | {min(apis)*100:+.2f} cm | {self_api*100:+.2f} cm |")
print()
print("解读：半径 3cm 是「夹爪实际需要的空间」；把半径放大是对「抓取点重建误差」做保守补偿。")
print("只要放大后仍然『全部别的物体 AABB 分离』，结论（柱内没有别的物体）就站得住。")
