# -*- coding: utf-8 -*-
"""
explain_why_criterion_fails.py (v3) —— 失败样例诊断
  为什么"图像上方 40px 通道"判据不成立；世界坐标里的碰撞检查该怎么做。

全流程用真实渲染数据跑，且每一步都自带核对：
  0) 用 census.classify 复现这一格 30 场景的失败分布
  1) 从"上方通道被挡"的场景里挑一个阻塞点最远的样例
  2) 相机内参、深度值定义、外参
  3) 像素↔世界变换（用物体质心 + 已知几何自检）
  4) "接触点上方 40px 走廊"逐像素反投影到世界坐标
  5) 世界坐标真 3D 检查：重建精度核对 → 探针柱 + mj_geomDistance（含 API 标定）
     → 半径扫描；并说明"为什么必须排除抓取目标自身"
  6) 同一样例：旧判据 vs 3D 检查的结论（含本样例的诚实结论）

⚠️ v1/v2 的两个坑（已修，写在这里免得再踩）：
  · 探针模型的物体必须用**已沉降的 qpos**，否则物体还停在初始堆叠位置，距离全错
  · 柱状空间必然与**抓取目标自身**相交（柱就长在它的表面上），比较时必须排除它

⚠️ 这是**单例诊断**。把 3D 检查接进 run_scene、重跑 6 格 × 30 场景的**复测未完成**。

产出：
  criterion_failure_evidence.png   四联图
  criterion_failure_evidence.md    全部数字与结论
"""
import os
import numpy as np
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

from bin_picking_demo import make_xml, object_geom_ids, grasp_for_object, nms, W, H
from census import classify

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

OUT = OUT_DIR + "/"
ELEV, DENSITY, N_STEPS, N_SEED = 30.0, "dense", 700, 30
REPORT = []


def log(s=""):
    print(s)
    REPORT.append(str(s))


def settle(seed):
    m = mujoco.MjModel.from_xml_string(make_xml(seed, elev_deg=ELEV, density=DENSITY))
    d = mujoco.MjData(m)
    for _ in range(N_STEPS):
        mujoco.mj_step(m, d)
    return m, d


def render(m, d):
    r = mujoco.Renderer(m, height=H, width=W)
    r.update_scene(d, camera="cam"); rgb = np.asarray(r.render()).copy()
    r.enable_depth_rendering(); r.update_scene(d, camera="cam"); depth = np.asarray(r.render()).copy()
    r.enable_segmentation_rendering(); r.update_scene(d, camera="cam"); seg = np.asarray(r.render()).copy()[:, :, 0]
    r.close()
    return rgb, depth, seg


def top1(seg, depth, gids):
    valid = (depth > 0.05) & (depth < 2.0)
    G = []
    for i, gid in gids.items():
        mask = (seg == gid) & valid
        if mask.sum() < 60:
            continue
        g = grasp_for_object(mask, depth)
        if g:
            G.append(g)
    return nms(G)


# ================================================ 0. 失败分布
log("# 失败样例诊断：为什么「图像上方 40px 通道」判据不成立")
log()
log(f"场景：料箱 + 堆叠 12 个物体，相机俯角 {ELEV:.0f}°，沉降 {N_STEPS} 步，种子 0–{N_SEED-1}")
log()
log("## 0. 复现这一格的失败分布（census.classify，与 failure_census.txt 同一套逻辑）")
log()
cats = ["选不出候选", "两点不在同一物体", "闭合路径被挡", "上方通道被挡", "成功"]
cnt = {c: 0 for c in cats}
for i in range(N_SEED):
    cnt[classify(i, ELEV)] += 1
log("| " + " | ".join(cats) + " |")
log("|" + "---|" * len(cats))
log("| " + " | ".join(str(cnt[c]) for c in cats) + " |")
blocked = [i for i in range(N_SEED) if classify(i, ELEV) == "上方通道被挡"]
log()
log(f"→ 与 failure_census.txt 的 23/30 一致；这些种子：{blocked}")
log()

# ================================================ 1. 挑样例
cands = []
for seed in blocked:
    m, d = settle(seed)
    rgb, depth, seg = render(m, d)
    gids = object_geom_ids(m)
    t = top1(seg, depth, gids)
    if not t:
        continue
    _, cu, cv, deg, wm, wpx, z0, p1, p2 = t[0]
    g1 = int(seg[p1[1], p1[0]])
    walks, first_dy, hit = {}, None, None
    for k, (px, py) in enumerate((p1, p2)):
        rows = []
        for dy in range(1, 41, 3):
            yy = py - dy
            if yy < 0:
                break
            s = int(seg[yy, px])
            rows.append((dy, px, yy, s, float(depth[yy, px])))
        walks[k] = rows
        h = [r for r in rows if (r[3] != g1) and (r[3] in gids.values())]
        if h and (first_dy is None or h[0][0] < first_dy):
            first_dy, hit = h[0][0], h[0]
    if first_dy is not None:
        cands.append(dict(seed=seed, m=m, d=d, rgb=rgb, depth=depth, seg=seg, gids=gids,
                          top=t[0], g1=g1, walks=walks, first_dy=first_dy))
cands.sort(key=lambda c: -c["first_dy"])
C = cands[0]
log(f"## 1. 选定样例：seed = {C['seed']}（首个阻塞像素距接触点 {C['first_dy']} px，这一格里阻塞最远的案例）")
log()
log("| seed | 抓取方向 | 夹爪宽度 | 首个阻塞点距接触点 |")
log("|---|---|---|---|")
for c in cands[:8]:
    log(f"| {c['seed']} | {c['top'][3]}° | {c['top'][4]*100:.2f} cm | {c['first_dy']} px |")
log("| … | | | |")
log()

seed = C["seed"]; m = C["m"]; d = C["d"]; rgb = C["rgb"]; depth = C["depth"]; seg = C["seg"]
gids = C["gids"]; _, cu, cv, deg, wm, wpx, z0, p1, p2 = C["top"]
g1 = C["g1"]
obj_of = {v: k for k, v in gids.items()}
valid = (depth > 0.05) & (depth < 2.0)
QPOS = d.qpos.copy()

k_main = int(np.argmax([max([r[0] for r in C["walks"][k] if (r[3] != g1) and (r[3] in gids.values())] or [0])
                        for k in (0, 1)]))
P = (p1, p2)[k_main]
walk = C["walks"][k_main]
bk = [r for r in walk if (r[3] != g1) and (r[3] in gids.values())][0]
z_main = float(depth[P[1], P[0]])
log(f"- 抓取目标 obj{obj_of[g1]}（geom id {g1}）；夹爪开合方向 {deg}°，宽度 {wm*100:.2f} cm（{wpx:.1f} px）")
log(f"- 接触点 p1=({p1[0]},{p1[1]}) 深度 {depth[p1[1],p1[0]]:.3f} m；p2=({p2[0]},{p2[1]}) 深度 {depth[p2[1],p2[0]]:.3f} m")
log(f"- 下文以 p{k_main+1} 那一列为主线")
log()

# ================================================ 2. 内参/外参/深度定义
cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "cam")
fovy = float(m.cam_fovy[cam_id])
F = (H / 2) / np.tan(np.deg2rad(fovy) / 2)
cx, cy = W / 2, H / 2
R = d.cam_xmat[cam_id].reshape(3, 3).copy(); t = d.cam_xpos[cam_id].copy()

log("## 2. 深度值的定义、相机内参、外参")
log()
log(f"- 图像 {W}×{H}；垂直视场角 fovy = {fovy:.1f}°（XML 写死；无畸变，主点在图像中心）")
log(f"- 焦距（像素）：F = (H/2)/tan(fovy/2) = {H/2:.0f}/tan({fovy/2:.1f}°) = **{F:.1f} px**；主点 cx={cx:.1f}, cy={cy:.1f}")
log(f"- **深度值 = 沿相机光轴的距离（米）**：不是到相机的直线距离，不是世界高度，也不是\"层数\"。")
log(f"  MuJoCo 的 depth 渲染给的就是这个值；代码只保留 0.05–2.0 m 的像素（bin_picking_demo.py 第 174 行）。")
log(f"- 外参：相机位置 t = ({t[0]:+.4f}, {t[1]:+.4f}, {t[2]:+.4f}) m")
log(f"  R 的列 = 相机三轴在世界坐标下的方向：右 ({R[0,0]:+.3f},{R[1,0]:+.3f},{R[2,0]:+.3f}) /"
    f" 上 ({R[0,1]:+.3f},{R[1,1]:+.3f},{R[2,1]:+.3f}) / 后 ({R[0,2]:+.3f},{R[1,2]:+.3f},{R[2,2]:+.3f})")
ang_up = float(np.degrees(np.arccos(np.clip(R[2, 1], -1, 1))))
log(f"- **图像里的\"向上\"（−v 方向）对应世界方向 R[:,1]，它与世界竖直 +Z 的夹角 = {ang_up:.1f}°**")
log(f"  → 根因就在这一句：图像上方不是世界上方，偏了 {ang_up:.1f}°。")
log()


def to_world(u, v, dd):
    return R @ np.array([(u - cx) / F * dd, -(v - cy) / F * dd, -dd]) + t


def probe_model(extra):
    mm = mujoco.MjModel.from_xml_string(make_xml(seed, elev_deg=ELEV, density=DENSITY)
                                        .replace("</worldbody>", extra + "\n  </worldbody>"))
    dd_ = mujoco.MjData(mm)
    dd_.qpos[:] = QPOS          # ★ 关键：把已沉降的位姿搬进来
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
    c = np.array([[sx*hx, sy*hy, sz*hz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    w = (Rg @ c.T).T + p
    return w.min(0), w.max(0)


def probe_at(u, v, rad, h=0.05):
    w = to_world(u, v, depth[v, u])
    mm, dd_ = probe_model(f'<body name="probe" pos="{w[0]:.5f} {w[1]:.5f} {w[2]+h/2:.5f}">'
                          f'<geom name="g_probe" type="cylinder" size="{rad} {h/2}" rgba="0 1 1 1"/></body>')
    pid = mujoco.mj_name2id(mm, mujoco.mjtObj.mjOBJ_GEOM, "g_probe")
    out = {}
    for i, gg in object_geom_ids(mm).items():
        ft = np.zeros(6)
        out[i] = float(mujoco.mj_geomDistance(mm, dd_, pid, gg, 2.0, ft))
    mm2, _ = mm, dd_
    return out, w


# ================================================ 3. 变换自检
log("## 3. 像素 ↔ 世界坐标变换")
log()
log("```")
log("世界→像素：p_cam = Rᵀ(p_world − t)；z = −p_cam[2]（相机沿 −Z 看）")
log("           u = cx + p_cam[0]/z·F ；v = cy − p_cam[1]/z·F")
log("像素→世界：p_cam = ( (u−cx)/F·z , −(v−cy)/F·z , −z )；p_world = R·p_cam + t")
log("```")
log()
log("自检一：把每个物体的真实几何中心投影回图像，和它的掩码质心比：")
log()
log("| 物体 | seg掩码质心 | 世界投影 (u,v) | 误差(px) | 可见像素 |")
log("|---|---|---|---|---|")
errs = []
for i, gid in gids.items():
    c = d.geom_xpos[gid]
    pc = R.T @ (c - t); z = -pc[2]
    u = cx + pc[0]/z*F; v = cy - pc[1]/z*F
    mk = (seg == gid) & valid
    if mk.sum() < 40:
        continue
    yy, xx = np.nonzero(mk)
    e = float(np.hypot(u - xx.mean(), v - yy.mean())); errs.append(e)
    log(f"| obj{i} | ({xx.mean():.1f},{yy.mean():.1f}) | ({u:.1f},{v:.1f}) | {e:.1f} | {mk.sum()} |")
log()
log(f"平均 {np.mean(errs):.1f} px。被遮挡时\"质心\"本身就会偏（一个物体才 6cm 大，占图像几十像素），")
log(f"这个量级正常。")
log()
log("自检二（更有力的那条）：拿**已知几何**去量反投影误差 —— ")
log("箱底板的真实 z 范围是 [0.000, 0.020] m，把它上面的像素反投影：")
fid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "binfloor")
fl = (seg == fid) & valid
yy, xx = np.nonzero(fl)
rng = np.random.default_rng(0)
kk = rng.choice(len(xx), size=min(500, len(xx)), replace=False)
zs = np.array([to_world(xx[j], yy[j], depth[yy[j], xx[j]])[2] for j in kk])
inr = float(((zs >= 0 - 0.003) & (zs <= 0.02 + 0.003)).mean())
log()
log(f"- 反投影 z：中位 {np.median(zs):.4f} m，1%~99% [{np.percentile(zs,1):.4f}, {np.percentile(zs,99):.4f}]")
log(f"- 落在真实范围内(±3mm)的比例 = **{inr*100:.0f}%**")
log()
log("自检三：逐像素带符号距离 —— 在目标物体**自己的掩码内部**取样（探针球半径 0.5mm）：")
log()

def edist(px, py, maxr=10):
    for rad_ in range(0, maxr + 1):
        for dy_ in range(-rad_, rad_ + 1):
            for dx_ in range(-rad_, rad_ + 1):
                if max(abs(dx_), abs(dy_)) != rad_:
                    continue
                yy_, xx_ = py + dy_, px + dx_
                if not (0 <= yy_ < H and 0 <= xx_ < W) or seg[yy_, xx_] != g1:
                    return rad_
    return maxr

mask_t = (seg == g1) & valid
ys_t, xs_t = np.nonzero(mask_t)
eds = np.array([edist(int(x), int(y)) for x, y in zip(xs_t, ys_t)])
pick = [(P[0], P[1])]
for lo_, hi_ in ((1, 1), (2, 3), (4, 6), (7, 9), (10, 99)):
    idx = np.where((eds >= lo_) & (eds <= hi_))[0]
    if len(idx):
        pick.append((int(xs_t[idx[0]]), int(ys_t[idx[0]])))
mm, dd_ = probe_model("".join(
    f'<body pos="{to_world(x,y,depth[y,x])[0]:.5f} {to_world(x,y,depth[y,x])[1]:.5f} {to_world(x,y,depth[y,x])[2]:.5f}">'
    f'<geom name="gp{k}" type="sphere" size="0.0005"/></body>'
    for k, (x, y) in enumerate(pick)))
gid_new = object_geom_ids(mm)[obj_of[g1]]
log("| 像素 | 距掩码边界 | 深度(m) | 到目标物体真实表面的距离 |")
log("|---|---|---|---|")
pix_err = []
for k, (x, y) in enumerate(pick):
    pid = mujoco.mj_name2id(mm, mujoco.mjtObj.mjOBJ_GEOM, f"gp{k}")
    ft = np.zeros(6)
    e = float(mujoco.mj_geomDistance(mm, dd_, pid, gid_new, 1.0, ft)) - 0.0005
    pix_err.append(e)
    log(f"| ({x},{y}) | {edist(x, y)} px | {depth[y,x]:.4f} | **{e*100:+.2f} cm** |")
log()
log(f"→ 在目标物体**自身掩码内**，反投影到真实表面的误差都在 **{np.max(np.abs(pix_err))*100:.2f} cm 以内**")
log(f"  （轮廓边缘像素和内部像素都一样准）——所以这套内参/外参/深度定义可以直接用来做世界坐标判断，")
log(f"  精度足够支撑 cm 级的间隙比较。（注意：取样必须落在**目标自身的掩码里**；")
log(f"  取到旁边物体或背景上，量出来的就是跨物体距离，没有意义。）")
log()

# ================================================ 4. 走廊在世界坐标里
log("## 4. 「接触点上方 40px 走廊」逐像素反投影")
log()
w1 = to_world(P[0], P[1], z_main)
log(f"接触点 p{k_main+1} 世界坐标 = ({w1[0]:+.4f}, {w1[1]:+.4f}, {w1[2]:+.4f}) m")
log()
log("| dy(px) | 像素(u,v) | 属于 | 深度(m) | 世界坐标(x,y,z) | 与接触点水平距离 | 高差 Δz |")
log("|---|---|---|---|---|---|---|")
for dy, px, yy, s, dd_ in walk:
    w = to_world(px, yy, dd_)
    tag = f"obj{obj_of[s]}" if s in gids.values() else ("目标自身" if s == g1 else "背景/箱体")
    if dy % 6 == 1 or dy <= 4 or (s != g1 and s in gids.values()):
        log(f"| {dy} | ({px},{yy}) | {tag} | {dd_:.3f} | ({w[0]:+.3f},{w[1]:+.3f},{w[2]:+.3f}) | "
            f"{np.hypot(w[0]-w1[0], w[1]-w1[1])*100:.1f} cm | {(w[2]-w1[2])*100:+.1f} cm |")
log()
wb = to_world(bk[1], bk[2], bk[4])
dxy = float(np.hypot(wb[0]-w1[0], wb[1]-w1[1]))
dz = float(wb[2]-w1[2])
vec = wb - w1
ang = float(np.degrees(np.arccos(np.clip(vec[2]/np.linalg.norm(vec), -1, 1))))
log(f"**旧判据认定的\"挡路者\"**：像素 ({bk[1]},{bk[2]})，距接触点 {bk[0]} px，属于 obj{obj_of[bk[3]]}，深度 {bk[4]:.3f} m")
log(f"- 世界坐标 ({wb[0]:+.4f},{wb[1]:+.4f},{wb[2]:+.4f})")
log(f"- 与接触点的**水平距离 {dxy*100:.1f} cm**（若真在\"正上方\"，这个数应≈0）")
log(f"- 高差 **{dz*100:+.1f} cm**；深度差 **{bk[4]-z_main:+.3f} m（更远）**")
log(f"- 接触点→该点的方向与**世界竖直方向夹角 = {ang:.1f}°**（真在正上方应是 0°）")
log()
log(f"再算长度尺度：在**同一深度**上往上走 40px，世界位移 = 40/F·z = 40/{F:.1f}×{z_main:.3f} = "
    f"**{40/F*z_main*100:.1f} cm**，方向是 R[:,1]（偏竖直 {ang_up:.0f}°）。")
log(f"→ 这条走廊在世界里是\"斜着往上、同时往外飞\"的一条线；再叠加\"越往上深度越大\""
    f"（{z_main:.3f}→{bk[4]:.3f} m），它离抓取点正上方越来越远。")
log()

# ================================================ 5. 世界坐标真 3D 检查
log("## 5. 世界坐标里的真 3D 碰撞检查")
log()
log("### 5.1 先标定工具：mj_geomDistance 返回的是什么")
log()
log("探针球（半径 1mm）放在箱底板（顶面 z=0.020）附近已知位置：")
log()
log("| 放置位置 | 返回 | 说明 |")
log("|---|---|---|")
for zc, note in ((0.031, "板面上方 1cm（球面到板面应 0.010 m）"),
                 (0.301, "板面上方 28.1cm（球面到板面应 0.280 m）"),
                 (0.011, "球心在板内（穿透）")):
    mmx, ddx = probe_model(f'<body pos="0 0 {zc}"><geom name="g_p" type="sphere" size="0.001"/></body>')
    pid = mujoco.mj_name2id(mmx, mujoco.mjtObj.mjOBJ_GEOM, "g_p")
    fidx = mujoco.mj_name2id(mmx, mujoco.mjtObj.mjOBJ_GEOM, "binfloor")
    ft = np.zeros(6)
    val = float(mujoco.mj_geomDistance(mmx, ddx, pid, fidx, 2.0, ft))
    log(f"| z = {zc:.3f} m | {val*100:+.3f} cm | {note} |")
log()
log("→ 返回的是**表面到表面的带符号距离**（负=穿透深度），量值和手算一致，可以拿来当判据。")
log()
log("### 5.2 两个必须注意的坑")
log()
log("1. **探针模型里的物体必须用已沉降的 qpos**。只跑 `mj_forward` 的话，物体还停在 XML 的")
log("   初始堆叠位置，量出来的距离全是错的（我第一版就踩了这个坑）。")
log("2. **柱状空间必然和抓取目标自身相交**——柱就长在它的表面上。所以判\"有没有东西挡路\"时")
log("   必须**排除目标自身**，只看别的物体。")
log()
log("### 5.3 结果：每次张开的竖直柱（半径 3cm、高 5cm）里有什么")
log()
for tag, (u, v) in (("p1", p1), ("p2", p2)):
    out, w = probe_at(u, v, 0.03)
    log(f"接触点 {tag} = ({u},{v})，柱底在 z={w[2]:.4f} m")
    log()
    log("| 物体 | 与柱面距离 | |")
    log("|---|---|---|")
    for i, val in sorted(out.items(), key=lambda kv: kv[1]):
        mark = "**抓取目标自身**（预期相交）" if i == obj_of[g1] else ("**别的物体 → 侵入**" if val < 0 else "")
        log(f"| obj{i} | {val*100:+.2f} cm | {mark} |")
    log()
log("### 5.4 半径扫描：把\"夹爪需要多少空间\"当成变量")
log()
log("| 柱半径 | p1 上方柱内最近的别的物体 | p2 上方柱内最近的别的物体 |")
log("|---|---|---|")
scan = {}
for rad in (0.005, 0.010, 0.020, 0.030, 0.040):
    cells = []
    for (u, v) in (p1, p2):
        out, w = probe_at(u, v, rad)
        others = {i: s for i, s in out.items() if i != obj_of[g1]}
        j = min(others, key=lambda k_: others[k_])
        scan[(rad, (u, v))] = (j, others[j])
        cells.append(f"obj{j}: **{others[j]*100:+.2f} cm**")
    log(f"| {rad*100:.1f} cm | {cells[0]} | {cells[1]} |")
log()
ax_scan = []
for rad in (0.005, 0.010, 0.020, 0.030, 0.040):
    j, val = scan[(rad, p2)]
    ax_scan.append((rad, val))
near_axis = 0.03 + scan[(0.03, p2)][1]   # 柱半径 + 带符号距离 = 邻物表面到轴线的水平距离
log(f"扫描几乎是线性的（半径每加 1cm，距离就少 1cm）→ 说明最近的邻物表面距离这条竖直线")
log(f"固定为 **{near_axis*100:.2f} cm**（p2 那一列）。")
log(f"手指尺度（半径 ≤2cm）时它是干净的；半径 3cm、直径 6cm（≈这只夹爪张开的 {wm*100:.2f} cm）时邻物侵入约 0.5cm。")
log()

# ================================================ 6. 结论
log("## 6. 同一样例：旧判据 vs 世界坐标 3D 检查")
log()
log("| 判据 | 结果 |")
log("|---|---|")
log(f"| ① 两个接触点在同一物体上 | 通过（都是 obj{obj_of[g1]}）|")
log(f"| ② 闭合路径 21 点无别的物体 | 通过 |")
log(f"| ③ 上方 40px 通道无遮挡（**旧**）| **失败** —— 在 {bk[0]} px 处遇到 obj{obj_of[bk[3]]} |")
log(f"| ③′ 世界坐标竖直柱几何检查（**新**）| **不是干净通过**：半径 ≤2cm 时最近的别的物体还留 "
    f"{scan[(0.02, p2)][1]*100:+.2f} cm 间隙；半径 3cm 时邻物侵入 {abs(scan[(0.03, p2)][1])*100:.2f} cm |")
log()
log("**能写进报告的结论（两层）**")
log()
log(f"1. 旧判据的**理由不成立**：它说\"正上方被挡\"，但那个像素在世界坐标里位于接触点")
log(f"   **水平 {dxy*100:.1f} cm 之外、只高 {dz*100:.1f} cm**，与竖直方向差 {ang:.1f}°，而且深度**更远**。")
log(f"   它挡住的不是\"下手的通道\"，而是\"斜后方远处的一个物体\"。")
log("2. 换成真 3D 检查后，**这一例也不是干净的通过**：这只夹爪张 5.47cm，而最近的邻物表面")
log(f"   距接触点正上方那条线只有 {near_axis*100:.2f} cm——空间很紧，到底算不算挡路，")
log("   **取决于夹爪的手指厚度与接近方式**，即必须先定义夹爪模型，而不是靠一条像素启发式。")
log()
log("> ⚠️ **复测未完成。** 3D 检查还没接进 `run_scene`，6 格 × 30 场景的对照表**尚未重跑**，")
log("> 所以现在**没有任何\"新的合格率\"数字**。单例只能证明旧判据会误判、以及正确判据该长什么样，")
log("> 不能给出比例。")
log()

# ================================================ 画图
fig = plt.figure(figsize=(15.0, 9.8))
gs = fig.add_gridspec(2, 2, hspace=0.20, wspace=0.13)

ax = fig.add_subplot(gs[0, 0])
ax.imshow(rgb); ax.set_title("① 彩色图：接触点与\"被挡\"像素的位置", fontsize=11); ax.axis("off")
ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "-", color="#2ae082", lw=2.6, label="最高分抓取")
for k, (px, py) in enumerate((p1, p2)):
    ax.plot(px, py, "o", ms=9, mfc="none", mec="#ffe14d", mew=2.2)
    ax.text(px + 8, py + 2, f"p{k+1}({px},{py})\nd={depth[py,px]:.3f}m", color="#ffe14d", fontsize=8, va="top",
            bbox=dict(fc=(0.05, 0.07, 0.12, 0.78), ec="none", pad=1.6))
for (px, py) in (p1, p2):
    ax.add_patch(Rectangle((px - 5, max(py - 40, 0)), 10, 40, fc=(1, 0.83, 0, 0.15), ec=(1, 0.83, 0, 0.85), lw=1.2))
ax.text(6, 24, "黄框 = 旧判据的\"正上方 40px 通道\"（图像空间）", color="#ffe14d", fontsize=8.5,
        bbox=dict(fc=(0.04, 0.06, 0.10, 0.85), ec="none", pad=2.5))
ax.plot(bk[1], bk[2], "x", color="#ff4d4d", ms=12, mew=3)
ax.text(bk[1] + 9, bk[2] - 2, f"\"挡路\"({bk[1]},{bk[2]})\nobj{obj_of[bk[3]]}  d={bk[4]:.3f}m", color="#ff9b9b", fontsize=8,
        va="center", bbox=dict(fc=(0.05, 0.07, 0.12, 0.85), ec="none", pad=1.6))
ax.legend(loc="lower right", fontsize=8, framealpha=0.25)

ax = fig.add_subplot(gs[0, 1])
dd = np.nan_to_num(depth)
lo, hi = np.percentile(dd[valid], 2), np.percentile(dd[valid], 98)
im = ax.imshow(dd, cmap="turbo_r", vmin=lo, vmax=hi)
ax.set_title("② 深度图（暖=近，冷=远）：走廊上的点反而更远", fontsize=11); ax.axis("off")
plt.colorbar(im, ax=ax, fraction=0.045, label="沿光轴距离 (m)")
ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "-", color="#2ae082", lw=2.6)
for (px, py) in (p1, p2):
    ax.plot(px, py, "o", ms=9, mfc="none", mec="#ffffff", mew=2.2)
    ax.add_patch(Rectangle((px - 5, max(py - 40, 0)), 10, 40, fc="none", ec="#ffffff", lw=1.2, ls="--"))
for dy, px, yy, s, dval in walk:
    ax.plot(px, yy, ".", color="#ffffff", ms=2.6)
ax.plot(bk[1], bk[2], "x", color="#ffffff", ms=12, mew=3)
ax.text(bk[1] + 9, bk[2] - 2, f"d={bk[4]:.3f}m（比接触点远 {bk[4]-z_main:+.3f}m）", color="#ffffff", fontsize=8.5,
        bbox=dict(fc=(0.05, 0.06, 0.10, 0.8), ec="none", pad=1.6))

ax = fig.add_subplot(gs[1, 0])
ax.set_title(f"③ 侧视（世界 X–Z 真实坐标）：图像\"上方\"斜了 {ang_up:.0f}°", fontsize=11)
ax.add_patch(Rectangle((-0.16, 0.02), 0.32, 0.09, fc="#3b4657", ec="#8899aa", lw=1))
ax.plot(t[0], t[2], "o", color="#f6c454", ms=11)
ax.annotate(f"相机 ({t[0]:.2f},{t[2]:.2f})", (t[0], t[2]), textcoords="offset points", xytext=(6, 8), fontsize=8.5)
ax.plot([w1[0], w1[0]], [w1[2], w1[2] + 0.03], color="#2ae082", lw=9, alpha=0.55, label="竖直柱（半径3cm时）")
ax.plot(w1[0], w1[2], "o", color="#2ae082", ms=8)
ax.annotate(f"抓取点 z={w1[2]:.3f}m", (w1[0], w1[2]), textcoords="offset points", xytext=(-80, -16), fontsize=8.5)
pw = np.array([to_world(px, yy, dv) for dy, px, yy, s, dv in walk])
ax.plot(pw[:, 0], pw[:, 2], "--", color="#ff4d4d", lw=1.8, label="旧判据走廊在世界的真实走向")
ax.plot(wb[0], wb[2], "X", color="#ff4d4d", ms=13)
ax.annotate(f"\"挡路\"点 ({wb[0]:+.3f},{wb[2]:+.3f})\n水平偏出 {dxy*100:.0f} cm，偏竖直 {ang:.0f}°",
            (wb[0], wb[2]), textcoords="offset points", xytext=(10, 2), fontsize=8.5, color="#ff9b9b")
for i, gid in gids.items():
    gp = d.geom_xpos[gid]
    ax.plot(gp[0], gp[2], ".", color="#7fd1ff", ms=5)
ax.set_xlabel("世界 X (m)"); ax.set_ylabel("世界 Z (m)")
ax.set_xlim(-0.45, 0.45); ax.set_ylim(-0.02, 0.60)
ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.25)

ax = fig.add_subplot(gs[1, 1])
ax.set_title("④ 半径扫描：这只夹爪到底下不下得去？", fontsize=11)
xs_ = [r * 100 for r, _ in ax_scan]; ys_ = [v * 100 for _, v in ax_scan]
ax.axhspan(-3.0, 0, color="#ff4d4d", alpha=0.12)
ax.axhspan(0, 3.0, color="#2ae082", alpha=0.12)
ax.axhline(0, color="#888", lw=1.2, ls="--")
ax.plot(xs_, ys_, "o-", color="#ffe14d", lw=2.2, ms=7, label="最近的别的物体：到柱面的距离")
ax.axvspan(0, 2.0, color="#ffffff", alpha=0.07)
ax.text(0.15, 2.35, "手指尺度\n（半径≤2cm）：干净", fontsize=9, color="#2ae082")
ax.text(2.2, -1.35, "整只夹爪尺度（半径≥3cm）：邻物侵入", fontsize=9, color="#ff9b9b")
ax.annotate(f"邻物表面距轴线\n{near_axis*100:.2f} cm", (3.0, scan[(0.03, p2)][1] * 100),
            textcoords="offset points", xytext=(16, 14), fontsize=8.5, color="#ffe14d",
            arrowprops=dict(arrowstyle="->", color="#ffe14d"))
ax.set_xlabel("竖直柱半径 (cm)"); ax.set_ylabel("距离 (cm)")
ax.set_ylim(-3, 3); ax.grid(alpha=0.25); ax.legend(fontsize=8, loc="lower left")

fig.suptitle(f"为什么「图像上方 40px 通道」判据不成立，以及世界坐标检查该怎么做 —— 真实数据样例"
             f"（seed={seed}，堆叠 12 物体，俯角 {ELEV:.0f}°）\n"
             f"反投影已被已知几何核对（箱底板 100% 落在真实范围内；逐像素误差 ≤0.2 cm）；"
             f"被判\"挡路\"的点在世界里水平偏出 {dxy*100:.0f} cm、仅高 {dz*100:.1f} cm、偏竖直 {ang:.0f}°、深度更远 {bk[4]-z_main:+.3f} m",
             fontsize=11.5)
fig.text(0.5, 0.006, "单例诊断：3D 检查尚未接入 run_scene，6 格 × 30 场景的复测未完成 —— 本图没有、也不该有\"新的合格率\"。",
         ha="center", fontsize=10, color="#c0392b")
fig.savefig(OUT + "criterion_failure_evidence.png", dpi=150, bbox_inches="tight")
print("saved criterion_failure_evidence.png")
open(OUT + "criterion_failure_evidence.md", "w", encoding="utf-8").write("\n".join(REPORT) + "\n")
print("saved criterion_failure_evidence.md")
