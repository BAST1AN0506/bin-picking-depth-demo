# -*- coding: utf-8 -*-
"""
bin_picking_demo.py  ——  最小可运行的"料箱抓取点选择"仿真 demo
==============================================================
做什么：
  1. 用 MuJoCo 搭一个料箱（bin）+ 随机摆放的物体，渲染 彩色图 / 深度图 / 分割图
  2. 只从 **深度图 + 分割图** 里，为每个物体找出"平行夹爪能抓"的位置与角度
     （经典 2D 抓取矩形思路：找一个方向，使物体在该方向上的宽度能塞进夹爪，
       且夹持面足够平坦 —— 关键：用"物体在该方向的宽度"当夹爪开合宽度）
  3. 把抓取矩形画回彩色图，并用分割图（真值）检查这个抓取是否合格：
     两个接触点在同一个物体上、且夹爪路径中间没有别的物体
  4. 跑 N 个随机场景，输出"选出率 / 合格率"

跑法：
  python bin_picking_demo.py          # 默认 20 个场景
  python bin_picking_demo.py 50       # 50 个场景

产出（同目录）：
  demo_result.png     4 联图：带抓取框的彩色图 / 原彩色图 / 深度图 / 抓取质量热图
  demo_summary.txt    统计结果
"""

import os
import sys
import numpy as np
import mujoco
from PIL import Image, ImageDraw

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

W, H, FOVY = 480, 360, 45.0           # 相机分辨率、垂直视场角
GRIP_MIN, GRIP_MAX, GRIP_IDEAL = 0.025, 0.085, 0.055   # 夹爪可张开宽度范围 / 最舒服的宽度
F = (H / 2) / np.tan(np.deg2rad(FOVY) / 2)             # 焦距（像素）


# ---------------------------------------------------------------- 1. 场景
def make_xml(seed, elev_deg=40.0, dist=0.90, target_h=0.10, density="sparse", n_obj=None):
    """elev_deg = 相机俯角(度)：90=正上方往下看，越小越斜
       density  = "sparse"（6 个物体、摊开）或 "dense"（12 个物体、堆叠）"""
    th = np.deg2rad(elev_deg)
    cam_pos = f"0.0 {-dist * np.cos(th):.3f} {target_h + dist * np.sin(th):.3f}"
    cam_axes = f"1 0 0 0 {np.sin(th):.4f} {np.cos(th):.4f}"
    rng = np.random.default_rng(seed)
    if n_obj is None:
        n_obj = 6 if density == "sparse" else 12
    spread = 0.09 if n_obj <= 6 else (0.105 if n_obj <= 12 else 0.118)
    step_z = 0.08 if n_obj <= 6 else (0.075 if n_obj <= 12 else 0.065)
    objs = []
    for i in range(n_obj):
        kind = ["box", "box", "cylinder", "sphere"][i % 4]
        x, y = rng.uniform(-spread, spread, 2)
        z = 0.07 + i * step_z
        rgba = f"{rng.uniform(.3,1):.2f} {rng.uniform(.3,1):.2f} {rng.uniform(.3,1):.2f} 1"
        size = {"box": "0.030 0.030 0.030", "cylinder": "0.028 0.040", "sphere": "0.033"}[kind]
        objs.append(f'<body name="obj{i}" pos="{x:.3f} {y:.3f} {z:.3f}">'
                    f'<freejoint/><geom name="g_obj{i}" type="{kind}" size="{size}" rgba="{rgba}"/></body>')
    walls = "".join(f'<geom type="box" size="{s}" pos="{p}" rgba=".35 .35 .4 1"/>' for s, p in [
        ("0.16 0.008 0.06", "0 0.15 0.06"), ("0.16 0.008 0.06", "0 -0.15 0.06"),
        ("0.008 0.15 0.06", "0.15 0 0.06"), ("0.008 0.15 0.06", "-0.15 0 0.06")])
    return f"""
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <light pos="0.2 -0.3 1.2" dir="-0.2 0.3 -1"/>
    <light pos="-0.3 0.2 1.0" dir="0.3 -0.2 -1"/>
    <geom name="table" type="plane" size="1 1 0.1" rgba=".65 .65 .68 1"/>
    <geom name="binfloor" type="box" size="0.16 0.16 0.01" pos="0 0 0.01" rgba=".75 .75 .78 1"/>
    {walls}
    {''.join(objs)}
    <camera name="cam" pos="{cam_pos}" xyaxes="{cam_axes}" fovy="{FOVY}"/>
  </worldbody>
</mujoco>
"""


def object_geom_ids(model):
    ids = {}
    for g in range(model.ngeom):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if nm.startswith("g_obj"):
            ids[int(nm[5:])] = g
    return ids


# ------------------------------------------------- 2. 深度图 -> 3D 点
def depth_to_points(depth):
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    x = (u - W / 2) / F * depth
    y = -(v - H / 2) / F * depth
    return np.stack([x, y, depth], axis=-1)


# ------------------------------------- 3. 为每个物体找最优 2D 抓取矩形（核心）
def grasp_for_object(mask, depth, seg=None, gid=None, require_reachable=False):
    """
    在一个物体的可见区域内搜索抓取：
      对每个方向 u=(cosθ,sinθ)，把物体像素投影到 u 上；
      在垂直于 u 的方向上取若干条"扫过线"（band），
      该 band 上物体在 u 方向的跨度 = 夹爪需要张开的宽度。
    评分 = 宽度是否合适(0.5) + 夹持面是否平坦(0.5)
    """
    ys, xs = np.nonzero(mask)
    if len(xs) < 60:
        return None
    gids_set = set(np.unique(seg).tolist()) - {0} if seg is not None else set()
    cx, cy = xs.mean(), ys.mean()
    z0 = float(np.median(depth[mask]))
    best = None
    for deg in range(0, 180, 10):                      # 方向
        th = np.deg2rad(deg)
        ux, uy = np.cos(th), np.sin(th)
        pu = (xs - cx) * ux + (ys - cy) * uy           # 沿夹爪闭合方向的投影
        pv = -(xs - cx) * uy + (ys - cy) * ux          # 垂直方向
        for off in (-8, -4, 0, 4, 8):                  # 扫过线相对质心的偏移(px)
            band = np.abs(pv - off) <= 3
            if band.sum() < 20:
                continue
            width_px = pu[band].max() - pu[band].min()
            width_m = width_px * z0 / F
            if not (GRIP_MIN <= width_m <= GRIP_MAX):
                continue
            # 夹持面平坦度：band 内深度的离散程度（越小越平，抓得越稳）
            rough = float(np.std(depth[mask][band]))
            s_w = 1.0 - min(abs(width_m - GRIP_IDEAL) / 0.035, 1.0)
            s_s = 1.0 - min(rough / 0.012, 1.0)
            score = 0.5 * s_w + 0.5 * s_s
            if best is None or score > best[0]:
                i1, i2 = np.argmin(pu[band]), np.argmax(pu[band])
                p1 = (int(xs[band][i1]), int(ys[band][i1]))     # 接触点 1
                p2 = (int(xs[band][i2]), int(ys[band][i2]))     # 接触点 2
                if require_reachable and seg is not None:
                    # 可达性检查：两个接触点正上方 40px 的通道里不能有别的物体
                    blocked = False
                    for (px, py) in (p1, p2):
                        for dy in range(1, 41, 3):
                            yy = py - dy
                            if yy < 0:
                                break
                            s = seg[yy, px]
                            if (s != gid) and (s in gids_set):
                                blocked = True
                                break
                        if blocked:
                            break
                    if blocked:
                        continue
                cu, cv = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
                best = (score, cu, cv, deg, width_m, width_px, z0, p1, p2)
    return best


def nms(grasps, min_px=40, top=3):
    out = []
    for g in sorted(grasps, key=lambda x: -x[0]):
        if all((g[1] - o[1]) ** 2 + (g[2] - o[2]) ** 2 > min_px ** 2 for o in out):
            out.append(g)
    return out[:top]


# ------------------------------------------------------------------ 4. 主流程
def run_scene(seed, want_image=False, elev_deg=40.0, img_path=None, density="sparse", reach=False, n_obj=None):
    model = mujoco.MjModel.from_xml_string(make_xml(seed, elev_deg=elev_deg, density=density, n_obj=n_obj))
    data = mujoco.MjData(model)
    for _ in range(700):
        mujoco.mj_step(model, data)

    r = mujoco.Renderer(model, height=H, width=W)
    # MuJoCo 的 render() 每次只返回"最后启用的"buffer：启用一种 → 渲染 → 再启用下一种
    r.update_scene(data, camera="cam"); rgb = np.asarray(r.render()).copy()
    r.enable_depth_rendering()
    r.update_scene(data, camera="cam"); depth = np.asarray(r.render()).copy()
    r.enable_segmentation_rendering()
    r.update_scene(data, camera="cam"); seg = np.asarray(r.render()).copy()
    r.close()

    seg = seg[:, :, 0]                                  # 第 0 通道 = 几何体 id
    valid = (depth > 0.05) & (depth < 2.0)
    pts = depth_to_points(depth)
    gids = object_geom_ids(model)

    grasps = []
    for i, gid in gids.items():
        mask = (seg == gid) & valid
        if mask.sum() < 60:
            continue
        g = grasp_for_object(mask, depth, seg=seg, gid=gid, require_reachable=reach)
        if g:
            grasps.append(g)
    top = nms(grasps)

    # 合格判定（用分割图真值）：两个接触点必须落在同一个物体上，
    # 且两点之间的夹爪路径上不能有别的物体（允许经过空白/箱体）
    ok = False
    if top:
        _, cu, cv, deg, width_m, width_px, z0, p1, p2 = top[0]
        g1, g2 = seg[p1[1], p1[0]], seg[p2[1], p2[0]]
        same = (g1 == g2) and (g1 in gids.values())
        ts = np.linspace(0, 1, 21)
        path = [seg[int(round(p1[1] + t * (p2[1] - p1[1]))), int(round(p1[0] + t * (p2[0] - p1[0])))]
                for t in ts]
        clear = all((s == g1) or (s == 0) or (s not in gids.values()) for s in path)
        # 夹爪要从"上方"接近：两个接触点正上方 40px 的通道里不能有别的物体
        def corridor_free(px, py):
            for dy in range(1, 41, 3):
                yy = py - dy
                if yy < 0:
                    break
                s = seg[yy, px]
                if (s != g1) and (s in gids.values()):
                    return False
            return True
        approach = corridor_free(p1[0], p1[1]) and corridor_free(p2[0], p2[1])
        ok = bool(same and clear and approach)

    if want_image:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # 抓取质量热图：把每个候选抓取按高斯核涂上去（越亮=越好抓）
        quality = np.zeros((H, W), np.float32)
        yy, xx = np.mgrid[0:H, 0:W]
        for s, u, v, deg, wm, wpx, z, p1, p2 in grasps:
            g = np.exp(-(((xx - u) ** 2 + (yy - v) ** 2) / (2 * 25.0 ** 2)))
            quality = np.maximum(quality, g.astype(np.float32) * s)

        img = Image.fromarray(rgb)
        dr = ImageDraw.Draw(img)
        # 最优抓取所在的物体：绿色描边，方便一眼看出抓的是哪个
        if top:
            gid_ok = seg[int(top[0][2]), int(top[0][1])]
            m = (seg == gid_ok)
            edge = m & ~(np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1))
            for y, x in zip(*np.nonzero(edge)):
                if 0 <= y < H and 0 <= x < W:
                    dr.point((x, y), fill=(0, 210, 90))
        for i, (s, cu, cv, deg, wm, wpx, z, p1, p2) in enumerate(top):
            col = (0, 210, 90) if i == 0 else (255, 200, 0)
            dr.line([p1[0], p1[1], p2[0], p2[1]], fill=col, width=3)
            for p in (p1, p2):
                dr.ellipse([p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4], outline=col, width=2)
            dr.text((p1[0] + 6, p1[1] - 14), f"{s:.2f} {wm*100:.1f}cm", fill=col)

        # 深度图：用 turbo 色表显示（近=暖色，远=冷色），按 2%~98% 分位拉伸
        d = np.nan_to_num(depth)
        lo, hi = np.percentile(d[valid], 2), np.percentile(d[valid], 98)
        dn = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
        dep_rgb = (plt.get_cmap("turbo")(dn)[:, :, :3] * 255).astype(np.uint8)
        # 质量热图
        q_rgb = (plt.get_cmap("inferno")(np.clip(quality, 0, 1))[:, :, :3] * 255).astype(np.uint8)

        panel = Image.new("RGB", (W * 2 + 6, H * 2 + 6), "white")
        panel.paste(img, (0, 0))                                  # 左上：抓取结果
        panel.paste(Image.fromarray(dep_rgb), (0, H + 6))         # 左下：深度图
        panel.paste(Image.fromarray(rgb), (W + 6, 0))             # 右上：原图
        panel.paste(Image.fromarray(q_rgb), (W + 6, H + 6))       # 右下：抓取质量热图
        panel.save(img_path or os.path.join(OUT_DIR, "demo_result.png"))
    return len(top) > 0, ok, len(grasps)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    picked = valid_g = total = 0
    for i in range(n):
        has, ok, k = run_scene(seed=i, want_image=(i == 0))
        picked += has; valid_g += ok; total += k
        print(f"scene {i:>2}: 每物候选 {k:>2}  选出最优={'Y' if has else 'N'}  合格={'Y' if ok else 'N'}")
    lines = [f"场景数          : {n}",
             f"选出抓取的场景   : {picked}/{n} = {picked/n*100:.0f}%",
             f"合格抓取的场景   : {valid_g}/{n} = {valid_g/n*100:.0f}%",
             f"平均候选数/场景  : {total/n:.1f}"]
    print("\n".join(lines))
    open(os.path.join(OUT_DIR, "demo_summary.txt"), "w", encoding="utf-8").write("\n".join(lines))
