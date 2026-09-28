# -*- coding: utf-8 -*-
"""
check_projection.py —— 自检：深度图的坐标约定到底对不对
======================================================
判据全部建在"把深度图像素反投影回世界坐标"上，所以这一步必须先确认。
三件事：

  1. 像素 → 世界 → 像素 往返，看偏差多少像素
  2. 已知高度的箱底（世界 z ∈ [0, 0.02]），反投影回去看落在什么范围
  3. 相机位姿和图像尺寸，与 criterion_failure_evidence.md 里推导的内参一致

跑法：./.venv/Scripts/python.exe check_projection.py
"""

import numpy as np
import mujoco

from bin_picking_demo import make_xml, W, H, F
from criterion3d import camera_pose, pixel_to_world, world_to_pixel, N_STEPS

SEED, ELEV = 14, 30.0

model = mujoco.MjModel.from_xml_string(make_xml(SEED, elev_deg=ELEV, density="dense"))
data = mujoco.MjData(model)
for _ in range(N_STEPS):
    mujoco.mj_step(model, data)

r = mujoco.Renderer(model, height=H, width=W)
r.enable_depth_rendering()
r.update_scene(data, camera="cam")
depth = np.asarray(r.render()).copy()
r.enable_segmentation_rendering()
r.update_scene(data, camera="cam")
seg = np.asarray(r.render()).copy()[:, :, 0]
r.close()

valid = (depth > 0.05) & (depth < 2.0)
R, t = camera_pose(data)
print("图像 %dx%d  fovy 45°  ->  F = %.1f px，主点 (%.1f, %.1f)" % (W, H, F, W / 2, H / 2))
print("相机位置 %s" % np.round(t, 3))
print("相机 R 的第三行（世界里的'看的方向'的反向）%s" % np.round(R[2], 3))

# 1. 往返
rng = np.random.default_rng(0)
ys, xs = np.nonzero(valid & (seg > 0))
picked = rng.choice(len(xs), size=min(500, len(xs)), replace=False)
dmax = 0.0
for k in picked:
    u, v = float(xs[k]), float(ys[k])
    P = pixel_to_world(u, v, float(depth[int(v), int(u)]), R, t)
    u2, v2, _ = world_to_pixel(P, R, t)
    dmax = max(dmax, abs(u2 - u), abs(v2 - v))
print("\n1) 像素→世界→像素 最大偏差: %.4f px  (%d 个像素)" % (dmax, len(picked)))

# 2. 箱底：反投影回世界坐标应落在 z ∈ [0, 0.02] 附近
zs = []
for k in picked[:2000]:
    u, v = float(xs[k]), float(ys[k])
    if seg[int(v), int(u)] == 0:
        continue
    P = pixel_to_world(u, v, float(depth[int(v), int(u)]), R, t)
    zs.append(P[2])
zs = np.array(zs)
# 只取箱内、排除台面：台面在 z≈0 以下一大片
inside = (np.abs(zs) <= 0.5)
print("2) 反投影点的世界高度: 中位 %.4f m  5%%~95%% 区间 [%.4f, %.4f] m"
      % (float(np.median(zs)), float(np.percentile(zs, 5)), float(np.percentile(zs, 95))))

# 3. 单点带符号距离的 API 标定（用两个已知相距的盒子验证）
xml = ('<mujoco><worldbody>'
       '<geom name="a" type="box" size="0.01 0.01 0.01"/>'
       '<geom name="b" type="box" size="0.01 0.01 0.01" pos="0.1 0 0" contype="0" conaffinity="0"/>'
       '</worldbody></mujoco>')
m2 = mujoco.MjModel.from_xml_string(xml)
d2 = mujoco.MjData(m2)
mujoco.mj_forward(m2, d2)
ft = np.zeros(6)
print("\n3) mj_geomDistance 标定：两个 1 cm 盒子中心相距 10 cm，表面距离应为 8 cm -> %.4f m"
      % float(mujoco.mj_geomDistance(m2, d2, 0, 1, 1.0, ft)))
