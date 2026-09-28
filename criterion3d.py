# -*- coding: utf-8 -*-
"""
criterion3d.py —— 把接近判据从 2D 图像空间搬到世界坐标 3D，然后重跑那 6 格 × 30 场景
=================================================================================
要解决的问题（criterion_failure_evidence.md 里的结论）：
旧判据用"接触点正上方 40 px 通道里没有别的物体"来判断夹爪能不能下得去手。
相机俯角 30° 时图像里的"上"与世界竖直差 30°，那条通道打到的是斜后方远处的物体，
所以旧判据的失败理由不成立。正确的问题得在世界坐标里问。

三种判据，同一批场景、同一批种子（0–29）：

  A  旧判据  2D 图像空间：闭合路径无遮挡 + 接触点正上方 40 px 通道无遮挡
             （复刻 census.py 的判定逻辑，逐字对照）
  B  新判据  单张深度图：把深度图反投影成世界坐标点云，看夹爪两个手指的
             "指尖段 + 指体段"扫过的那两个盒子有没有被点云穿过
  D  参考真值 仿真真实几何：把同样四个盒子作为几何体加进模型，用 mj_geomDistance
             算带符号表面距离。这一列在真实系统里拿不到（真实系统只有传感器数据），
             放进来是为了量出"单视角点云漏掉了多少东西"

A 与 B/D 的区别只在"接近性"这一条，前面几条（两点同物体、闭合路径）三种判据共用。
指厚扫 0.5/1/2/3 cm，因为指厚是选夹爪时才定的参数。

输出（同目录）：
  criterion3d_result.md    新判据的数字表（引用这一份）
  criterion3d_scenes.csv   每个场景的逐条记录，可核对
  criterion3d_figure.png   两张图：三种判据的合格率 / 指厚敏感性

跑法：
  ./.venv/Scripts/python.exe criterion3d.py            # 6 格 × 30 场景
  ./.venv/Scripts/python.exe criterion3d.py 6          # 每格 6 个场景（冒烟测试）
"""

import csv
import os
import sys
import time

import numpy as np
import mujoco

from bin_picking_demo import (make_xml, object_geom_ids, grasp_for_object, nms,
                              W, H, F)
from gripper_model import GRIPPER, finger_boxes, points_inside, antipodality_deg

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
THICKNESSES = [0.005, 0.010, 0.020, 0.030]      # 指厚扫描（米）
CANON = 0.010                                   # 默认指厚
CELLS = [(30, "sparse"), (45, "sparse"), (60, "sparse"),
         (30, "dense"), (45, "dense"), (60, "dense")]
N_STEPS = 700

FINGER_GEOMS = "".join(
    '<geom name="finger%d" type="box" size="0.001 0.001 0.001" pos="0 0 -5" '
    'contype="0" conaffinity="0" group="4"/>' % i for i in range(4))


def xml_with_fingers(seed, elev, density):
    """在场景里加四个不可碰撞、也不参与渲染的盒子几何体，专门用来量距离。"""
    return make_xml(seed, elev_deg=elev, density=density).replace(
        "</worldbody>", FINGER_GEOMS + "\n  </worldbody>")


# --------------------------------------------------------------- 坐标转换
def camera_pose(data, cam_id=0):
    return np.array(data.cam_xmat[cam_id]).reshape(3, 3), np.array(data.cam_xpos[cam_id])


def pixel_to_world(u, v, z, R, t):
    """像素 + 沿光轴深度 -> 世界坐标（与 criterion_failure_evidence.md 的推导一致）"""
    pc = np.array([(u - W / 2) / F * z, -(v - H / 2) / F * z, -z])
    return R @ pc + t


def world_to_pixel(P, R, t):
    pc = R.T @ (np.asarray(P, float) - t)
    z = -pc[2]
    return W / 2 + pc[0] / z * F, H / 2 - pc[1] / z * F, z


def cloud_from_depth(depth, seg, R, t, valid):
    """把有效像素反投影成世界点云，带上每个点所属的几何体 id。"""
    vv, uu = np.nonzero(valid)
    z = depth[vv, uu]
    x = (uu - W / 2) / F * z
    y = -(vv - H / 2) / F * z
    cam = np.stack([x, y, -z], axis=1)                  # 相机系（前方 = -Z）
    pts = cam @ R.T + t
    return pts, seg[vv, uu]


# --------------------------------------------------------------- 判据
def admissible(seg, gids, p1, p2):
    """两种判据共用的前置条件：两点同物体 + 图像上闭合路径无遮挡。"""
    g1, g2 = seg[p1[1], p1[0]], seg[p2[1], p2[0]]
    if not ((g1 == g2) and (g1 in gids.values())):
        return False, g1, "两点不在同一物体"
    ts = np.linspace(0, 1, 21)
    for tt in ts:
        s = seg[int(round(p1[1] + tt * (p2[1] - p1[1]))),
                int(round(p1[0] + tt * (p2[0] - p1[0])))]
        if not ((s == g1) or (s == 0) or (s not in gids.values())):
            return False, g1, "闭合路径被挡"
    return True, g1, ""


def old_criterion_2d(seg, gids, g1, p1, p2):
    """A：复刻 census.py 的正上方 40 px 通道检查（逐字对照，别改）。"""
    for px, py in (p1, p2):
        for dy in range(1, 41, 3):
            yy = py - dy
            if yy < 0:
                break
            s = seg[yy, px]
            if (s != g1) and (s in gids.values()):
                return False
    return True


def new_criterion_cloud(pts, labels, boxes, target_gid, tol):
    """B：点云版本。指尖段只挡非目标点；指体段任何点都挡（目标鼓出来也算）。"""
    for kind, center, axes, half in boxes:
        local = (pts - center) @ axes
        inside = np.all(np.abs(local) <= (half - tol), axis=1)
        if not inside.any():
            continue
        if kind == "tip":
            if (labels[inside] != target_gid).any():
                return False
        else:
            return False
    return True


def new_criterion_exact(model, data, boxes, target_gid, tol):
    """D：把同样的盒子当几何体，用 mj_geomDistance 量真实几何。"""
    name2gid = {}
    for i in range(model.ngeom):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
        if nm.startswith("finger"):
            name2gid[nm] = i
    ids = [name2gid["finger%d" % k] for k in range(4)]
    for k, (kind, center, axes, half) in enumerate(boxes):
        gid = ids[k]
        model.geom_pos[gid] = center
        model.geom_size[gid] = half
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, axes.flatten())
        model.geom_quat[gid] = quat
    mujoco.mj_forward(model, data)
    fromto = np.zeros(6)
    for k, (kind, center, axes, half) in enumerate(boxes):
        gid = ids[k]
        for other in range(model.ngeom):
            if other == gid:
                continue
            nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other) or ""
            if nm.startswith("finger"):
                continue
            if kind == "tip" and other == target_gid:
                continue                        # 指尖贴住目标是正常的
            dist = float(mujoco.mj_geomDistance(model, data, gid, other, 0.30, fromto))
            if dist < -tol:                     # 侵入超过 2 mm 才算挡
                return False
    return True


def exact_min_clearance(model, data, boxes, target_gid):
    """D 的连续版本：返回四个盒子到真实几何的最小带符号距离（负 = 侵入）。"""
    ids = []
    for i in range(model.ngeom):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
        if nm.startswith("finger"):
            ids.append(i)
    ids.sort()
    for k, (kind, center, axes, half) in enumerate(boxes):
        gid = ids[k]
        model.geom_pos[gid] = center
        model.geom_size[gid] = half
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, axes.flatten())
        model.geom_quat[gid] = quat
    mujoco.mj_forward(model, data)
    fromto = np.zeros(6)
    worst = np.inf
    for k, (kind, center, axes, half) in enumerate(boxes):
        gid = ids[k]
        for other in range(model.ngeom):
            if other == gid:
                continue
            nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other) or ""
            if nm.startswith("finger"):
                continue
            if kind == "tip" and other == target_gid:
                continue
            worst = min(worst, float(mujoco.mj_geomDistance(model, data, gid, other, 0.30, fromto)))
    return worst


# --------------------------------------------------------------- 单场景
def scene_record(seed, elev, density, thicknesses):
    model = mujoco.MjModel.from_xml_string(xml_with_fingers(seed, elev, density))
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
    gids = object_geom_ids(model)
    R, t = camera_pose(data)
    pts, labels = cloud_from_depth(depth, seg, R, t, valid)

    grasps = []
    for i, gid in gids.items():
        mask = (seg == gid) & valid
        if mask.sum() < 60:
            continue
        g = grasp_for_object(mask, depth)
        if g:
            grasps.append(g)
    top = nms(grasps)
    rec = dict(seed=seed, elev=elev, density=density, n_objects=len(gids),
               n_cloud=len(pts), has_candidate=int(bool(top)))
    if not top:
        rec.update(pick="none")
        return rec

    _, cu, cv, deg, width_m, wpx, z0, p1, p2 = top[0]
    ok_adm, target_gid, why = admissible(seg, gids, p1, p2)
    rec.update(width_cm=round(width_m * 100, 2), score_rank=1, admiss=int(ok_adm),
               reject_reason=why)
    if not ok_adm:
        return rec

    z1 = float(depth[p1[1], p1[0]])
    z2 = float(depth[p2[1], p2[0]])
    c1 = pixel_to_world(p1[0], p1[1], z1, R, t)
    c2 = pixel_to_world(p2[0], p2[1], z2, R, t)
    rec.update(antipodality_deg=round(antipodality_deg(c1, c2), 1),
               opening_cm=round(float(np.linalg.norm(c2 - c1)) * 100, 2),
               # 旧判据里被判定"挡住"的那颗像素在世界里的位置，用来对照
               old_2d=int(old_criterion_2d(seg, gids, target_gid, p1, p2)))

    for th in thicknesses:
        g = dict(GRIPPER, finger_thickness=th)
        boxes = finger_boxes(c1, c2, g)
        key = key_fix(th)
        rec["A_" + key] = rec["old_2d"]
        rec["B_" + key] = int(new_criterion_cloud(pts, labels, boxes, target_gid, g["tol"]))
        rec["D_" + key] = int(new_criterion_exact(model, data, boxes, target_gid, g["tol"]))
        if abs(th - CANON) < 1e-9:
            rec["boxes_canon"] = boxes
    return rec


def key_fix(th):
    return "t%02d" % round(th * 1000)


# --------------------------------------------------------------- 汇总
def load_records():
    """从 criterion3d_scenes.csv 读回逐场景记录，用来只重算汇总和图（不重跑仿真）。"""
    path = os.path.join(OUT_DIR, "criterion3d_scenes.csv")
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            conv = {}
            for k, v in row.items():
                if v == "" or v is None:
                    continue
                try:
                    conv[k] = int(v)
                except ValueError:
                    try:
                        conv[k] = float(v)
                    except ValueError:
                        conv[k] = v
            out.append(conv)
    return out


def contingency(records, canon, name1, name2, label1, label2):
    """两个判据在这批场景上的列联表（只在通过前置条件的场景里比较）。"""
    cells = {"11": 0, "10": 0, "01": 0, "00": 0}
    for r in records:
        if not r.get("admiss"):
            continue
        a = int(bool(r.get(name1 + canon)))
        b = int(bool(r.get(name2 + canon)))
        cells["%d%d" % (a, b)] += 1
    lines = ["", "| %s \\ %s | 真实几何判能抓 | 真实几何判挡 |" % (label1, label2),
             "|---|---|---|",
             "| %s 判能抓 | %d | %d |" % (label1, cells["11"], cells["10"]),
             "| %s 判挡 | %d | %d |" % (label1, cells["01"], cells["00"])]
    return lines, cells


def summarize(records, thicknesses):
    """把逐场景记录压成 6 格 × 3 种判据的表。"""
    lines = []
    for elev, density in CELLS:
        sub = [r for r in records if r["elev"] == elev and r["density"] == density]
        n = len(sub)
        cand = sum(r["has_candidate"] for r in sub)
        adm = sum(r.get("admiss", 0) for r in sub)
        row = dict(elev=elev, density=density, n=n, candidate=cand, admissible=adm)
        for th in thicknesses:
            k = key_fix(th)
            row["A_" + k] = sum(r.get("A_" + k, 0) for r in sub)
            row["B_" + k] = sum(r.get("B_" + k, 0) for r in sub)
            row["D_" + k] = sum(r.get("D_" + k, 0) for r in sub)
        lines.append(row)
    return lines


def main(n_per_cell):
    t0 = time.time()
    records = []
    for elev, density in CELLS:
        for seed in range(n_per_cell):
            records.append(scene_record(seed, elev, density, THICKNESSES))
        got = [r for r in records if r["elev"] == elev and r["density"] == density]
        print("elev %2d %-6s  done %2d scenes  (%.1f s)" %
              (elev, density, len(got), time.time() - t0), flush=True)

    report = build_report(records, n_per_cell)
    print("\n".join(report))
    print("总耗时 %.1f s" % (time.time() - t0))


def build_report(records, n_per_cell):
    rows = summarize(records, THICKNESSES)
    canon = key_fix(CANON)
    report = []
    report.append("# 3D 接近判据的复测结果（判据 A/B/D，同一批种子 0–%d）" % (n_per_cell - 1))
    report.append("")
    report.append("场景走的是 bin_picking_demo.py 里同一套生成与渲染，种子、物体数、"
                  "仿真步数都没动。前置条件（两点同物体、闭合路径无遮挡）三种判据共用，"
                  "所以差别只来自「接近性」这一条。")
    report.append("")
    report.append("| 场景 | 俯角 | 选出候选 | 前置通过 | A 旧2D | B 新3D 单视角 | D 真实几何 |")
    report.append("|---|---|---|---|---|---|---|")
    name = {"sparse": "稀疏 6 物体", "dense": "堆叠 12 物体"}
    for row in rows:
        report.append("| %s | %d° | %d/%d | %d/%d | %d/%d | %d/%d | %d/%d |" % (
            name[row["density"]], row["elev"], row["candidate"], row["n"],
            row["admissible"], row["n"],
            row["A_" + canon], row["n"], row["B_" + canon], row["n"],
            row["D_" + canon], row["n"]))
    report.append("")
    b_bad = [r for r in records if r.get("admiss") and not r.get("B_" + canon)]
    d_bad = [r for r in records if r.get("admiss") and not r.get("D_" + canon)]
    a_bad = [r for r in records if r.get("admiss") and not r.get("A_" + canon)]
    n_adm = sum(r.get("admiss", 0) for r in records)
    report.append("指厚按 %.0f mm 算，前置条件通过的 %d 个场景里：" %
                  (CANON * 1000, n_adm))
    report.append("")
    report.append("- 旧判据 A 挡掉 %d 个，剩下 %d 个" % (len(a_bad), n_adm - len(a_bad)))
    report.append("- 新判据 B 挡掉 %d 个，剩下 %d 个" % (len(b_bad), n_adm - len(b_bad)))
    report.append("- 真实几何 D 挡掉 %d 个，剩下 %d 个" % (len(d_bad), n_adm - len(d_bad)))
    only_a = [r for r in records if r.get("admiss") and not r.get("A_" + canon)
              and r.get("D_" + canon)]
    only_b = [r for r in records if r.get("admiss") and r.get("B_" + canon)
              and not r.get("D_" + canon)]
    report.append("")
    report.append("旧判据判「挡」、真实几何判「能抓」的场景（旧判据的误杀）：%d 个" % len(only_a))
    report.append("单视角点云判「能抓」、真实几何判「挡」的场景（看不见的地方漏判）：%d 个" % len(only_b))
    report.append("")
    report.append("## 列联表（只算通过前置条件的 %d 个场景）" % n_adm)
    lines, cells = contingency(records, canon, "A_", "D_", "A 旧2D", "D 真实几何")
    report.extend(lines)
    lines, cells_b = contingency(records, canon, "B_", "D_", "B 新3D 单视角", "D 真实几何")
    report.extend(lines)
    report.append("")
    report.append("单视角点云相对真实几何的两种错法：多放行 %d 个（看不见的地方漏判），"
                  "多拦下 %d 个（点云离散与 2 mm 容差造成的假挡）。"
                  % (cells_b["10"], cells_b["01"]))
    report.append("")
    report.append("## 指厚敏感性")
    report.append("")
    report.append("| 指厚 | 旧2D A | 新3D B（单视角） | 真实几何 D |")
    report.append("|---|---|---|---|")
    for th in THICKNESSES:
        k = key_fix(th)
        report.append("| %.1f cm | %d/%d | %d/%d | %d/%d |" % (
            th * 100, sum(r.get("A_" + k, 0) for r in records), len(records),
            sum(r.get("B_" + k, 0) for r in records), len(records),
            sum(r.get("D_" + k, 0) for r in records), len(records)))
    any_t = [r for r in records if r.get("admiss")
             and all(r.get("D_" + key_fix(t), 0) for t in THICKNESSES)]
    none_t = [r for r in records if r.get("admiss")
              and not any(r.get("D_" + key_fix(t), 0) for t in THICKNESSES)]
    thin_only = [r for r in records if r.get("admiss")
                 and r.get("D_t05") and not r.get("D_t30")]
    report.append("")
    report.append("这张表的读法：前置通过的 %d 个场景里，%d 个在四种指厚下都过不去（挡它的是别的东西，"
                  "不是手指粗），%d 个在四种指厚下都能抓，剩下 %d 个只有 0.5 cm 的薄手指能抓、"
                  "改用 3 cm 的手指就过不去。" % (n_adm, len(none_t), len(any_t), len(thin_only)))
    anti = [r["antipodality_deg"] for r in records if "antipodality_deg" in r]
    if anti:
        report.append("")
        report.append("闭合方向偏离竖直的程度（0° = 闭合方向水平、夹爪竖直下压，越接近 0 越理想）："
                      "中位 %.1f°，最小 %.1f°，最大 %.1f°"
                      % (float(np.median(anti)), min(anti), max(anti)))
    report.append("")
    write_report(report, records, rows, n_adm, name)
    make_figure(records, rows, canon)
    return report


def write_report(report, records, rows, n_adm, name):
    path = os.path.join(OUT_DIR, "criterion3d_result.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    keys = sorted({k for r in records for k in r})
    with open(os.path.join(OUT_DIR, "criterion3d_scenes.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in records:
            w.writerow({k: v for k, v in r.items() if k != "boxes_canon"})
    print("\n已写 criterion3d_result.md、criterion3d_scenes.csv")


def make_figure(records, rows, canon):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    labels = ["%s\n%d°" % ("稀疏" if r["density"] == "sparse" else "堆叠", r["elev"])
              for r in rows]
    x = np.arange(len(rows))
    for offset, (key, col, lab) in enumerate([("A_", "#c0392b", "A 旧判据（2D）"),
                                              ("B_", "#2f5d8a", "B 新判据（单张深度图）"),
                                              ("D_", "#27ae60", "D 真实几何（参考）")]):
        vals = [r[key + canon] / r["n"] * 100 for r in rows]
        axes[0].bar(x + (offset - 1) * 0.27, vals, width=0.26, color=col, label=lab)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=8)
    axes[0].set_ylabel("接近性满足的场景 / %")
    axes[0].legend(fontsize=8)
    axes[0].set_title("指厚 1 cm：接近性判据的通过率", fontsize=10)
    axes[0].grid(axis="y", alpha=0.25)

    ths = [0.5, 1.0, 2.0, 3.0]
    for key, col, lab in [("B_", "#2f5d8a", "B 单张深度图"), ("D_", "#27ae60", "D 真实几何")]:
        vals = [sum(r.get(key + ("t%02d" % round(t * 10)), 0) for r in records)
                for t in ths]
        axes[1].plot(ths, vals, "o-", color=col, label=lab)
    axes[1].set_xlabel("手指厚度 / cm")
    axes[1].set_ylabel("满足接近性的场景数（共 %d）" % len(records))
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)
    axes[1].set_title("指厚敏感性", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "criterion3d_figure.png"), dpi=150)
    print("已写 criterion3d_figure.png")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--resummarize":
        recs = load_records()
        report = build_report(recs, max(1, len(recs) // len(CELLS)))
        print("\n".join(report))
        sys.exit(0)
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    main(n)
