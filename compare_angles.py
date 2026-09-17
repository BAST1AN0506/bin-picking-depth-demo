# -*- coding: utf-8 -*-
"""
compare_angles.py —— 对比实验：相机俯角 × 场景密集度，对抓取成功率的影响
=====================================================================
两个变量：
  · 相机俯角：30°（很斜）/ 45° / 60°（接近正上方）
  · 场景密集度：sparse（6 个物体、摊开）/ dense（12 个物体、堆叠）

每种组合跑 N 个随机场景，输出对比表 + 每个组合一张样例图。

跑法：
  python compare_angles.py            # 每种组合 30 个场景（约 20 秒）
  python compare_angles.py 50

产出：
  compare_result.md / .txt            对比表（Markdown，可直接贴进报告）
  compare_<角度>deg_<密集度>.png       各组合样例（4 联图）
"""

import os
import sys, time
from bin_picking_demo import run_scene

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

ANGLES = [30, 45, 60]
DENSITIES = ["sparse", "dense"]
CN = {"sparse": "稀疏(6个)", "dense": "堆叠(12个)"}


def main(n):
    rows = []
    for dens in DENSITIES:
        for a in ANGLES:
            t0 = time.time()
            picked = valid = 0
            for i in range(n):
                has, ok, _ = run_scene(seed=i, want_image=(i == 0), elev_deg=a, density=dens,
                                       img_path=os.path.join(OUT_DIR, f"compare_{a}deg_{dens}.png"))
                picked += has
                valid += ok
            rows.append((dens, a, picked, valid))
            print(f"{CN[dens]:<10} 俯角 {a:>2}° ：选出 {picked}/{n}，合格 {valid}/{n}，{time.time()-t0:.1f}s")

    lines = [f"# 相机俯角 × 场景密集度 对比实验（每格 {n} 个随机场景）", "",
             "| 场景 | 相机俯角 | 选出抓取 | 抓取合格 | 选出率 | 合格率 |",
             "|---|---|---|---|---|---|"]
    for dens, a, p, v in rows:
        lines.append(f"| {CN[dens]} | {a}° | {p}/{n} | {v}/{n} | {p/n*100:.0f}% | {v/n*100:.0f}% |")
    lines += ["", "判定标准：两个接触点在同一物体上 + 夹爪闭合路径无遮挡 + 上方接近通道无遮挡。"]
    txt = "\n".join(lines)
    print("\n" + txt)
    open(os.path.join(OUT_DIR, "compare_result.md"), "w", encoding="utf-8").write(txt + "\n")
    open(os.path.join(OUT_DIR, "compare_result.txt"), "w", encoding="utf-8").write(txt + "\n")
    print("\n已保存 compare_result.md / compare_result.txt / compare_*deg_*.png")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 30)
