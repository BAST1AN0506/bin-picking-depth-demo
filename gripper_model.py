# -*- coding: utf-8 -*-
"""
gripper_model.py —— 夹爪模型（把"手指有多厚"变成一个可以调的参数）
==================================================================
旧判据用"接触点正上方 40 px 的图像通道"近似夹爪下行空间，问题在
criterion_failure_evidence.md 里已经写明：那条通道在世界坐标里是斜的。

这份文件给出替代物：一个两指平行夹爪的几何模型。
抓取面上方那一段空间拆成两截，因为它们的物理约束不一样：

  指尖段 tip   抓取面往上 0~5 mm    贴住目标本身是正常的
  指体段 body  抓取面往上 5 mm 起   这一段里出现任何物体（包括目标自己
                                   鼓出来的部分）都会挡住夹爪

手指沿闭合方向从接触面向外只有 thickness 那么厚，横向宽度 finger_width，
从抓取面上方 finger_length 开始算指体，standoff 是夹爪开始下压的高度。
默认值取的是常见电动两指夹爪的量级（指厚 1 cm），脚本里会扫 0.5/1/2/3 cm
看结论对指厚有多敏感——因为指厚本来就是选夹爪时才定的，不该遮在结论里。

坐标：全部用世界坐标（米），approach_up 默认世界竖直 +Z。
"""

import numpy as np

GRIPPER = dict(
    finger_thickness=0.010,   # 沿闭合方向，从接触面向外的厚度
    finger_width=0.020,       # 与闭合方向、接近方向都垂直的宽度
    finger_length=0.040,      # 抓取面上方指体的长度
    standoff=0.060,           # 从抓取面上方多高处开始下压
    tip_height=0.005,         # 指尖段高度（这段允许贴住目标）
    tol=0.002,                # 2 mm 容差：反投影误差上限，见证据文件
)


def _unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def finger_boxes(c1, c2, g=GRIPPER, approach_up=(0.0, 0.0, 1.0)):
    """返回四个盒子（两个指尖段 + 两个指体段）。

    每个盒子 = (kind, center, axes, half)
      kind  'tip' 或 'body'
      axes  3x3，列向量 = 盒子的局部 x/y/z 轴在世界坐标下的方向
      half  三个半边长
    """
    c1 = np.asarray(c1, float)
    c2 = np.asarray(c2, float)
    up = _unit(approach_up)
    close = c2 - c1
    if np.linalg.norm(close) < 1e-9:
        return []
    # 闭合方向里去掉接近方向的分量，剩下的才是手指真正张合的方向
    close_orth = close - np.dot(close, up) * up
    close_orth = _unit(close_orth)

    t, w = g["finger_thickness"], g["finger_width"]
    tip_h, body_h = g["tip_height"], g["finger_length"] + g["standoff"]

    boxes = []
    for contact, outward in ((c1, -close_orth), (c2, +close_orth)):
        # 每个手指各自建正交右手系：x = 向外, y = up×x, z = up，保证 det = +1
        side = _unit(np.cross(up, outward))
        axes = np.column_stack([outward, side, up])
        center_tip = contact + outward * (t / 2) + up * (tip_h / 2)
        center_body = contact + outward * (t / 2) + up * (tip_h + body_h / 2)
        boxes.append(("tip", center_tip, axes, np.array([t / 2, w / 2, tip_h / 2])))
        boxes.append(("body", center_body, axes, np.array([t / 2, w / 2, body_h / 2])))
    return boxes


def points_inside(points, box, tol=0.0):
    """点是否落在盒子里（tol 收缩盒子，用来吸收反投影误差）。返回布尔数组。"""
    _, center, axes, half = box
    local = (points - center) @ axes                        # 投影到局部坐标
    return np.all(np.abs(local) <= (half - tol), axis=1)


def antipodality_deg(c1, c2, approach_up=(0.0, 0.0, 1.0)):
    """闭合方向偏离竖直的程度：0° = 闭合方向水平（夹爪竖直下压，两指平行），值越小越理想。"""
    up = _unit(approach_up)
    v = _unit(np.asarray(c2, float) - np.asarray(c1, float))
    return float(np.degrees(np.arcsin(np.clip(abs(np.dot(v, up)), 0.0, 1.0))))
