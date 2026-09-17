# -*- coding: utf-8 -*-
"""生成抓取 demo 的图标（多尺寸 ICO）—— 简化版：一眼能看懂 = 夹爪抓箱中物体。"""
import os
from PIL import Image, ImageDraw

S = 512
img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
d = ImageDraw.Draw(img)
# 深蓝圆角底
d.rounded_rectangle([8, 8, S - 8, S - 8], radius=100, fill=(18, 26, 43, 255))
# 料箱轮廓（粗线，浅灰）
d.polygon([(80, 176), (432, 176), (384, 440), (128, 440)], outline=(226, 232, 240, 255), width=36)
# 目标物体（一个蓝色方块）
d.rectangle([196, 236, 316, 356], fill=(96, 165, 250, 255))
# 夹爪：一根粗绿横线 + 两端大圆（这是唯一焦点）
d.line([132, 296, 380, 296], fill=(52, 226, 130, 255), width=34)
for cx in (132, 380):
    d.ellipse([cx - 33, 263, cx + 33, 329], fill=(52, 226, 130, 255))

out = r"D:\Hermes\抓取demo\.assets"
os.makedirs(out, exist_ok=True)
ico = os.path.join(out, "grasp_demo.ico")
img.save(os.path.join(out, "grasp_demo_512.png"))
img.save(ico, sizes=[(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)])
# 小尺寸预览（32px 放大看）
prev = Image.open(ico)
prev.size = (32, 32)
prev.resize((256, 256), Image.NEAREST).save(os.path.join(out, "preview_32.png"))
print("ok", os.path.getsize(ico))
