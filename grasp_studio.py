# -*- coding: utf-8 -*-
"""
抓取仿真台 (Grasp Studio) —— 一个能转旋钮、立刻看结果的抓取仿真小程序
====================================================================
左边调参数，右边看画面，下面看统计。全部基于 MuJoCo + 深度图抓取点选择。

启动：
  双击 run_studio.bat         （推荐）
  或命令行：.venv\\Scripts\\pythonw.exe grasp_studio.py

自检（不开窗，验证能启动）：
  .venv\\Scripts\\python.exe grasp_studio.py --selftest
"""

import os, sys, time, random, queue, threading
import tkinter as tk
from tkinter import ttk

import bin_picking_demo as B
from PIL import Image, ImageTk

HERE = os.path.dirname(os.path.abspath(__file__))
TMP_IMG = os.path.join(HERE, "_studio_latest.png")


class Studio:
    def __init__(self, root):
        self.root = root
        root.title("抓取仿真台 Grasp Studio")
        root.geometry("1060x780")
        root.configure(bg="#121a2b")
        self.q = queue.Queue()
        self.running = False
        self.last_img = None
        self._build()
        self.root.after(120, self._poll)

    # ---------------------------------------------------------------- 界面
    def _build(self):
        pad = dict(padx=8, pady=5)
        left = tk.Frame(self.root, bg="#121a2b")
        left.pack(side="left", fill="y", **pad)
        right = tk.Frame(self.root, bg="#121a2b")
        right.pack(side="right", fill="both", expand=True, **pad)

        def lab(t, fg="#c9d8ef", size=10, bold=False):
            return tk.Label(left, text=t, bg="#121a2b", fg=fg,
                            font=("Microsoft YaHei UI", size, "bold" if bold else "normal"),
                            anchor="w", justify="left")

        lab("抓取仿真台", fg="#ffd766", size=14, bold=True).pack(fill="x", pady=(0, 8))

        self.v_ang = tk.IntVar(value=45)
        self.v_num = tk.IntVar(value=8)
        self.v_scn = tk.IntVar(value=30)
        self.v_gmin = tk.DoubleVar(value=2.5)
        self.v_gmax = tk.DoubleVar(value=8.5)
        self.v_reach = tk.BooleanVar(value=False)

        def slider(title, var, lo, hi, unit="", res=1):
            lab(title)
            f = tk.Frame(left, bg="#121a2b"); f.pack(fill="x")
            s = tk.Scale(f, from_=lo, to=hi, orient="horizontal", variable=var,
                         resolution=res, bg="#121a2b", fg="#e8eefc", troughcolor="#25304a",
                         highlightthickness=0, showvalue=True, length=190)
            s.pack(side="left")
            tk.Label(f, text=unit, bg="#121a2b", fg="#8fa6c8",
                     font=("Microsoft YaHei UI", 9)).pack(side="left")

        slider("相机俯角（度）", self.v_ang, 15, 85, "越小越斜")
        slider("箱内物体数量", self.v_num, 3, 20, "越多越难")
        slider("每个参数跑多少场景", self.v_scn, 5, 100)
        slider("夹爪最小张开 (cm)", self.v_gmin, 1.0, 6.0, res=0.5)
        slider("夹爪最大张开 (cm)", self.v_gmax, 4.0, 12.0, res=0.5)

        tk.Checkbutton(left, text="要求“能从上方下手”(reach)", variable=self.v_reach,
                       bg="#121a2b", fg="#c9d8ef", selectcolor="#25304a",
                       activebackground="#121a2b", activeforeground="#fff",
                       font=("Microsoft YaHei UI", 10), anchor="w").pack(fill="x", pady=4)

        bf = tk.Frame(left, bg="#121a2b"); bf.pack(fill="x", pady=8)
        for txt, cmd, col in [("跑单场", self.run_one, "#2f6fed"),
                              ("跑一批", self.run_batch, "#1e8449"),
                              ("保存当前图", self.save_img, "#5a4fcf")]:
            tk.Button(bf, text=txt, command=cmd, bg=col, fg="white", relief="flat",
                      font=("Microsoft YaHei UI", 10, "bold"), width=11,
                      activebackground=col).pack(side="left", padx=3, pady=2)

        lab("— 统计 —", fg="#8fa6c8", size=10).pack(fill="x", pady=(10, 2))
        self.lbl_stat = tk.Label(left, text="还没跑过", bg="#121a2b", fg="#7ce8a8",
                                 font=("Consolas", 11), anchor="w", justify="left")
        self.lbl_stat.pack(fill="x")

        self.canvas = tk.Label(right, bg="#0c111c")
        self.canvas.pack(fill="both", expand=True)
        self.log = tk.Text(self.root, height=7, bg="#0c111c", fg="#bcd0ea",
                           font=("Consolas", 9), relief="flat")
        self.log.pack(side="bottom", fill="x")

    def logw(self, s):
        self.q.put(("log", s))

    # ---------------------------------------------------------------- 动作
    def _apply(self):
        B.GRIP_MIN = self.v_gmin.get() / 100.0
        B.GRIP_MAX = self.v_gmax.get() / 100.0
        B.GRIP_IDEAL = (B.GRIP_MIN + B.GRIP_MAX) / 2

    def run_one(self):
        if self.running:
            return
        self._apply()
        ang, num, reach = self.v_ang.get(), self.v_num.get(), self.v_reach.get()
        self.logw(f"→ 跑单场：俯角 {ang}°，物体 {num} 个，夹爪 {self.v_gmin.get():.1f}~{self.v_gmax.get():.1f} cm")
        threading.Thread(target=self._work_one, args=(ang, num, reach), daemon=True).start()

    def _work_one(self, ang, num, reach):
        self.running = True
        try:
            t0 = time.time()
            has, ok, k = B.run_scene(random.randint(0, 9999), want_image=True, elev_deg=ang,
                                     n_obj=num, reach=reach, img_path=TMP_IMG)
            self.q.put(("img", TMP_IMG))
            self.q.put(("stat", f"单场：选出={'是' if has else '否'}  合格={'是' if ok else '否'}\n"
                                f"候选抓取 {k} 个   用时 {time.time()-t0:.2f}s"))
        except Exception as e:
            self.q.put(("log", f"出错：{e}"))
        self.running = False

    def run_batch(self):
        if self.running:
            return
        self._apply()
        ang, num, n, reach = self.v_ang.get(), self.v_num.get(), self.v_scn.get(), self.v_reach.get()
        self.logw(f"→ 跑一批：{n} 个场景，俯角 {ang}°，物体 {num} 个")
        threading.Thread(target=self._work_batch, args=(ang, num, n, reach), daemon=True).start()

    def _work_batch(self, ang, num, n, reach):
        self.running = True
        try:
            t0 = time.time()
            picked = valid = 0
            for i in range(n):
                has, ok, _ = B.run_scene(i, want_image=(i == 0), elev_deg=ang, n_obj=num,
                                         reach=reach, img_path=TMP_IMG)
                picked += has; valid += ok
                if i % 5 == 0:
                    self.logw(f"   进行中 {i}/{n} …")
            self.q.put(("img", TMP_IMG))
            self.q.put(("stat", f"批量：{n} 个场景，俯角 {ang}°，物体 {num} 个\n"
                                f"选出抓取 {picked}/{n} = {picked/n*100:.0f}%\n"
                                f"抓取合格 {valid}/{n} = {valid/n*100:.0f}%\n"
                                f"总用时 {time.time()-t0:.1f}s"))
        except Exception as e:
            self.q.put(("log", f"出错：{e}"))
        self.running = False

    def save_img(self):
        if not os.path.exists(TMP_IMG):
            self.logw("还没有图可存")
            return
        out = os.path.join(HERE, f"studio_{int(time.time())}.png")
        Image.open(TMP_IMG).save(out)
        self.logw(f"已保存：{out}")

    # ---------------------------------------------------------------- 循环
    def _poll(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "log":
                    self.log.insert("end", payload + "\n"); self.log.see("end")
                elif kind == "stat":
                    self.lbl_stat.config(text=payload)
                elif kind == "img":
                    self._show(payload)
        except queue.Empty:
            pass
        self.root.after(120, self._poll)

    def _show(self, path):
        try:
            im = Image.open(path)
        except Exception:
            return
        w = max(self.canvas.winfo_width(), 400)
        h = max(self.canvas.winfo_height(), 300)
        im.thumbnail((w, h))
        self.last_img = ImageTk.PhotoImage(im)
        self.canvas.configure(image=self.last_img)


def selftest():
    root = tk.Tk(); root.withdraw()
    st = Studio(root)
    st._apply()
    t0 = time.time()
    has, ok, k = B.run_scene(0, want_image=True, elev_deg=45, n_obj=8, img_path=TMP_IMG)
    print(f"SELFTEST ok: has={has} ok={ok} cand={k} img_exists={os.path.exists(TMP_IMG)} "
          f"t={time.time()-t0:.2f}s widgets_built={bool(st.canvas)}")
    root.destroy()


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        r = tk.Tk()
        Studio(r)
        r.mainloop()
