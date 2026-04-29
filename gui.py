import tkinter as tk
from tkinter import ttk
import customtkinter as ctk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import glob
import os

from kalman import filter_path
from robot import PlanarArm

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ── Palette ──────────────────────────────────────
BG        = "#0d1117"   # near-black base
BG_PANEL  = "#161b22"   # slightly lighter panel
BG_CARD   = "#21262d"   # card/control surfaces
BORDER    = "#30363d"   # subtle borders
ACCENT    = "#58a6ff"   # blue accent
ACCENT2   = "#388bfd"   # darker accent for hover
RED       = "#f85149"
GREEN     = "#3fb950"
TEXT      = "#e6edf3"
TEXT_DIM  = "#8b949e"
CANVAS_BG = "#0d1117"


def _style_ax(ax, fig, title="", title_color=TEXT):
    """Apply consistent dark styling to a matplotlib axis."""
    ax.set_facecolor(CANVAS_BG)
    fig.patch.set_facecolor(BG_PANEL)
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.2, 2.2)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(True, alpha=0.08, color="#ffffff", linewidth=0.5)
    ax.set_title(title, color=title_color, fontsize=10,
                 fontfamily="monospace", pad=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
        spine.set_linewidth(0.8)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.02)


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Drawing Robot — Tremor Suppression")
        self.geometry("1300x780")
        self.minsize(900, 600)
        self.configure(fg_color=BG)

        # Header
        header = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=0, height=48)
        header.pack(fill=tk.X, side=tk.TOP)
        header.pack_propagate(False)
        ctk.CTkLabel(header, text="DRAWING ROBOT",
                     font=ctk.CTkFont(family="Courier New", size=15, weight="bold"),
                     text_color=ACCENT).pack(side=tk.LEFT, padx=20, pady=12)
        ctk.CTkLabel(header, text="Kalman Filter · Tremor Suppression · 2-DOF Arm",
                     font=ctk.CTkFont(size=11),
                     text_color=TEXT_DIM).pack(side=tk.LEFT, padx=0, pady=12)

        # Tabs
        self.tabs = ctk.CTkTabview(
            self, fg_color=BG_PANEL, corner_radius=8,
            segmented_button_fg_color=BG_CARD,
            segmented_button_selected_color=ACCENT,
            segmented_button_unselected_color=BG_CARD,
            segmented_button_selected_hover_color=ACCENT2,
            segmented_button_unselected_hover_color=BORDER,
            text_color=TEXT, text_color_disabled=TEXT_DIM,
        )
        self.tabs.pack(fill=tk.BOTH, expand=True, padx=10, pady=(6, 10))
        self.tabs.add("  Robot Demo  ")
        self.tabs.add("  Results  ")

        self._build_demo_tab(self.tabs.tab("  Robot Demo  "))
        self._build_results_tab(self.tabs.tab("  Results  "))

    # ─────────────────────────────────────────────
    #  Tab 1 — Robot Demo
    # ─────────────────────────────────────────────

    def _build_demo_tab(self, frame):
        frame.configure(fg_color=BG_PANEL)

        # ── Controls bar ──
        ctrl = ctk.CTkFrame(frame, fg_color=BG_CARD, corner_radius=8, height=52)
        ctrl.pack(fill=tk.X, padx=4, pady=(4, 8))
        ctrl.pack_propagate(False)

        self.tremor_freq = ctk.DoubleVar(value=60.0)
        self.tremor_amp  = ctk.DoubleVar(value=8.0)
        self.shaky_mode  = ctk.BooleanVar(value=True)

        def lbl(parent, text):
            return ctk.CTkLabel(parent, text=text, text_color=TEXT_DIM,
                                font=ctk.CTkFont(size=11))

        def val_lbl(parent, var, fmt=int):
            l = ctk.CTkLabel(parent, text=str(fmt(var.get())),
                             text_color=ACCENT, width=32,
                             font=ctk.CTkFont(family="Courier New", size=11))
            var.trace_add("write", lambda *_: l.configure(text=str(fmt(var.get()))))
            return l

        # Group 1: freq
        g1 = ctk.CTkFrame(ctrl, fg_color="transparent")
        g1.pack(side=tk.LEFT, padx=(16, 4), pady=10)
        lbl(g1, "FREQ (Hz)").pack(side=tk.LEFT, padx=(0, 6))
        ctk.CTkSlider(g1, from_=5, to=120, variable=self.tremor_freq,
                      width=130, button_color=ACCENT,
                      progress_color=ACCENT, fg_color=BORDER).pack(side=tk.LEFT)
        val_lbl(g1, self.tremor_freq).pack(side=tk.LEFT, padx=(6, 0))

        # Divider
        tk.Frame(ctrl, bg=BORDER, width=1).pack(side=tk.LEFT, fill=tk.Y, padx=12, pady=8)

        # Group 2: amplitude
        g2 = ctk.CTkFrame(ctrl, fg_color="transparent")
        g2.pack(side=tk.LEFT, padx=4, pady=10)
        lbl(g2, "AMP").pack(side=tk.LEFT, padx=(0, 6))
        ctk.CTkSlider(g2, from_=1, to=30, variable=self.tremor_amp,
                      width=130, button_color=ACCENT,
                      progress_color=ACCENT, fg_color=BORDER).pack(side=tk.LEFT)
        val_lbl(g2, self.tremor_amp).pack(side=tk.LEFT, padx=(6, 0))

        # Divider
        tk.Frame(ctrl, bg=BORDER, width=1).pack(side=tk.LEFT, fill=tk.Y, padx=12, pady=8)

        # Group 3: toggle
        ctk.CTkSwitch(ctrl, text="Shaky Mode", variable=self.shaky_mode,
                      onvalue=True, offvalue=False,
                      progress_color=ACCENT, text_color=TEXT_DIM,
                      font=ctk.CTkFont(size=11)).pack(side=tk.LEFT, padx=8)

        # Right: buttons
        ctk.CTkButton(ctrl, text="Run Robot ▶", width=130, height=34,
                      fg_color=ACCENT, hover_color=ACCENT2,
                      font=ctk.CTkFont(size=12, weight="bold"),
                      corner_radius=6,
                      command=self._run_robot).pack(side=tk.RIGHT, padx=(4, 16), pady=9)
        ctk.CTkButton(ctrl, text="Clear", width=80, height=34,
                      fg_color="transparent", hover_color=BG_CARD,
                      border_width=1, border_color=BORDER,
                      text_color=TEXT_DIM, corner_radius=6,
                      command=self._clear).pack(side=tk.RIGHT, padx=4, pady=9)

        # ── Main split ──
        pane = tk.PanedWindow(frame, orient=tk.HORIZONTAL,
                              sashwidth=4, sashrelief=tk.FLAT,
                              bg=BG, bd=0)
        pane.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 4))

        # Left: drawing canvas
        left = tk.Frame(pane, bg=BG_PANEL)
        pane.add(left, minsize=380)
        ctk.CTkLabel(left, text="✏  DRAW HERE",
                     text_color=TEXT_DIM,
                     font=ctk.CTkFont(family="Courier New", size=10)).pack(
                         anchor="w", padx=8, pady=(4, 2))
        canvas_wrap = tk.Frame(left, bg=BORDER, padx=1, pady=1)
        canvas_wrap.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 4))
        self.canvas = tk.Canvas(canvas_wrap, bg=CANVAS_BG,
                                cursor="crosshair", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<ButtonPress-1>",  self._on_press)
        self.canvas.bind("<B1-Motion>",      self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        # Right: arm
        right = tk.Frame(pane, bg=BG_PANEL)
        pane.add(right, minsize=380)
        ctk.CTkLabel(right, text="🤖  ROBOT ARM",
                     text_color=TEXT_DIM,
                     font=ctk.CTkFont(family="Courier New", size=10)).pack(
                         anchor="w", padx=8, pady=(4, 2))

        self.arm_fig = Figure(figsize=(5, 5), dpi=96)
        self.arm_ax  = self.arm_fig.add_subplot(111)
        _style_ax(self.arm_ax, self.arm_fig, "Waiting for input...", TEXT_DIM)

        arm_wrap = tk.Frame(right, bg=BORDER, padx=1, pady=1)
        arm_wrap.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 4))
        self.arm_canvas = FigureCanvasTkAgg(self.arm_fig, master=arm_wrap)
        self.arm_canvas.draw()
        self.arm_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Status
        self.status = ctk.StringVar(value="Draw something, then click Run Robot.")
        ctk.CTkLabel(frame, textvariable=self.status, text_color=TEXT_DIM,
                     font=ctk.CTkFont(family="Courier New", size=10),
                     anchor="w").pack(fill=tk.X, padx=10, pady=(2, 4))

        # State
        self.raw_points   = []
        self.shaky_points = []
        self.drawing      = False
        self.tremor_t     = 0.0
        self._anim_job    = None
        self._anim_frames = []
        self._anim_index  = 0
        self._arm         = PlanarArm(l1=1.0, l2=0.8)
        self._trace_x     = []
        self._trace_y     = []
        self._joint_angles = None

    def _apply_tremor(self, x, y):
        freq = self.tremor_freq.get()
        amp  = self.tremor_amp.get()
        self.tremor_t += 1.0 / 60.0
        tx = amp * np.sin(2 * np.pi * freq * self.tremor_t)
        ty = amp * np.cos(2 * np.pi * freq * self.tremor_t + np.pi / 4)
        return x + tx + np.random.normal(0, amp * 0.15), \
               y + ty + np.random.normal(0, amp * 0.15)

    def _on_press(self, event):
        self.drawing  = True
        self.tremor_t = 0.0
        x, y = event.x, event.y
        self.raw_points.append((x, y))
        sx, sy = self._apply_tremor(x, y) if self.shaky_mode.get() else (x, y)
        self.shaky_points.append((sx, sy))

    def _on_drag(self, event):
        if not self.drawing:
            return
        x, y = event.x, event.y
        self.raw_points.append((x, y))
        sx, sy = self._apply_tremor(x, y) if self.shaky_mode.get() else (x, y)
        self.shaky_points.append((sx, sy))
        if len(self.shaky_points) >= 2:
            x0, y0 = self.shaky_points[-2]
            self.canvas.create_line(x0, y0, sx, sy, fill=RED,
                                    width=2, capstyle=tk.ROUND)

    def _on_release(self, event):
        self.drawing = False
        self.status.set(f"{len(self.shaky_points)} points captured. Click Run Robot.")

    def _clear(self):
        if self._anim_job:
            self.after_cancel(self._anim_job)
            self._anim_job = None
        self.canvas.delete("all")
        self.raw_points.clear()
        self.shaky_points.clear()
        self.tremor_t = 0.0
        self._trace_x.clear()
        self._trace_y.clear()
        self.arm_ax.clear()
        _style_ax(self.arm_ax, self.arm_fig, "Waiting for input...", TEXT_DIM)
        self.arm_canvas.draw()
        self.status.set("Canvas cleared. Draw something.")

    def _run_robot(self):
        if len(self.shaky_points) < 10:
            self.status.set("Draw more before running!")
            return
        if self._anim_job:
            self.after_cancel(self._anim_job)
            self._anim_job = None

        self.status.set("Filtering…")
        self.update()

        pts = np.array(self.shaky_points, dtype=float)
        raw = np.array(self.raw_points,   dtype=float)
        cx  = (pts[:, 0].max() + pts[:, 0].min()) / 2
        cy  = (pts[:, 1].max() + pts[:, 1].min()) / 2
        sc  = max(max(pts[:, 0].max() - pts[:, 0].min(),
                      pts[:, 1].max() - pts[:, 1].min()) / 2, 1.0)

        def norm(p):
            return np.column_stack(((p[:, 0] - cx) / sc,
                                    -(p[:, 1] - cy) / sc))

        noisy_n    = norm(pts)
        raw_n      = norm(raw)
        filtered_n = filter_path(noisy_n, process_noise=1e-3, measurement_noise=0.5)

        # Blue overlay on canvas
        self.canvas.delete("filtered")
        fp = filtered_n.copy()
        fp[:, 0] = fp[:, 0] * sc + cx
        fp[:, 1] = -fp[:, 1] * sc + cy
        for i in range(1, len(fp)):
            self.canvas.create_line(fp[i-1][0], fp[i-1][1],
                                    fp[i][0],   fp[i][1],
                                    fill=ACCENT, width=2,
                                    capstyle=tk.ROUND, tags="filtered")

        ne = np.linalg.norm(noisy_n - raw_n, axis=1).mean()
        fe = np.linalg.norm(filtered_n - raw_n, axis=1).mean()
        rd = (1 - fe / ne) * 100 if ne > 0 else 0
        self.status.set(
            f"error  {ne:.4f} → {fe:.4f}   reduction {rd:.1f}%   "
            f"· red=shaky  ·  blue=filtered  ·  arm=robot"
        )

        joint_angles, _ = self._arm.follow_path(filtered_n)
        step = max(1, len(joint_angles) // 250)
        self._anim_frames  = list(range(0, len(joint_angles), step))
        self._anim_index   = 0
        self._joint_angles = joint_angles
        self._trace_x.clear()
        self._trace_y.clear()

        bg_raw = np.column_stack((raw_n[:, 0] * 0.5 + 0.9, raw_n[:, 1] * 0.5))
        bg_nsy = np.column_stack((noisy_n[:, 0] * 0.5 + 0.9, noisy_n[:, 1] * 0.5))

        ax = self.arm_ax
        ax.clear()
        _style_ax(ax, self.arm_fig, "robot drawing your filtered path")
        ax.plot(bg_raw[:, 0], bg_raw[:, 1], color=GREEN, lw=0.8,
                alpha=0.25, ls="--", label="intended")
        ax.plot(bg_nsy[:, 0], bg_nsy[:, 1], color=RED, lw=0.6,
                alpha=0.18, label="shaky")
        ax.legend(loc="upper left", fontsize=7, facecolor=BG_CARD,
                  labelcolor=TEXT_DIM, framealpha=0.8, edgecolor=BORDER)

        self._arm_line,   = ax.plot([], [], "o-", color=ACCENT,
                                    linewidth=3, markersize=9,
                                    markerfacecolor=BG, markeredgecolor=ACCENT,
                                    markeredgewidth=2)
        self._trace_line, = ax.plot([], [], "-", color=ACCENT,
                                    linewidth=1.5, alpha=0.5)
        self._tick_animation()

    def _tick_animation(self):
        if self._anim_index >= len(self._anim_frames):
            return
        i = self._anim_frames[self._anim_index]
        t1, t2 = self._joint_angles[i]
        base, elbow, tip = self._arm.forward_kinematics(t1, t2)
        self._arm_line.set_data([base[0], elbow[0], tip[0]],
                                [base[1], elbow[1], tip[1]])
        self._trace_x.append(tip[0])
        self._trace_y.append(tip[1])
        self._trace_line.set_data(self._trace_x, self._trace_y)
        self.arm_canvas.draw_idle()
        self._anim_index += 1
        self._anim_job = self.after(20, self._tick_animation)

    # ─────────────────────────────────────────────
    #  Tab 2 — Results Browser
    # ─────────────────────────────────────────────

    _SHAPE_PREFIXES = ("comparison_", "error_plot_")

    def _build_results_tab(self, frame):
        frame.configure(fg_color=BG_PANEL)

        # Nav bar
        nav = ctk.CTkFrame(frame, fg_color=BG_CARD, corner_radius=8, height=52)
        nav.pack(fill=tk.X, padx=4, pady=(4, 8))
        nav.pack_propagate(False)

        ctk.CTkButton(nav, text="◀  Prev", width=100, height=34,
                      fg_color="transparent", border_width=1,
                      border_color=BORDER, text_color=TEXT_DIM,
                      hover_color=BG_CARD, corner_radius=6,
                      command=self._prev_result).pack(side=tk.LEFT, padx=(14, 4), pady=9)
        ctk.CTkButton(nav, text="Next  ▶", width=100, height=34,
                      fg_color="transparent", border_width=1,
                      border_color=BORDER, text_color=TEXT_DIM,
                      hover_color=BG_CARD, corner_radius=6,
                      command=self._next_result).pack(side=tk.LEFT, padx=4, pady=9)

        self.result_label = ctk.StringVar(value="No results found.")
        ctk.CTkLabel(nav, textvariable=self.result_label, text_color=TEXT,
                     font=ctk.CTkFont(family="Courier New", size=11)
                     ).pack(side=tk.LEFT, padx=16)

        ctk.CTkButton(nav, text="↺  Refresh", width=100, height=34,
                      fg_color="transparent", border_width=1,
                      border_color=BORDER, text_color=TEXT_DIM,
                      hover_color=BG_CARD, corner_radius=6,
                      command=self._load_results).pack(side=tk.RIGHT, padx=14, pady=9)

        # Split pane
        pane = tk.PanedWindow(frame, orient=tk.HORIZONTAL,
                              sashwidth=4, sashrelief=tk.FLAT,
                              bg=BG, bd=0)
        pane.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 4))

        # Left: result image
        left = tk.Frame(pane, bg=BG_PANEL)
        pane.add(left, minsize=340)
        img_wrap = tk.Frame(left, bg=BORDER, padx=1, pady=1)
        img_wrap.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)
        self.result_fig = Figure(facecolor=BG_PANEL)
        self.result_ax  = self.result_fig.add_subplot(111)
        self.result_ax.set_facecolor(BG_PANEL)
        self.result_ax.axis("off")
        self.result_fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.result_canvas = FigureCanvasTkAgg(self.result_fig, master=img_wrap)
        self.result_canvas.draw()
        self.result_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.result_canvas.get_tk_widget().configure(bg=BG_PANEL, highlightthickness=0)

        # Right: shape animation
        right = tk.Frame(pane, bg=BG_PANEL)
        pane.add(right, minsize=340)
        ctk.CTkLabel(right, text="🤖  SHAPE ANIMATION",
                     text_color=TEXT_DIM,
                     font=ctk.CTkFont(family="Courier New", size=10)
                     ).pack(anchor="w", padx=8, pady=(4, 2))
        anim_wrap = tk.Frame(right, bg=BORDER, padx=1, pady=1)
        anim_wrap.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 4))

        self.res_anim_fig = Figure(figsize=(5, 5), dpi=96)
        self.res_anim_ax  = self.res_anim_fig.add_subplot(111)
        _style_ax(self.res_anim_ax, self.res_anim_fig,
                  "navigate to a shape result", TEXT_DIM)

        self.res_anim_canvas = FigureCanvasTkAgg(self.res_anim_fig, master=anim_wrap)
        self.res_anim_canvas.draw()
        self.res_anim_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # State
        self._res_anim_job     = None
        self._res_anim_frames  = []
        self._res_anim_index   = 0
        self._res_trace_x      = []
        self._res_trace_y      = []
        self._res_arm          = PlanarArm(l1=1.0, l2=0.8)
        self._res_joint_angles = None
        self._result_files     = []
        self._result_index     = 0
        self._load_results()

    def _load_results(self):
        base = os.path.dirname(os.path.abspath(__file__))
        self._result_files = sorted(
            glob.glob(os.path.join(base, "results", "*.png")))
        self._result_index = 0
        if self._result_files:
            self._show_result(0)
        else:
            self.result_label.set("No results — run experiments.py first.")

    def _shape_from_filename(self, filename):
        name = os.path.basename(filename)
        for prefix in self._SHAPE_PREFIXES:
            if name.startswith(prefix):
                shape = name.replace(prefix, "").replace(".png", "")
                if shape in ("circle", "figure8", "spiral"):
                    return shape
        return None

    def _show_result(self, idx):
        if not self._result_files:
            return
        if self._res_anim_job:
            self.after_cancel(self._res_anim_job)
            self._res_anim_job = None

        path  = self._result_files[idx]
        name  = os.path.basename(path)
        shape = self._shape_from_filename(path)

        img = plt.imread(path)
        self.result_ax.clear()
        self.result_ax.set_facecolor(BG_PANEL)
        self.result_ax.imshow(img)
        self.result_ax.axis("off")
        self.result_fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
        self.result_canvas.draw()
        self.result_label.set(f"{name}   {idx + 1} / {len(self._result_files)}")

        if shape:
            self._start_res_animation(shape)
        else:
            self.res_anim_ax.clear()
            _style_ax(self.res_anim_ax, self.res_anim_fig,
                      "math result — no animation", TEXT_DIM)
            self.res_anim_canvas.draw()

    def _start_res_animation(self, shape):
        from tremor import generate_path, add_tremor
        true_path     = generate_path(shape=shape, n_points=300)
        noisy_path    = add_tremor(true_path, frequency=60.0, amplitude=0.055)
        filtered_path = filter_path(noisy_path, process_noise=1e-3,
                                    measurement_noise=0.5)
        joint_angles, _ = self._res_arm.follow_path(filtered_path)
        step = max(1, len(joint_angles) // 250)
        self._res_anim_frames  = list(range(0, len(joint_angles), step))
        self._res_anim_index   = 0
        self._res_joint_angles = joint_angles
        self._res_trace_x.clear()
        self._res_trace_y.clear()

        ax = self.res_anim_ax
        ax.clear()
        _style_ax(ax, self.res_anim_fig, f"robot arm — {shape}")
        scaled = np.column_stack((true_path[:, 0] * 0.5 + 0.9,
                                  true_path[:, 1] * 0.5))
        ax.plot(scaled[:, 0], scaled[:, 1], color=GREEN, lw=1,
                alpha=0.3, ls="--", label="intended path")
        ax.legend(loc="upper left", fontsize=7, facecolor=BG_CARD,
                  labelcolor=TEXT_DIM, framealpha=0.8, edgecolor=BORDER)

        self._res_arm_line,   = ax.plot([], [], "o-", color=ACCENT,
                                        linewidth=3, markersize=9,
                                        markerfacecolor=BG,
                                        markeredgecolor=ACCENT,
                                        markeredgewidth=2)
        self._res_trace_line, = ax.plot([], [], "-", color=ACCENT,
                                        linewidth=1.5, alpha=0.5)
        self._tick_res_animation()

    def _tick_res_animation(self):
        if self._res_anim_index >= len(self._res_anim_frames):
            return
        i = self._res_anim_frames[self._res_anim_index]
        t1, t2 = self._res_joint_angles[i]
        base, elbow, tip = self._res_arm.forward_kinematics(t1, t2)
        self._res_arm_line.set_data([base[0], elbow[0], tip[0]],
                                    [base[1], elbow[1], tip[1]])
        self._res_trace_x.append(tip[0])
        self._res_trace_y.append(tip[1])
        self._res_trace_line.set_data(self._res_trace_x, self._res_trace_y)
        self.res_anim_canvas.draw_idle()
        self._res_anim_index += 1
        self._res_anim_job = self.after(20, self._tick_res_animation)

    def _prev_result(self):
        if not self._result_files:
            return
        self._result_index = (self._result_index - 1) % len(self._result_files)
        self._show_result(self._result_index)

    def _next_result(self):
        if not self._result_files:
            return
        self._result_index = (self._result_index + 1) % len(self._result_files)
        self._show_result(self._result_index)


if __name__ == "__main__":
    app = App()
    app.mainloop()
