"""
Stewart Platform Analyzer V12 (ipywidgets)

Interactive Stewart Platform visualization using ipywidgets and matplotlib.

Usage:
    Run in a Jupyter-compatible environment (JupyterLab, VS Code interactive, etc.):
        %matplotlib widget
        from stewart_widget import StewartPlatformWidget
        app = StewartPlatformWidget()
        app.show()
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
import ipywidgets as widgets
from IPython.display import display
import io


class StewartPlatformWidget:
    def __init__(self):
        # --- 1. SYSTEM PARAMETERS ---
        self.B = np.array([
            [ 501, -1597, -1482],
            [1132, -1232, -1482],
            [1132,  1232, -1482],
            [ 501,  1597, -1482],
            [-1633,  365, -1482],
            [-1633, -365, -1482]
        ], dtype=float)

        self.P_local = np.array([
            [-400, -1043, 0],
            [1104,  -175, 0],
            [1104,   175, 0],
            [-400,  1043, 0],
            [-703,   868, 0],
            [-703,  -868, 0]
        ], dtype=float)

        self.L_mid = 1821.0
        self.Stroke_Base = 158.0
        self.Swivel_Max = 17.0

        self.Leg_Vec_Neutral = np.zeros((6, 3))
        for i in range(6):
            v = self.P_local[i] - self.B[i]
            self.Leg_Vec_Neutral[i] = v / np.linalg.norm(v)

        # Shared data
        self.raw_data_xyz = None
        self.loaded_data = None
        self.time_vector = None
        self.is_playing = False
        self.current_stroke_limit = self.Stroke_Base
        self.current_amp = 100.0
        self.workspace_full = None
        self.workspace_safe = None
        self.surf_workspace = None

        self._build_figure()
        self._init_graphics()
        self._build_controls()
        self._build_layout()

        self.update_platform([0, 0, 0])
        self._apply_view('Default')
        self._apply_zoom(False)

    # ------------------------------------------------------------------ #
    #  FIGURE
    # ------------------------------------------------------------------ #
    def _build_figure(self):
        self.fig = plt.figure(figsize=(9, 8))
        self.ax_3d = self.fig.add_axes([0.05, 0.27, 0.9, 0.70], projection='3d')
        self.ax_3d.set_xlabel('X'); self.ax_3d.set_ylabel('Y'); self.ax_3d.set_zlabel('Z')
        self.ax_time = self.fig.add_axes([0.1, 0.05, 0.85, 0.15])
        self.ax_time.set_xlabel('Time (sec)'); self.ax_time.set_ylabel('Disp (mm)')
        self.ax_time.grid(True)

    def _init_graphics(self):
        bx, by, bz = self.B[:,0], self.B[:,1], self.B[:,2]
        self.scat_base = self.ax_3d.scatter(bx, by, bz, c='k', s=50, depthshade=True)

        base_radius = np.mean(np.sqrt(bx**2 + by**2))
        th = np.linspace(0, 2*np.pi, 100)
        self.line_base, = self.ax_3d.plot(base_radius*np.cos(th), base_radius*np.sin(th),
                                          np.full_like(th, bz[0]), 'k--')

        self.text_leg_nums = []
        for i in range(6):
            t = self.ax_3d.text(bx[i], by[i], bz[i], f" {i+1}", color='black',
                                fontsize=10, fontweight='bold')
            self.text_leg_nums.append(t)

        self.lines_legs = [self.ax_3d.plot([],[],[], color='gray', linewidth=3)[0] for _ in range(6)]
        self.scat_plat = self.ax_3d.scatter([], [], [], c='b', s=60, depthshade=True)
        self.line_plat, = self.ax_3d.plot([], [], [], 'b-', linewidth=1.5)
        self.scat_center = self.ax_3d.scatter([0], [0], [0], c='b', s=30, edgecolors='none', zorder=100)
        self.line_path, = self.ax_3d.plot([], [], [], color='m', alpha=0.3, linewidth=1)
        self.scat_fail = self.ax_3d.scatter([], [], [], c='r', marker='.', s=1, zorder=50)

        self.line_tx, = self.ax_time.plot([], [], 'b-', label='X')
        self.line_ty, = self.ax_time.plot([], [], 'r-', label='Y')
        self.line_tz, = self.ax_time.plot([], [], 'g-', label='Z')
        self.line_cursor = self.ax_time.axvline(0, color='k', linewidth=2)
        self.ax_time.legend(loc='upper right', ncol=3, fontsize='small')

    # ------------------------------------------------------------------ #
    #  CONTROLS
    # ------------------------------------------------------------------ #
    def _build_controls(self):
        style = {'description_width': '70px'}
        slider_layout = widgets.Layout(width='100%')

        # --- View & Config ---
        self.dd_view = widgets.Dropdown(
            options=['Default (20, -75)', 'Front View (X-Z)', 'Side View (Y-Z)', 'Top View (X-Y)'],
            value='Default (20, -75)', description='Camera:', style=style, layout=slider_layout)
        self.dd_view.observe(self._on_view_change, names='value')

        self.chk_safe = widgets.Checkbox(value=False, description='Limit Stroke to 80%', indent=False)
        self.chk_safe.observe(self._on_safety_change, names='value')

        self.chk_zoom = widgets.Checkbox(value=False, description='Zoom to Center (±300 mm)', indent=False)
        self.chk_zoom.observe(self._on_zoom_change, names='value')

        self.chk_workspace = widgets.Checkbox(value=False, description='Show Workspace Surface', indent=False)
        self.chk_workspace.observe(self._on_workspace_change, names='value')

        self.chk_track = widgets.Checkbox(value=True, description='Show Trajectory & Failures', indent=False)
        self.chk_track.observe(self._on_track_change, names='value')

        # --- Stroke readouts (single HTML block) ---
        self.lbl_strokes = widgets.HTML(value=self._build_strokes_html([0.0]*6, [False]*6))

        # --- Position label ---
        self.lbl_pos = widgets.HTML(value='<b>X:</b> 0.0 | <b>Y:</b> 0.0 | <b>Z:</b> 0.0')

        # --- Manual sliders ---
        self.sld_x = widgets.FloatSlider(value=0, min=-300, max=300, step=1,
                                         description='X:', style=style, layout=slider_layout,
                                         continuous_update=True)
        self.sld_y = widgets.FloatSlider(value=0, min=-300, max=300, step=1,
                                         description='Y:', style=style, layout=slider_layout,
                                         continuous_update=True)
        self.sld_z = widgets.FloatSlider(value=0, min=-300, max=300, step=1,
                                         description='Z:', style=style, layout=slider_layout,
                                         continuous_update=True)
        self.sld_x.observe(self._on_manual, names='value')
        self.sld_y.observe(self._on_manual, names='value')
        self.sld_z.observe(self._on_manual, names='value')

        # --- File upload ---
        self.file_upload = widgets.FileUpload(accept='.csv,.txt', multiple=False,
                                              description='Load Data')
        self.file_upload.observe(self._on_file_upload, names='value')

        self.file_upload.layout = widgets.Layout(width='100%', overflow='visible')

        # --- Amplitude ---
        self.sld_amp = widgets.IntSlider(value=100, min=0, max=200, step=10,
                                         description='Amp %:', style=style,
                                         layout=slider_layout, disabled=True)
        self.sld_amp.observe(self._on_amp_change, names='value')

        # --- Play widget (replaces custom threading animation) ---
        self.player = widgets.Play(
            value=0, min=0, max=1, step=1,
            interval=33,  # ~30 fps
            disabled=True,
            show_repeat=True,
            layout=widgets.Layout(min_width='180px', overflow='visible')
        )
        self.player.observe(self._on_player_change, names='value')

        self.sld_progress = widgets.IntSlider(value=0, min=0, max=1, step=1,
                                              description='Frame:', style=style,
                                              layout=slider_layout, disabled=True,
                                              continuous_update=False)
        # Link Play widget to progress slider (bidirectional, client-side)
        widgets.jslink((self.player, 'value'), (self.sld_progress, 'value'))
        self.sld_progress.observe(self._on_progress_change, names='value')

        self.lbl_progress = widgets.HTML(value='T=0.00s (0/0)')

        # --- Status ---
        self.lbl_status = widgets.HTML(value='<i>Initializing...</i>')

        # --- Precompute button ---
        self.btn_precompute = widgets.Button(description='Precompute Workspaces',
                                             layout=widgets.Layout(width='100%'))
        self.btn_precompute.on_click(self._on_precompute)

        self.btn_precompute.layout = widgets.Layout(width='100%', overflow='visible')

    @staticmethod
    def _build_strokes_html(strokes, over_limits):
        lines = []
        for i, (val, over) in enumerate(zip(strokes, over_limits)):
            color = 'red' if over else 'black'
            lines.append(f"<span style='color:{color}'>Leg {i+1}: {val:+.1f} mm</span>")
        return "<pre style='margin:0;font-size:13px;line-height:1.6;overflow:visible;max-height:none'>" + "\n".join(lines) + "</pre>"

    def _build_layout(self):
        section = lambda title: widgets.HTML(f"<b style='font-size:13px'>{title}</b>")

        strokes_box = widgets.VBox([self.lbl_strokes],
                                   layout=widgets.Layout(border='1px solid #ddd', padding='4px',
                                                         overflow='visible', height='auto',
                                                         max_height='none'))

        # Player row: center the Play buttons
        player_row = widgets.HBox([self.player],
                                  layout=widgets.Layout(justify_content='center',
                                                        overflow='visible',
                                                        width='100%'))

        self.controls = widgets.VBox([
            section('View & Config'),
            self.dd_view,
            self.chk_safe, self.chk_zoom, self.chk_workspace, self.chk_track,

            section('Actuator Strokes (mm)'),
            strokes_box,

            section('Current Position (mm)'),
            self.lbl_pos,

            section('Manual Adjust (±300 mm)'),
            self.sld_x, self.sld_y, self.sld_z,

            section('Motion Simulation'),
            self.file_upload,
            self.sld_amp,
            player_row,
            self.lbl_progress,
            self.sld_progress,

            widgets.HTML('<hr>'),
            self.btn_precompute,
            self.lbl_status,
        ], layout=widgets.Layout(width='380px', padding='8px',
                                 overflow='visible'))

        self.ui = widgets.HBox([self.fig.canvas, self.controls],
                               layout=widgets.Layout(align_items='flex-start'))

    def show(self):
        display(self.ui)

    # ------------------------------------------------------------------ #
    #  EVENT HANDLERS
    # ------------------------------------------------------------------ #
    def _on_view_change(self, change):
        self._apply_view(change['new'])

    def _apply_view(self, mode):
        if 'Default' in mode:   self.ax_3d.view_init(elev=20, azim=-75)
        elif 'Front' in mode:   self.ax_3d.view_init(elev=0, azim=-90)
        elif 'Side' in mode:    self.ax_3d.view_init(elev=0, azim=0)
        elif 'Top' in mode:     self.ax_3d.view_init(elev=90, azim=-90)
        self.fig.canvas.draw_idle()

    def _on_safety_change(self, change):
        if change['new']:
            self.current_stroke_limit = self.Stroke_Base * 0.8
            self.lbl_status.value = 'Safety Mode: Limit set to 80%.'
            ws = self.workspace_safe
        else:
            self.current_stroke_limit = self.Stroke_Base
            self.lbl_status.value = 'Full Mode: Limit set to 100%.'
            ws = self.workspace_full
        if self.chk_workspace.value:
            self._draw_workspace(ws)
        self.validate_trajectory()
        self._on_manual(None)

    def _on_zoom_change(self, change):
        self._apply_zoom(change['new'])

    def _apply_zoom(self, zoomed):
        if zoomed:
            self.ax_3d.set_xlim([-300, 300])
            self.ax_3d.set_ylim([-300, 300])
            self.ax_3d.set_zlim([-300, 300])
        else:
            self.ax_3d.set_xlim([-1500, 1500])
            self.ax_3d.set_ylim([-1500, 1500])
            self.ax_3d.set_zlim([-1500, 500])
        self._on_manual(None)

    def _on_workspace_change(self, change):
        if change['new']:
            ws = self.workspace_safe if self.chk_safe.value else self.workspace_full
            self._draw_workspace(ws)
        else:
            if self.surf_workspace:
                self.surf_workspace.remove()
                self.surf_workspace = None
            self.fig.canvas.draw_idle()

    def _draw_workspace(self, ws):
        if self.surf_workspace:
            self.surf_workspace.remove()
            self.surf_workspace = None
        if ws is None:
            self.lbl_status.value = '<span style="color:orange">Workspace not computed yet. Click Precompute.</span>'
            return
        pts, simplices = ws
        self.surf_workspace = self.ax_3d.plot_trisurf(
            pts[:,0], pts[:,1], pts[:,2], triangles=simplices,
            color='cyan', alpha=0.35, edgecolor='none', shade=True)
        self.fig.canvas.draw_idle()

    def _on_track_change(self, change):
        vis = change['new']
        self.line_path.set_visible(vis)
        self.scat_fail.set_visible(vis)
        self.fig.canvas.draw_idle()

    def _on_manual(self, change):
        self.is_playing = False
        pos = [self.sld_x.value, self.sld_y.value, self.sld_z.value]
        self.update_platform(pos)

    # ------------------------------------------------------------------ #
    #  PLATFORM UPDATE
    # ------------------------------------------------------------------ #
    def update_platform(self, pos):
        T_curr = np.array(pos)
        P_curr = self.P_local + T_curr

        alpha_plat = 0.05 if self.chk_zoom.value else 1.0

        self.scat_base.set_alpha(alpha_plat)
        self.line_base.set_alpha(alpha_plat)
        for txt in self.text_leg_nums:
            txt.set_alpha(alpha_plat)

        self.scat_plat._offsets3d = (P_curr[:,0], P_curr[:,1], P_curr[:,2])
        self.scat_plat.set_alpha(alpha_plat)

        plat_radius = np.mean(np.sqrt(self.P_local[:,0]**2 + self.P_local[:,1]**2))
        th = np.linspace(0, 2*np.pi, 50)
        xc = plat_radius * np.cos(th) + T_curr[0]
        yc = plat_radius * np.sin(th) + T_curr[1]
        zc = np.zeros_like(th) + T_curr[2]
        self.line_plat.set_data(xc, yc)
        self.line_plat.set_3d_properties(zc)
        self.line_plat.set_alpha(alpha_plat)

        self.scat_center._offsets3d = ([T_curr[0]], [T_curr[1]], [T_curr[2]])

        overall_safe = True
        warn_msg = ''
        strokes = []
        over_limits = []
        for k in range(6):
            leg_vec = P_curr[k] - self.B[k]
            L = np.linalg.norm(leg_vec)
            stroke = L - self.L_mid

            self.lines_legs[k].set_data([self.B[k,0], P_curr[k,0]],
                                        [self.B[k,1], P_curr[k,1]])
            self.lines_legs[k].set_3d_properties([self.B[k,2], P_curr[k,2]])

            color = 'gray'
            is_leg_safe = True

            if abs(stroke) > self.current_stroke_limit:
                color = 'red'; overall_safe = False; is_leg_safe = False
                warn_msg = f'Leg {k+1} Stroke!'

            u = leg_vec / L
            ang = np.degrees(np.arccos(np.clip(np.dot(u, self.Leg_Vec_Neutral[k]), -1, 1)))
            if ang > self.Swivel_Max:
                color = 'orange'; overall_safe = False
                warn_msg = f'Leg {k+1} Angle!'

            self.lines_legs[k].set_color(color)
            self.lines_legs[k].set_alpha(alpha_plat)
            strokes.append(stroke)
            over_limits.append(not is_leg_safe)

        self.lbl_strokes.value = self._build_strokes_html(strokes, over_limits)

        self.scat_center.set_color('b' if overall_safe else 'r')

        if warn_msg:
            self.lbl_status.value = f'<span style="color:red">LIMIT WARNING: {warn_msg}</span>'
        elif not self.is_playing:
            self.lbl_status.value = 'Status: OK'

        self.lbl_pos.value = (f'<b>X:</b> {pos[0]:.1f} | '
                              f'<b>Y:</b> {pos[1]:.1f} | '
                              f'<b>Z:</b> {pos[2]:.1f}')
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------ #
    #  DATA / MOTION
    # ------------------------------------------------------------------ #
    def _on_file_upload(self, change):
        uploaded = change['new']
        if not uploaded:
            return
        # Handle both dict-style (v7) and tuple-style (v8+) FileUpload formats
        if isinstance(uploaded, dict):
            info = list(uploaded.values())[0]
            content = info['content']
            name = info.get('name', info.get('metadata', {}).get('name', 'data'))
        else:
            info = uploaded[0]
            content = getattr(info, 'content', info['content'] if isinstance(info, dict) else None)
            name = getattr(info, 'name', info['name'] if isinstance(info, dict) else 'data')

        raw_bytes = content.tobytes() if hasattr(content, 'tobytes') else content
        try:
            text = raw_bytes.decode('utf-8') if isinstance(raw_bytes, bytes) else raw_bytes
            try:
                raw = np.loadtxt(io.StringIO(text), delimiter=',')
            except Exception:
                raw = np.loadtxt(io.StringIO(text))
        except Exception as e:
            self.lbl_status.value = f'<span style="color:red">Read Error: {e}</span>'
            return
        if raw.ndim < 2 or raw.shape[1] < 4:
            self.lbl_status.value = '<span style="color:red">Format Error: Need 4 cols [Time, X, Y, Z]</span>'
            return

        self.time_vector = raw[:, 0]
        self.raw_data_xyz = raw[:, 1:4]

        n = len(self.time_vector)
        dt = np.mean(np.diff(self.time_vector))
        if dt <= 0:
            dt = 0.01
        step = max(1, int((1.0 / 30) / dt))

        # Configure Play widget to match data length and playback rate
        self.player.max = n - 1
        self.player.step = step
        self.player.interval = 33  # ~30 fps
        self.player.value = 0
        self.player.disabled = False

        self.sld_amp.disabled = False
        self.sld_amp.value = 100
        self.sld_progress.disabled = False
        self.sld_progress.max = n - 1
        self.current_amp = 100.0
        self.apply_amplification()
        self.update_scene_to_index(0)
        self.lbl_status.value = f'Loaded: {name} ({n} points, step={step})'

    def _on_amp_change(self, change):
        self.current_amp = change['new']
        self.apply_amplification()
        self.update_scene_to_index(self.sld_progress.value)

    def apply_amplification(self):
        if self.raw_data_xyz is None:
            return
        self.loaded_data = self.raw_data_xyz * (self.current_amp / 100.0)
        t = self.time_vector
        self.line_tx.set_data(t, self.loaded_data[:, 0])
        self.line_ty.set_data(t, self.loaded_data[:, 1])
        self.line_tz.set_data(t, self.loaded_data[:, 2])
        self.ax_time.relim(); self.ax_time.autoscale_view()
        self.line_path.set_data(self.loaded_data[:, 0], self.loaded_data[:, 1])
        self.line_path.set_3d_properties(self.loaded_data[:, 2])
        self.validate_trajectory()

    def validate_trajectory(self):
        if self.loaded_data is None:
            return
        fails = np.zeros(len(self.loaded_data), dtype=bool)
        for k in range(6):
            vecs = (self.P_local[k] + self.loaded_data) - self.B[k]
            lens = np.linalg.norm(vecs, axis=1)
            fails |= (np.abs(lens - self.L_mid) > self.current_stroke_limit)
            vn = self.Leg_Vec_Neutral[k]
            dots = np.clip(np.sum(vecs * vn, axis=1) / lens, -1, 1)
            fails |= (np.degrees(np.arccos(dots)) > self.Swivel_Max)
        fail_idx = np.where(fails)[0]
        if len(fail_idx) > 0:
            self.scat_fail._offsets3d = (self.loaded_data[fail_idx, 0],
                                         self.loaded_data[fail_idx, 1],
                                         self.loaded_data[fail_idx, 2])
        else:
            self.scat_fail._offsets3d = ([], [], [])
        self.lbl_status.value = f'Data: {len(self.loaded_data)} pts | Fail: {len(fail_idx)}'

    def _on_player_change(self, change):
        """Driven by widgets.Play — runs on the main event loop, no threading."""
        self.is_playing = True
        self.update_scene_to_index(change['new'])

    def _on_progress_change(self, change):
        self.update_scene_to_index(change['new'])

    def update_scene_to_index(self, idx):
        if self.loaded_data is None:
            return
        idx = max(0, min(idx, len(self.loaded_data) - 1))
        pos = self.loaded_data[idx]
        t = self.time_vector[idx]
        self.update_platform(pos.tolist())
        self.line_cursor.set_xdata([t, t])
        self.lbl_progress.value = f'T={t:.2f}s ({idx}/{len(self.loaded_data)})'
        # Update sliders without triggering manual callback
        self.sld_x.unobserve(self._on_manual, names='value')
        self.sld_y.unobserve(self._on_manual, names='value')
        self.sld_z.unobserve(self._on_manual, names='value')
        self.sld_x.value = pos[0]
        self.sld_y.value = pos[1]
        self.sld_z.value = pos[2]
        self.sld_x.observe(self._on_manual, names='value')
        self.sld_y.observe(self._on_manual, names='value')
        self.sld_z.observe(self._on_manual, names='value')

    # ------------------------------------------------------------------ #
    #  WORKSPACE PRECOMPUTE
    # ------------------------------------------------------------------ #
    def _on_precompute(self, btn):
        self.lbl_status.value = '<i>Pre-computing Workspaces... Please wait.</i>'
        self.workspace_full = self._calculate_geometry(self.Stroke_Base)
        self.workspace_safe = self._calculate_geometry(self.Stroke_Base * 0.8)
        self.lbl_status.value = 'Ready (Workspaces Cached).'

    def _calculate_geometry(self, limit_val):
        res = 15; rng = 300
        axis = np.arange(-rng, rng + res, res)
        X, Y, Z = np.meshgrid(axis, axis, axis)
        pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        valid_mask = np.ones(len(pts), dtype=bool)
        for k in range(6):
            offset = self.P_local[k] - self.B[k]
            Legs = pts + offset
            Lengths = np.linalg.norm(Legs, axis=1)
            invalid_stroke = np.abs(Lengths - self.L_mid) > limit_val
            vn = self.Leg_Vec_Neutral[k]
            dots = np.sum(Legs * vn, axis=1) / Lengths
            dots = np.clip(dots, -1.0, 1.0)
            angles = np.degrees(np.arccos(dots))
            invalid_angle = angles > self.Swivel_Max
            valid_mask[invalid_stroke | invalid_angle] = False
        valid_pts = pts[valid_mask]
        if len(valid_pts) < 4:
            return None
        try:
            return (valid_pts, ConvexHull(valid_pts).simplices)
        except Exception:
            return None
