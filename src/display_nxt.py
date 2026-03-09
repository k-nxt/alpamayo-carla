"""
Pygame display for CARLA Alpamayo Agent (NXT)

Shows a real-time dashboard with:
  - 4 camera views (matching the Alpamayo rig)
  - Vehicle HUD (speed, inference count, FPS)
  - Predicted trajectory in bird's-eye view
  - Latest chain-of-thought reasoning text

Optional MP4 recording via ffmpeg (``record_path`` parameter).
The video is written at the *simulation* FPS so that playback runs at
the intended real-time speed regardless of how long inference takes.

ESC or window close to quit.
"""

import numpy as np
import subprocess
import shutil
import time
from typing import Dict, List, Optional

try:
    import pygame
except ImportError:
    pygame = None  # graceful fallback; checked at init time


# ── Layout constants ──────────────────────────────────────────────
WINDOW_W, WINDOW_H = 1280, 760

# Camera strip (top)
_CAM_PAD = 4
_CAM_COLS = 4
_CAM_AREA_W = WINDOW_W - _CAM_PAD * (_CAM_COLS + 1)
_CAM_W = _CAM_AREA_W // _CAM_COLS
_CAM_H = int(_CAM_W * 1080 / 1900)  # keep source aspect ratio
_CAM_LABEL_H = 20
_CAM_STRIP_H = _CAM_H + _CAM_LABEL_H + _CAM_PAD * 2

# Bottom area
_BOTTOM_Y = _CAM_STRIP_H + 4
_BOTTOM_H = WINDOW_H - _BOTTOM_Y
_HUD_W = WINDOW_W // 2
_BEV_W = WINDOW_W - _HUD_W

# Colors
_BG = (18, 18, 24)
_PANEL_BG = (30, 30, 40)
_TEXT = (210, 210, 210)
_TEXT_DIM = (120, 120, 130)
_ACCENT = (80, 180, 255)
_GREEN = (50, 220, 100)
_TRAJ_COLOR = (0, 200, 80)           # selected / optimised trajectory
_TRAJ_RAW_COLOR = (200, 200, 60)     # pre-optimisation trajectory (yellow-ish)
_TRAJ_CANDIDATE_COLORS = [           # palette for candidate trajectories
    (255, 100, 100),  # red
    (100, 180, 255),  # blue
    (255, 200, 60),   # yellow
    (200, 100, 255),  # purple
    (100, 255, 200),  # cyan
    (255, 140, 60),   # orange
    (180, 180, 180),  # grey (fallback)
    (255, 180, 200),  # pink (fallback)
]
_EGO_COLOR = (255, 255, 255)
_ACTUAL_PATH_COLOR = (255, 150, 140)   # pastel red — actual vehicle path (observer)


class Display:
    """
    Optional pygame dashboard for the CARLA Alpamayo agent.

    Usage::

        disp = Display()           # opens window
        disp.tick(cameras, state)  # call every frame
        if disp.should_quit:
            break
        disp.close()
    """

    def __init__(
        self,
        width: int = WINDOW_W,
        height: int = WINDOW_H,
        record_path: Optional[str] = None,
        record_fps: float = 10.0,
        record_crf: int = 23,
    ):
        """
        Args:
            width / height: window size in pixels.
            record_path: if set, record the dashboard to this MP4 file
                via ffmpeg.  The video plays back at ``record_fps``.
            record_fps: frames-per-second for the output video.
                Set this to ``sim_fps`` so the video runs at real-time.
            record_crf: H.264 CRF value (0 = lossless, 23 = default,
                51 = worst).  Lower → better quality / larger file.
        """
        if pygame is None:
            raise ImportError(
                "pygame is required for the display. Install with: pip install pygame"
            )

        pygame.init()
        pygame.display.set_caption("Alpamayo-R1  –  CARLA")
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_sm = pygame.font.SysFont("monospace", 15)
        self.font_xs = pygame.font.SysFont("monospace", 13)

        self.should_quit = False
        self._frame_times: List[float] = []

        # ── Video recording ──
        self._ffmpeg_proc: Optional[subprocess.Popen] = None
        self._record_path = record_path
        if record_path is not None:
            ffmpeg_bin = shutil.which("ffmpeg")
            if ffmpeg_bin is None:
                raise RuntimeError(
                    "ffmpeg not found on PATH. Install ffmpeg to use --record."
                )
            # Ensure width/height are even (H.264 requirement)
            vw = width if width % 2 == 0 else width + 1
            vh = height if height % 2 == 0 else height + 1
            cmd = [
                ffmpeg_bin,
                "-y",                         # overwrite output
                "-f", "rawvideo",
                "-vcodec", "rawvideo",
                "-pix_fmt", "rgb24",
                "-s", f"{vw}x{vh}",
                "-r", str(record_fps),
                "-i", "-",                    # read from stdin
                "-c:v", "libx264",
                "-crf", str(record_crf),
                "-pix_fmt", "yuv420p",
                "-preset", "medium",
                record_path,
            ]
            self._ffmpeg_proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._record_vw = vw
            self._record_vh = vh
            print(f"Recording to {record_path}  "
                  f"(fps={record_fps}, crf={record_crf}, {vw}×{vh})")

    # ── public API ────────────────────────────────────────────────

    def tick(
        self,
        camera_images: Optional[Dict[str, np.ndarray]] = None,
        vehicle_state: Optional[Dict] = None,
        trajectory_xy: Optional[np.ndarray] = None,
        reasoning: Optional[str] = None,
        inference_count: int = 0,
        tick_count: int = 0,
        inference_time: float = 0.0,
        all_trajectories_xy: Optional[np.ndarray] = None,
        selected_traj_index: int = 0,
        raw_trajectory_xy: Optional[np.ndarray] = None,
        # ── Observer-mode extras (ignored when not in observer mode) ──
        observer_mode: bool = False,
        delay_ticks: int = -1,
        actual_path_rig: Optional[np.ndarray] = None,
        autopilot_state: Optional[Dict] = None,
    ) -> None:
        """
        Redraw one frame of the dashboard.

        Args:
            camera_images: camera_name → latest HWC uint8 RGB image.
            vehicle_state: dict with ``speed_kmh``, ``speed_ms``, etc.
            trajectory_xy: (T, 2) active waypoints (may be optimised).
            reasoning: latest chain-of-thought text.
            inference_count: total inference steps so far.
            tick_count: total simulation ticks so far.
            inference_time: wall-clock seconds for the last inference call.
            all_trajectories_xy: (N, T, 2) all candidate trajectories, or None.
            selected_traj_index: index of the selected trajectory.
            raw_trajectory_xy: (T, 2) pre-optimisation model output (or None).
            observer_mode: if True, show observer-specific overlay.
            delay_ticks: ticks elapsed since the inference input was captured
                (-1 = no result yet).
            actual_path_rig: (N, 2) recent actual vehicle path in rig frame.
            autopilot_state: dict with ``throttle``, ``brake``, ``steer`` from
                the autopilot (for HUD display in observer mode).
        """
        self._handle_events()
        if self.should_quit:
            return

        now = time.monotonic()
        self._frame_times.append(now)
        self._frame_times = [t for t in self._frame_times if now - t < 1.0]
        fps = len(self._frame_times)

        self.screen.fill(_BG)

        # ── Camera strip ──
        self._draw_cameras(camera_images)

        # ── HUD (bottom-left) ──
        self._draw_hud(
            vehicle_state, inference_count, tick_count, fps, inference_time,
            observer_mode=observer_mode,
            delay_ticks=delay_ticks,
            autopilot_state=autopilot_state,
        )

        # ── BEV trajectory + reasoning (bottom-right) ──
        self._draw_bev(
            trajectory_xy, all_trajectories_xy, selected_traj_index,
            raw_traj=raw_trajectory_xy,
            actual_path_rig=actual_path_rig,
            observer_mode=observer_mode,
        )
        self._draw_reasoning(reasoning)

        pygame.display.flip()

        # ── Write frame to ffmpeg if recording ──
        if self._ffmpeg_proc is not None and self._ffmpeg_proc.stdin:
            try:
                # Grab the display surface as a 3-D uint8 array (W, H, 3)
                raw = pygame.surfarray.array3d(self.screen)  # (W, H, 3)
                # Transpose to (H, W, 3) = row-major for ffmpeg
                frame = raw.transpose(1, 0, 2)
                # Pad to even dimensions if needed
                if frame.shape[1] != self._record_vw or frame.shape[0] != self._record_vh:
                    padded = np.zeros(
                        (self._record_vh, self._record_vw, 3), dtype=np.uint8
                    )
                    padded[: frame.shape[0], : frame.shape[1]] = frame
                    frame = padded
                self._ffmpeg_proc.stdin.write(frame.tobytes())
            except (BrokenPipeError, OSError):
                # ffmpeg process died; stop trying
                self._ffmpeg_proc = None

        self.clock.tick(30)  # cap display refresh at 30 fps

    def close(self) -> None:
        self._stop_recording()
        pygame.quit()

    def _stop_recording(self) -> None:
        """Finalize the ffmpeg recording."""
        if self._ffmpeg_proc is not None:
            try:
                if self._ffmpeg_proc.stdin:
                    self._ffmpeg_proc.stdin.close()
                self._ffmpeg_proc.wait(timeout=10)
                print(f"Video saved to {self._record_path}")
            except Exception as e:
                print(f"Warning: ffmpeg finalize error: {e}")
            self._ffmpeg_proc = None

    # ── internals ─────────────────────────────────────────────────

    def _handle_events(self) -> None:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                self.should_quit = True
            elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                self.should_quit = True

    # -- cameras ---------------------------------------------------

    _CAM_LABELS = [
        ("camera_cross_left_120fov", "Cross Left 120°"),
        ("camera_front_wide_120fov", "Front Wide 120°"),
        ("camera_cross_right_120fov", "Cross Right 120°"),
        ("camera_front_tele_30fov", "Front Tele 30°"),
    ]

    def _draw_cameras(self, images: Optional[Dict[str, np.ndarray]]) -> None:
        for i, (cam_key, label) in enumerate(self._CAM_LABELS):
            x = _CAM_PAD + i * (_CAM_W + _CAM_PAD)
            y = _CAM_PAD

            # Panel background
            pygame.draw.rect(
                self.screen, _PANEL_BG, (x, y, _CAM_W, _CAM_H + _CAM_LABEL_H)
            )

            # Label
            lbl_surf = self.font_xs.render(label, True, _TEXT_DIM)
            self.screen.blit(lbl_surf, (x + 4, y + 2))

            # Image
            if images and cam_key in images:
                img = images[cam_key]
                surf = pygame.surfarray.make_surface(img.swapaxes(0, 1))
                surf = pygame.transform.scale(surf, (_CAM_W, _CAM_H))
                self.screen.blit(surf, (x, y + _CAM_LABEL_H))
            else:
                # placeholder
                rect = pygame.Rect(x, y + _CAM_LABEL_H, _CAM_W, _CAM_H)
                pygame.draw.rect(self.screen, (40, 40, 50), rect)
                no_img = self.font_sm.render("no image", True, _TEXT_DIM)
                self.screen.blit(
                    no_img,
                    (x + _CAM_W // 2 - no_img.get_width() // 2,
                     y + _CAM_LABEL_H + _CAM_H // 2 - 8),
                )

    # -- HUD -------------------------------------------------------

    def _draw_hud(
        self,
        state: Optional[Dict],
        inf_count: int,
        tick: int,
        fps: int,
        inference_time: float = 0.0,
        observer_mode: bool = False,
        delay_ticks: int = -1,
        autopilot_state: Optional[Dict] = None,
    ) -> None:
        x, y = 0, _BOTTOM_Y
        w, h = _HUD_W, _BOTTOM_H
        pygame.draw.rect(self.screen, _PANEL_BG, (x, y, w, h))

        lines: List[tuple] = []  # (text, color)

        speed = state["speed_kmh"] if state else 0.0
        lines.append((f"Speed    {speed:6.1f} km/h", _TEXT))
        lines.append((f"Tick     {tick:6d}", _TEXT))
        lines.append((f"Infer #  {inf_count:6d}", _ACCENT))

        # Inference time with color coding
        if inference_time > 0:
            it_color = _GREEN if inference_time < 1.0 else (
                (255, 200, 50) if inference_time < 3.0 else (255, 80, 80)
            )
            lines.append((f"Inf Time {inference_time:5.2f} s", it_color))
        else:
            lines.append((f"Inf Time     - s", _TEXT_DIM))

        lines.append((f"FPS      {fps:6d}", _GREEN if fps >= 15 else _TEXT))

        if state:
            loc = state.get("location", {})
            lines.append((
                f"Pos  x={loc.get('x', 0):7.1f}  y={loc.get('y', 0):7.1f}",
                _TEXT_DIM,
            ))
            rot = state.get("rotation", {})
            lines.append((f"Yaw      {rot.get('yaw', 0):6.1f}°", _TEXT_DIM))

        # ── Observer-mode extras ──
        if observer_mode:
            lines.append(("", _TEXT_DIM))  # spacer
            lines.append(("── OBSERVER MODE ──", (255, 180, 60)))

            # Delay info
            if delay_ticks >= 0:
                delay_s = delay_ticks * 0.1  # assuming 10 Hz
                delay_color = (
                    _GREEN if delay_ticks <= 20
                    else ((255, 200, 50) if delay_ticks <= 50 else (255, 80, 80))
                )
                lines.append((
                    f"Delay    {delay_ticks:4d} tick ({delay_s:.1f}s)",
                    delay_color,
                ))
            else:
                lines.append(("Delay       - (waiting)", _TEXT_DIM))

            # Autopilot control state
            ap = autopilot_state or state
            if ap:
                thr = ap.get("throttle", 0.0)
                brk = ap.get("brake", 0.0)
                steer = ap.get("steer", 0.0)
                lines.append((
                    f"AP  Thr {thr:.2f}  Brk {brk:.2f}  Str {steer:+.2f}",
                    _ACTUAL_PATH_COLOR,
                ))

        # Section title
        title_text = "Observer HUD" if observer_mode else "Vehicle HUD"
        title = self.font.render(title_text, True, _ACCENT)
        self.screen.blit(title, (x + 12, y + 8))

        ty = y + 40
        for text, color in lines:
            surf = self.font_sm.render(text, True, color)
            self.screen.blit(surf, (x + 16, ty))
            ty += 22

        # Keyboard hints at bottom
        hints = "ESC: quit"
        h_surf = self.font_xs.render(hints, True, _TEXT_DIM)
        self.screen.blit(h_surf, (x + 16, y + h - 22))

    # -- Bird's-eye trajectory -------------------------------------

    def _draw_bev(
        self,
        traj: Optional[np.ndarray],
        all_traj: Optional[np.ndarray] = None,
        selected_idx: int = 0,
        raw_traj: Optional[np.ndarray] = None,
        actual_path_rig: Optional[np.ndarray] = None,
        observer_mode: bool = False,
    ) -> None:
        bev_x = _HUD_W
        bev_y = _BOTTOM_Y
        bev_w = _BEV_W
        bev_h = _BOTTOM_H // 2
        pygame.draw.rect(self.screen, _PANEL_BG, (bev_x, bev_y, bev_w, bev_h))

        n_candidates = all_traj.shape[0] if all_traj is not None else (1 if traj is not None else 0)
        if observer_mode:
            title_text = f"BEV  (Observer: {n_candidates} samples)"
        else:
            title_text = f"Trajectory BEV  ({n_candidates} samples)"
        title = self.font.render(title_text, True, _ACCENT)
        self.screen.blit(title, (bev_x + 12, bev_y + 8))

        cx = bev_x + bev_w // 2
        cy = bev_y + bev_h - 30  # ego at bottom-center
        scale = 6.0  # pixels per meter

        # Ego marker (triangle pointing up)
        ego_pts = [(cx, cy - 10), (cx - 6, cy + 6), (cx + 6, cy + 6)]
        pygame.draw.polygon(self.screen, _EGO_COLOR, ego_pts)

        def _traj_to_points(t: np.ndarray):
            pts = []
            for wp in t:
                px = cx - int(wp[1] * scale)
                py = cy - int(wp[0] * scale)
                pts.append((px, py))
            return pts

        # Draw all candidate trajectories (thin, semi-transparent)
        if all_traj is not None and len(all_traj) > 1:
            for i, cand in enumerate(all_traj):
                if i == selected_idx:
                    continue  # draw selected last (on top)
                if len(cand) < 2:
                    continue
                pts = _traj_to_points(cand)
                color = _TRAJ_CANDIDATE_COLORS[i % len(_TRAJ_CANDIDATE_COLORS)]
                # Dim the candidate color
                dim = tuple(max(0, c // 2) for c in color)
                pygame.draw.lines(self.screen, dim, False, pts, 1)

        # Draw raw (pre-optimisation) trajectory if available
        if raw_traj is not None and len(raw_traj) >= 2:
            raw_pts = _traj_to_points(raw_traj)
            pygame.draw.lines(self.screen, _TRAJ_RAW_COLOR, False, raw_pts, 2)

        # Draw selected / optimised trajectory (thick, bright green)
        if traj is not None and len(traj) >= 2:
            points = _traj_to_points(traj)
            pygame.draw.lines(self.screen, _TRAJ_COLOR, False, points, 3)

            # Waypoint dots (every 5th)
            for i, pt in enumerate(points):
                if i % 5 == 0:
                    alpha_f = max(0.3, 1.0 - i / len(points))
                    c = tuple(int(v * alpha_f) for v in _TRAJ_COLOR)
                    pygame.draw.circle(self.screen, c, pt, 3)

        # Legend for candidate colors
        if all_traj is not None and len(all_traj) > 1:
            lx = bev_x + bev_w - 90
            ly = bev_y + 32
            for i in range(min(len(all_traj), len(_TRAJ_CANDIDATE_COLORS))):
                color = _TRAJ_CANDIDATE_COLORS[i % len(_TRAJ_CANDIDATE_COLORS)]
                if i == selected_idx:
                    color = _TRAJ_COLOR
                    label = f"#{i} ★"
                else:
                    color = tuple(max(0, c // 2) for c in color)
                    label = f"#{i}"
                pygame.draw.line(self.screen, color, (lx, ly + 6), (lx + 16, ly + 6), 2)
                lbl_surf = self.font_xs.render(label, True, color)
                self.screen.blit(lbl_surf, (lx + 20, ly))
                ly += 16

        # Legend entry for raw vs optimised
        if raw_traj is not None:
            lx = bev_x + 12
            ly = bev_y + 32
            pygame.draw.line(self.screen, _TRAJ_RAW_COLOR, (lx, ly + 6), (lx + 16, ly + 6), 2)
            lbl = self.font_xs.render("raw", True, _TRAJ_RAW_COLOR)
            self.screen.blit(lbl, (lx + 20, ly))
            ly += 16
            pygame.draw.line(self.screen, _TRAJ_COLOR, (lx, ly + 6), (lx + 16, ly + 6), 3)
            lbl = self.font_xs.render("opt", True, _TRAJ_COLOR)
            self.screen.blit(lbl, (lx + 20, ly))

        # Draw actual vehicle path LAST so it renders on top (observer mode)
        if actual_path_rig is not None and len(actual_path_rig) >= 2:
            actual_pts = _traj_to_points(actual_path_rig)
            pygame.draw.lines(self.screen, _ACTUAL_PATH_COLOR, False, actual_pts, 2)

        # Legend for observer mode
        if observer_mode:
            lx = bev_x + 12
            ly = bev_y + 32
            if actual_path_rig is not None:
                pygame.draw.line(
                    self.screen, _ACTUAL_PATH_COLOR,
                    (lx, ly + 6), (lx + 16, ly + 6), 2,
                )
                lbl = self.font_xs.render("actual", True, _ACTUAL_PATH_COLOR)
                self.screen.blit(lbl, (lx + 20, ly))
                ly += 16
            if traj is not None:
                pygame.draw.line(
                    self.screen, _TRAJ_COLOR,
                    (lx, ly + 6), (lx + 16, ly + 6), 3,
                )
                lbl = self.font_xs.render("alpamayo", True, _TRAJ_COLOR)
                self.screen.blit(lbl, (lx + 20, ly))

        # Scale bar
        bar_len_m = 10
        bar_px = int(bar_len_m * scale)
        bar_y = bev_y + bev_h - 14
        bar_x = bev_x + 12
        pygame.draw.line(
            self.screen, _TEXT_DIM, (bar_x, bar_y), (bar_x + bar_px, bar_y), 1
        )
        lbl = self.font_xs.render(f"{bar_len_m}m", True, _TEXT_DIM)
        self.screen.blit(lbl, (bar_x + bar_px + 4, bar_y - 7))

    # -- Reasoning text --------------------------------------------

    def _draw_reasoning(self, reasoning: Optional[str]) -> None:
        rx = _HUD_W
        ry = _BOTTOM_Y + _BOTTOM_H // 2
        rw = _BEV_W
        rh = _BOTTOM_H // 2
        pygame.draw.rect(self.screen, _PANEL_BG, (rx, ry, rw, rh))

        title = self.font.render("Chain of Thought", True, _ACCENT)
        self.screen.blit(title, (rx + 12, ry + 8))

        if not reasoning:
            placeholder = self.font_xs.render(
                "(waiting for inference…)", True, _TEXT_DIM
            )
            self.screen.blit(placeholder, (rx + 16, ry + 38))
            return

        # Word-wrap the reasoning text into the panel
        max_chars = (rw - 32) // 8  # approx chars per line at font_xs
        lines = []
        for paragraph in reasoning.split("\n"):
            while len(paragraph) > max_chars:
                split = paragraph[:max_chars].rfind(" ")
                if split == -1:
                    split = max_chars
                lines.append(paragraph[:split])
                paragraph = paragraph[split:].lstrip()
            lines.append(paragraph)

        ty = ry + 36
        max_lines = (rh - 44) // 16
        for line in lines[:max_lines]:
            surf = self.font_xs.render(line, True, _TEXT)
            self.screen.blit(surf, (rx + 16, ty))
            ty += 16

