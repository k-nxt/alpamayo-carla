"""
Pygame display for CARLA Alpamayo Agent (NXT)

Shows a real-time dashboard with:
  - 4 camera views (matching the Alpamayo rig)
  - Vehicle HUD (speed, inference count, FPS)
  - Predicted trajectory in bird's-eye view
  - Latest chain-of-thought reasoning text

ESC or window close to quit.
"""

import numpy as np
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
_TRAJ_COLOR = (0, 200, 80)
_EGO_COLOR = (255, 255, 255)


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

    def __init__(self, width: int = WINDOW_W, height: int = WINDOW_H):
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
    ) -> None:
        """
        Redraw one frame of the dashboard.

        Args:
            camera_images: camera_name → latest HWC uint8 RGB image.
            vehicle_state: dict with ``speed_kmh``, ``speed_ms``, etc.
            trajectory_xy: (T, 2) predicted waypoints in rig frame (X fwd, Y left).
            reasoning: latest chain-of-thought text.
            inference_count: total inference steps so far.
            tick_count: total simulation ticks so far.
            inference_time: wall-clock seconds for the last inference call.
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
        self._draw_hud(vehicle_state, inference_count, tick_count, fps, inference_time)

        # ── BEV trajectory + reasoning (bottom-right) ──
        self._draw_bev(trajectory_xy)
        self._draw_reasoning(reasoning)

        pygame.display.flip()
        self.clock.tick(30)  # cap display refresh at 30 fps

    def close(self) -> None:
        pygame.quit()

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

        # Section title
        title = self.font.render("Vehicle HUD", True, _ACCENT)
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

    def _draw_bev(self, traj: Optional[np.ndarray]) -> None:
        bev_x = _HUD_W
        bev_y = _BOTTOM_Y
        bev_w = _BEV_W
        bev_h = _BOTTOM_H // 2
        pygame.draw.rect(self.screen, _PANEL_BG, (bev_x, bev_y, bev_w, bev_h))

        title = self.font.render("Trajectory BEV", True, _ACCENT)
        self.screen.blit(title, (bev_x + 12, bev_y + 8))

        cx = bev_x + bev_w // 2
        cy = bev_y + bev_h - 30  # ego at bottom-center

        # Ego marker (triangle pointing up)
        ego_pts = [(cx, cy - 10), (cx - 6, cy + 6), (cx + 6, cy + 6)]
        pygame.draw.polygon(self.screen, _EGO_COLOR, ego_pts)

        if traj is not None and len(traj) >= 2:
            scale = 6.0  # pixels per meter
            points = []
            for wp in traj:
                # rig frame: X forward (screen up), Y left (screen left)
                px = cx - int(wp[1] * scale)
                py = cy - int(wp[0] * scale)
                points.append((px, py))

            # Draw trajectory line
            pygame.draw.lines(self.screen, _TRAJ_COLOR, False, points, 2)

            # Waypoint dots (every 5th)
            for i, pt in enumerate(points):
                if i % 5 == 0:
                    alpha = max(0.3, 1.0 - i / len(points))
                    c = tuple(int(v * alpha) for v in _TRAJ_COLOR)
                    pygame.draw.circle(self.screen, c, pt, 3)

        # Scale bar
        bar_len_m = 10
        bar_px = int(bar_len_m * 6)
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

