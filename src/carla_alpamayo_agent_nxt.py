"""
CARLA Alpamayo Agent (NXT)

Autonomous driving agent that runs Alpamayo-R1 VLA inside CARLA.

- Collects 4-camera temporal frames (default 4 per camera) for model context
- Records ego pose history (16 steps × 0.1 s) in rig frame
- Converts predicted trajectory waypoints into CARLA VehicleControl
  via Pure-Pursuit steering + proportional speed control
"""

import carla
import numpy as np
import torch
import time
import threading
from typing import Dict, Optional
from dataclasses import dataclass
from collections import deque
from scipy.spatial.transform import Rotation

from .alpamayo_wrapper_nxt import AlpamayoWrapper, AlpamayoOutput, detect_model_version, VERSION_15
from .sensor_manager_nxt import SensorManager, ALPAMAYO_CAMERA_ORDER, RESOLUTION_PRESETS
from .display_nxt import Display
from .trajectory_optimizer_nxt import (
    TrajectoryOptimizer,
    VehicleConstraints,
    add_heading_to_trajectory,
    OptimizationResult,
)

# ---------------------------------------------------------------------------
# Constants (must match Alpamayo-R1 expectations)
# ---------------------------------------------------------------------------
NUM_HISTORY_STEPS = 16    # ego history length
HISTORY_TIME_STEP = 0.1   # seconds between history samples

# CARLA rear-axle offset from bounding-box center in vehicle-local frame.
# CARLA's get_transform() returns the bounding-box center, but Alpamayo's
# rig origin is at the rear axle.  This offset converts between the two.
# Value from alpasim's transfuser_impl.py (CARLA_REAR_AXLE).
_REAR_AXLE_OFFSET_LOCAL = np.array([-1.389, 0.0, 0.0])


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class EgoPose:
    """Vehicle pose snapshot in CARLA world coordinates."""
    timestamp_us: int
    location: np.ndarray          # (3,) world x y z
    rotation_matrix: np.ndarray   # (3, 3) world rotation
    rotation_quat: np.ndarray     # (4,) world rotation quaternion (x, y, z, w)


@dataclass
class AgentConfig:
    """Configuration for the CARLA Alpamayo agent."""
    # CARLA connection
    host: str = "localhost"
    port: int = 2000
    timeout: float = 10.0
    map_name: Optional[str] = None  # e.g. "Town03" — None = keep current map
    weather: str = "ClearNoon"      # weather preset name (see WEATHER_PRESETS)

    # Vehicle
    vehicle_filter: str = "vehicle.tesla.model3"
    spawn_point_index: int = -1   # -1 = random

    # Alpamayo model
    model_name: str = "nvidia/Alpamayo-1.5-10B"
    use_dummy_model: bool = False
    context_length: int = 4       # temporal image frames
    num_traj_samples: int = 6
    max_generation_length: int = 256  # VLM max tokens (64 is faster but may truncate reasoning)
    diffusion_steps: int = 5          # Flow-matching denoising steps (default 10, 5 for speed)
    cam_resolution: str = "full"      # Camera resolution: "full", "half", "low", or "WxH"
    vlm_temperature: float = 0.6      # VLM text-generation temperature (lower = more deterministic)
    vlm_top_p: float = 0.98           # VLM nucleus-sampling threshold
    timing_log: bool = False          # print per-inference timing breakdown

    # Simulation
    sim_fps: float = 10.0         # 10 Hz = 0.1 s/tick, matching AR1 training data
    inference_interval: int = 1   # run inference every tick (= every 0.1 s)
    inference_interval_sec: Optional[float] = None  # optional wall-clock inference interval
    use_async_inference: bool = False  # default: sync-wait mode (worker thread + blocking)
    num_npc_vehicles: int = 0     # number of NPC vehicles to spawn
    num_npc_walkers: int = 0      # number of NPC pedestrians to spawn

    # Control
    control_mode: str = "legacy"      # "legacy" (TrajectoryFollower) or "official_pid"
    max_speed_kmh: float = 30.0
    min_speed_kmh: float = 0.0        # 0 = no minimum; >0 = floor for desired speed
    lookahead_distance: float = 5.0   # pure-pursuit look-ahead (m)
    steer_gain: float = 1.0           # steering multiplier (>1 = sharper turns)
    steer_normalize_deg: float = 70.0 # steering normalisation angle (deg)
    # Official PID follower parameters (used when control_mode="official_pid")
    pid_lookahead_min_m: float = 4.0
    pid_lookahead_max_m: float = 12.0
    pid_lookahead_speed_gain: float = 0.4
    pid_target_speed_min_kmh: float = 10.0
    pid_target_speed_max_kmh: float = 35.0
    pid_target_speed_extent_gain: float = 0.5
    pid_lat_kp: float = 1.1
    pid_lat_ki: float = 0.02
    pid_lat_kd: float = 0.15
    pid_lon_kp: float = 0.6
    pid_lon_ki: float = 0.05
    pid_lon_kd: float = 0.0
    pid_max_throttle: float = 0.35
    pid_max_brake: float = 1.0

    # Display
    enable_display: bool = True       # pygame dashboard window
    display_camera_downsample: int = 1  # display-only camera downsample factor
    display_max_cameras: int = 4        # max camera panels to render
    display_fetch_cameras: bool = True  # fetch camera frames for display path
    loop_timing_log: bool = False       # print world/step/display loop timings

    # Video recording
    record_path: Optional[str] = None  # MP4 output path (None = no recording)
    record_crf: int = 23              # H.264 CRF (0=lossless, 23=default, 51=worst)

    # Navigation (Alpamayo 1.5)
    nav_enabled: bool = True              # enable nav instructions (auto-disabled for R1)
    use_cfg_nav: bool = False             # use classifier-free guidance for nav
    cfg_nav_guidance_weight: Optional[float] = None  # CFG alpha (None = model default)
    nav_destination_index: int = -1       # spawn-point index for route destination (-1 = random)
    nav_text_override: Optional[str] = None  # fixed nav text for debugging (overrides planner)
    use_camera_indices: bool = True       # include camera indices in 1.5 prompt construction

    # Debug log
    debug_log_path: Optional[str] = None   # CSV log file path (None = disabled)

    # Trajectory optimiser (post-processing for smoothness / comfort)
    traj_opt_enabled: bool = False          # enable trajectory optimisation
    traj_opt_smoothness_w: float = 1.0      # smoothness cost weight
    traj_opt_deviation_w: float = 0.1       # deviation from model output weight
    traj_opt_comfort_w: float = 2.0         # comfort penalty weight
    traj_opt_max_iter: int = 50             # max scipy iterations
    traj_opt_retime: bool = True            # Frenet retiming
    traj_opt_retime_alpha: float = 0.25     # retiming strength [0..1]


# ---------------------------------------------------------------------------
# Trajectory → vehicle-control converter
# ---------------------------------------------------------------------------
class TrajectoryFollower:
    """
    Pure-Pursuit + proportional speed controller.

    Converts Alpamayo's predicted trajectory (rig frame waypoints)
    into a CARLA ``VehicleControl``.

    Because Alpamayo uses stochastic sampling (temperature, top-p),
    consecutive inferences with near-identical inputs can produce
    wildly different trajectories (long "go" vs. short "stop").
    To prevent the vehicle from oscillating between throttle and brake,
    the raw desired-speed is smoothed via an exponential moving average
    (EMA) before being used for throttle/brake decisions.
    """

    # EMA smoothing factor for *acceleration* (speed increasing).
    # Low value absorbs the stochastic oscillation from the diffusion
    # model (which frequently alternates between "go" and "stop"
    # trajectories even on clear straight roads).
    SPEED_EMA_ALPHA_ACCEL = 0.15

    # EMA smoothing factor for *deceleration* (speed decreasing).
    # Higher value allows the car to respond quickly to curves,
    # obstacles, and stop commands without the sluggish response
    # of the acceleration EMA.
    SPEED_EMA_ALPHA_DECEL = 0.5

    # EMA smoothing for steering to prevent sudden direction changes.
    # 0.7 favours the *new* value — responsive enough for curves
    # while still filtering stochastic noise from the diffusion model.
    STEER_EMA_ALPHA = 0.7

    # After this many consecutive ticks at standstill with no meaningful
    # trajectory, apply a gentle "nudge" throttle to give the model
    # motion cues in its temporal frames.
    STANDSTILL_NUDGE_TICKS = 15
    STANDSTILL_NUDGE_THROTTLE = 0.8

    def __init__(
        self,
        lookahead_distance: float = 5.0,
        max_speed_kmh: float = 30.0,
        min_speed_kmh: float = 0.0,
        steer_gain: float = 1.0,
        steer_normalize_deg: float = 70.0,
        wheelbase: float = 2.875,          # Mercedes coupe approx
        output_frequency_hz: float = 10.0,  # Alpamayo outputs at 10 Hz
    ):
        self.lookahead_distance = lookahead_distance
        self.max_speed_ms = max_speed_kmh / 3.6
        self.min_speed_ms = min_speed_kmh / 3.6
        self.steer_gain = steer_gain
        if steer_normalize_deg <= 0.0:
            raise ValueError("steer_normalize_deg must be > 0")
        self._steer_norm_rad = np.radians(steer_normalize_deg)
        self.wheelbase = wheelbase
        self.dt = 1.0 / output_frequency_hz

        self.trajectory: Optional[np.ndarray] = None
        self.headings: Optional[np.ndarray] = None
        self._raw_desired_speed: float = 0.0   # instant (before EMA)
        self._last_desired_speed: float = 0.0  # smoothed (after EMA), for display
        self._last_steer: float = 0.0          # EMA-smoothed steering
        self._raw_steer: float = 0.0           # instant steering (for display)
        self._hdg_weight: float = 0.0          # heading blend weight (for logging)
        self._pp_steer: float = 0.0            # pure-pursuit component (for logging)
        self._hdg_steer: float = 0.0           # heading component (for logging)
        self._explicit_stop: bool = False       # trajectory < 0.5m (for logging)
        self._last_control: Optional[carla.VehicleControl] = None  # for logging
        # Start in "stuck" state so NUDGE can trigger immediately after launch.
        self._standstill_ticks: int = self.STANDSTILL_NUDGE_TICKS
        self._curve_safe_speed: float = 999.0  # curvature-based speed limit (for logging)
        self._smoothed_curve_safe: float = 999.0  # EMA-smoothed curve-safe speed
        self._short_traj_count: int = 0        # consecutive short-trajectory ticks

    def set_trajectory(
        self,
        trajectory_xy: np.ndarray,
        headings: np.ndarray,
    ) -> None:
        """
        Store a new reference trajectory.

        Args:
            trajectory_xy: (T, 2) in rig frame (X fwd, Y left).
            headings: (T,) heading angles in radians.
        """
        self.trajectory = trajectory_xy
        self.headings = headings

    def compute_control(self, current_speed_ms: float) -> carla.VehicleControl:
        """Compute vehicle control from stored trajectory."""
        if self.trajectory is None or len(self.trajectory) == 0:
            ctrl = carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)
            self._last_control = ctrl
            return ctrl

        traj_length = self._trajectory_length()

        # ── Steering: Pure Pursuit + Heading blend ──
        #
        # Pure Pursuit provides smooth path-following but tends to
        # *under-steer* on tight curves because the lookahead point
        # smooths out the curvature.
        #
        # The trajectory headings give us a direct measure of the
        # desired yaw change.  By blending the two we get the best
        # of both: stable tracking on straights with sharper response
        # in curves.
        #
        # Blend ratio: heading contribution increases with curvature.

        # Speed-adaptive lookahead (conservative):
        # keep high-speed lookahead close to the base distance to avoid
        # under-steer at intersections / tight bends.
        la_base = self.lookahead_distance
        la_adaptive = np.clip(la_base * (0.6 + current_speed_ms / 12.0), 2.0, la_base)

        lookahead = self._find_lookahead_point(override_distance=la_adaptive)
        if lookahead is None:
            ctrl = carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.5)
            self._last_control = ctrl
            return ctrl

        dx, dy = float(lookahead[0]), float(lookahead[1])
        dist = np.hypot(dx, dy)

        # --- Component 1: Pure Pursuit ---
        if dist < 0.1:
            pp_steer = 0.0
        else:
            curvature = 2.0 * dy / (dist ** 2)
            steer_rad = np.arctan(curvature * self.wheelbase)
            pp_steer = float(np.clip(steer_rad / self._steer_norm_rad, -1.0, 1.0))
            # Rig Y-left positive → CARLA steer-right positive ⇒ flip
            pp_steer = -pp_steer

        self._pp_steer = pp_steer

        # --- Component 2: Heading-based steering ---
        # Use the heading at a point ~1-2s ahead to determine the
        # desired yaw change, then convert to a steering command.
        hdg_steer = self._heading_based_steer(current_speed_ms)
        self._hdg_steer = hdg_steer

        # --- Adaptive blend ---
        # On straights (both components ≈ 0) Pure Pursuit dominates.
        # When heading component is large (curve), it gets more weight.
        # Blend weight = clamp(|hdg_steer| * 5, 0, 0.7)
        # The high cap (0.7) allows heading-based steering to dominate
        # during tight turns where Pure Pursuit under-steers due to
        # the lookahead smoothing out the curvature.
        hdg_weight = float(np.clip(abs(hdg_steer) * 5.0, 0.0, 0.7))
        self._hdg_weight = hdg_weight
        raw_steer = (1.0 - hdg_weight) * pp_steer + hdg_weight * hdg_steer

        # Apply user-configurable steering gain (>1 for sharper turns)
        raw_steer = float(np.clip(raw_steer * self.steer_gain, -1.0, 1.0))

        # EMA-smooth steering to prevent abrupt direction changes
        # from stochastic trajectory predictions.
        # When the heading blend is active (curve detected), use a
        # faster EMA so the steering responds quickly enough to
        # track the curve.  On straights, keep the slower EMA to
        # filter out noise.
        self._raw_steer = raw_steer
        steer_alpha = self.STEER_EMA_ALPHA  # 0.7 base
        if hdg_weight > 0.2:
            # In a curve — allow faster steering response (up to 0.9)
            steer_alpha = min(0.9, self.STEER_EMA_ALPHA + hdg_weight * 0.4)
        steer = steer_alpha * raw_steer + (1.0 - steer_alpha) * self._last_steer
        self._last_steer = steer

        # ── Speed target ──
        raw_speed = min(self._estimate_desired_speed(), self.max_speed_ms)

        # ── Stop-intent detection ──
        #
        # The model signals "stop" via short trajectories.  A single
        # short trajectory can be stochastic noise, so we track how
        # many consecutive ticks produce short trajectories.
        #
        # STOP_TRAJ_THRESHOLD: trajectories shorter than this are
        #   considered a "stop/slow" signal from the model.
        # STOP_CONFIRM_TICKS: require this many consecutive short
        #   trajectories before treating it as a genuine stop command
        #   (overriding min_speed).
        _STOP_TRAJ_THRESHOLD = 3.0   # meters
        _STOP_CONFIRM_TICKS = 3      # consecutive ticks

        if traj_length < _STOP_TRAJ_THRESHOLD:
            self._short_traj_count += 1
        else:
            self._short_traj_count = 0

        explicit_stop = traj_length < 0.5
        self._explicit_stop = explicit_stop
        if explicit_stop:
            raw_speed = 0.0

        # Model wants to stop: confirmed by N consecutive short
        # trajectories.  Don't force min_speed in this case.
        model_wants_stop = self._short_traj_count >= _STOP_CONFIRM_TICKS

        self._raw_desired_speed = raw_speed

        # Asymmetric EMA: fast decel (react to curves / stops quickly),
        # slow accel (absorb stochastic go/stop oscillation).
        if raw_speed < self._last_desired_speed:
            alpha = self.SPEED_EMA_ALPHA_DECEL   # 0.5 — responsive
        else:
            alpha = self.SPEED_EMA_ALPHA_ACCEL   # 0.15 — conservative
        desired_speed = alpha * raw_speed + (1.0 - alpha) * self._last_desired_speed

        # ── Curvature-based speed limit ──
        max_curv = self._estimate_max_curvature()
        _A_LAT_MAX = 2.5   # comfortable lateral acceleration (m/s²)
        if max_curv > 0.005:
            curve_safe_raw = float(np.sqrt(_A_LAT_MAX / max_curv))
        else:
            curve_safe_raw = 999.0

        if curve_safe_raw < self._smoothed_curve_safe:
            cs_alpha = 0.7
        else:
            cs_alpha = 0.02
        self._smoothed_curve_safe = (
            cs_alpha * curve_safe_raw
            + (1.0 - cs_alpha) * self._smoothed_curve_safe
        )
        curve_safe = self._smoothed_curve_safe
        self._curve_safe_speed = curve_safe

        # ── Minimum speed floor ──
        #
        # Apply min_speed ONLY when the model is NOT signalling a stop.
        # "model_wants_stop" = N consecutive short trajectories.
        # This allows the car to cruise at min_speed on clear roads
        # (where short trajectories are stochastic noise) while still
        # stopping at red lights (where the model consistently produces
        # short trajectories).
        # If the model predicts a clear turn (larger steering demand),
        # relax the min-speed floor so the car can slow down enough to rotate.
        turning_intent = max(abs(self._pp_steer), abs(self._hdg_steer)) > 0.12
        if self.min_speed_ms > 0 and not model_wants_stop:
            effective_min = min(self.min_speed_ms, curve_safe)
            if turning_intent:
                effective_min *= 0.6
            desired_speed = max(desired_speed, effective_min)

        # Hard-cap: never exceed the curvature-safe speed.
        if curve_safe < 100.0:
            desired_speed = min(desired_speed, curve_safe)

        self._last_desired_speed = desired_speed

        # ── Standstill tracker ──
        if current_speed_ms < 0.1:
            self._standstill_ticks += 1
        else:
            self._standstill_ticks = 0

        # ── Throttle / brake ──
        err = desired_speed - current_speed_ms

        at_standstill = current_speed_ms < 0.5

        if at_standstill:
            # --- Standstill logic ---
            # When already stopped, NEVER apply active brake — just
            # release throttle.  The car won't roll on flat ground,
            # and active braking prevents the occasional positive
            # trajectory from producing any movement.
            if desired_speed > 0.10:
                # Model wants to go
                throttle = float(np.clip(err * 0.5, 0.35, 0.8))
                brake = 0.0
            elif self._standstill_ticks >= self.STANDSTILL_NUDGE_TICKS:
                # Stuck too long — apply a gentle nudge to give the
                # diffusion model motion cues in its temporal frames.
                # Many maps/spawns produce near-zero trajectories when
                # all 4 temporal frames are identical (no motion).
                throttle = self.STANDSTILL_NUDGE_THROTTLE
                brake = 0.0
            else:
                # Desired speed ≈ 0 and not stuck yet — coast (no brake)
                throttle = 0.0
                brake = 0.0
        else:
            # --- Moving logic (feed-forward + proportional) ---
            #
            # Pure proportional control (P-only) suffers from steady-
            # state error: at equilibrium the throttle exactly cancels
            # drag, leaving a permanent gap to the target speed.
            #
            # Fix: add a *feed-forward* term that estimates the base
            # throttle needed to maintain ``desired_speed`` against drag.
            # The proportional term then only needs to close the gap.
            #
            #   throttle = ff(desired_speed) + Kp * max(err, 0)
            #
            # Empirically, CARLA's Mercedes coupe needs approximately
            # 0.07 × speed_ms throttle to hold a given speed on flat
            # road (this absorbs rolling + aero drag).
            _FF_COEFF = 0.07   # feed-forward coefficient
            _KP = 0.5          # proportional gain for positive error

            stop_threshold = 0.30

            if desired_speed < stop_threshold:
                # Model wants a stop
                throttle = 0.0
                brake = float(np.clip(0.5 + current_speed_ms * 0.1, 0.3, 1.0))
            elif err < -1.0:
                # Way too fast — active braking
                throttle = 0.0
                brake = float(np.clip(-err * 0.3, 0.1, 1.0))
            elif err < 0:
                # Slightly above target — coast with reduced throttle
                coast_thr = desired_speed * _FF_COEFF * 0.7
                throttle = float(np.clip(coast_thr, 0.0, 0.3))
                brake = 0.0
            else:
                # At or below target — feed-forward + proportional
                ff = desired_speed * _FF_COEFF
                p = err * _KP
                throttle = float(np.clip(ff + p, 0.1, 0.8))
                brake = 0.0

        ctrl = carla.VehicleControl(
            throttle=throttle,
            steer=steer,
            brake=brake,
        )
        self._last_control = ctrl
        return ctrl

    # -- helpers --

    def _trajectory_length(self) -> float:
        """Total arc length of the stored trajectory (meters)."""
        traj = self.trajectory
        if traj is None or len(traj) < 2:
            return 0.0
        diffs = np.diff(traj, axis=0)
        return float(np.sum(np.linalg.norm(diffs, axis=1)))

    def _find_lookahead_point(
        self,
        override_distance: Optional[float] = None,
    ) -> Optional[np.ndarray]:
        """Interpolate to find the point at the given distance along the trajectory."""
        traj = self.trajectory
        if traj is None:
            return None

        la_dist = override_distance if override_distance is not None else self.lookahead_distance

        diffs = np.diff(traj, axis=0)
        seg_len = np.linalg.norm(diffs, axis=1)
        cum_dist = np.concatenate([[0.0], np.cumsum(seg_len)])

        if cum_dist[-1] < la_dist:
            return traj[-1]

        idx = int(np.searchsorted(cum_dist, la_dist))
        if idx >= len(traj):
            return traj[-1]
        if idx == 0:
            return traj[0]

        frac = (la_dist - cum_dist[idx - 1]) / (
            cum_dist[idx] - cum_dist[idx - 1] + 1e-6
        )
        return traj[idx - 1] + frac * (traj[idx] - traj[idx - 1])

    def _estimate_desired_speed(self) -> float:
        """
        Estimate desired speed from the **local** speed profile of the
        trajectory, rather than a global average.

        The trajectory's waypoints are spaced at ``self.dt`` intervals.
        The local speed at waypoint *i* is ``|wp[i+1] − wp[i]| / dt``.
        We look at a window around the point that is ~1-2 seconds ahead
        of the vehicle (roughly where the car will be soon) and return
        the **median** speed in that window.  Median is robust against
        single-waypoint noise.

        Falls back to total_length / total_time if the trajectory is
        very short (< 5 waypoints).
        """
        traj = self.trajectory
        if traj is None or len(traj) < 2:
            return 0.0

        # Per-segment speeds
        diffs = np.diff(traj, axis=0)                      # (T-1, 2)
        seg_speeds = np.linalg.norm(diffs, axis=1) / self.dt  # (T-1,)

        if len(seg_speeds) < 5:
            # Too short for a meaningful local window — use global avg
            total_length = float(np.sum(np.linalg.norm(diffs, axis=1)))
            total_time = (len(traj) - 1) * self.dt
            return total_length / max(total_time, 1e-6)

        # Window: waypoints 5..25 (≈ 0.5s – 2.5s ahead).
        # This captures the "near future" where the car should match
        # the trajectory's speed.  Waypoints 0-4 are very close to the
        # origin and often noisy; waypoints 25+ are far ahead and may
        # reflect conditions the car won't reach for many seconds.
        win_start = min(5, len(seg_speeds) - 1)
        win_end = min(25, len(seg_speeds))
        window = seg_speeds[win_start:win_end]

        if len(window) == 0:
            window = seg_speeds

        return float(np.median(window))

    def _estimate_max_curvature(self) -> float:
        """
        Estimate the maximum curvature (1/R) in the stored trajectory.

        Uses the Menger curvature formula on consecutive triplets of
        waypoints:  κ = 4·Area / (a·b·c).

        The 2-D cross product ``|d01 × d02|`` gives *twice* the
        triangle area, so κ = 2·|cross| / (a·b·c).

        Returns 0 if the trajectory is too short.
        """
        traj = self.trajectory
        if traj is None or len(traj) < 3:
            return 0.0

        # Sample every few points to be robust against noise
        step = max(1, len(traj) // 20)
        indices = list(range(0, len(traj) - 2 * step, step))
        if not indices:
            return 0.0

        max_k = 0.0
        for i in indices:
            p0 = traj[i]
            p1 = traj[i + step]
            p2 = traj[i + 2 * step]
            # Triangle area via cross product:  |cross| = 2 * area
            d01 = p1 - p0
            d02 = p2 - p0
            cross = abs(float(d01[0] * d02[1] - d01[1] * d02[0]))
            a = float(np.linalg.norm(d01))
            b = float(np.linalg.norm(p2 - p1))
            c = float(np.linalg.norm(d02))
            denom = a * b * c
            if denom < 1e-6:
                continue
            # Menger curvature:  κ = 4·area / (a·b·c) = 2·|cross| / (a·b·c)
            k = 2.0 * cross / denom
            max_k = max(max_k, k)
        return max_k

    def _heading_based_steer(self, current_speed_ms: float) -> float:
        """
        Compute a steering command from the trajectory's heading profile.

        Strategy: look at the heading at a point ~1-2 seconds ahead in the
        trajectory and compute the yaw change needed.  Convert that to a
        normalised steering command via the bicycle model:

            steer_rad = arctan(yaw_rate × wheelbase / speed)

        where yaw_rate ≈ Δheading / Δt.

        This gives a more direct measure of how sharply the road curves
        than Pure Pursuit, which is biased by the lookahead distance.
        """
        traj = self.trajectory
        headings = self.headings
        if traj is None or headings is None or len(headings) < 5:
            return 0.0

        # Pick a "look-ahead" index in time:
        #   At 8 m/s (~30 km/h), we want ~1.5 s ahead → index 15
        #   At 3 m/s (~10 km/h), we want ~0.8 s ahead → index 8
        #   Clamp to valid range.
        look_time = np.clip(0.15 * current_speed_ms + 0.5, 0.5, 2.0)  # seconds
        look_idx = int(look_time / self.dt)
        look_idx = min(look_idx, len(headings) - 1)

        # Heading at origin is 0 (facing forward in rig frame).
        # heading[i] is the heading at waypoint i.
        target_heading = float(headings[look_idx])

        # Wrap to [-π, π]
        target_heading = (target_heading + np.pi) % (2.0 * np.pi) - np.pi

        # Geometric approach: compute the arc-length to the look-ahead
        # waypoint, then estimate the curvature needed to sweep the
        # heading angle θ over that distance d:
        #   κ = 2 sin(θ) / d    (chord-based approximation)
        # Then convert to steering via the bicycle model.
        look_d = 0.0
        for i in range(min(look_idx, len(traj) - 1)):
            look_d += float(np.linalg.norm(traj[i + 1] - traj[i]))

        if look_d < 0.3:
            return 0.0

        # Required curvature to sweep heading θ over distance d:
        #   κ = 2 sin(θ) / d  (chord-based approximation)
        sin_th = np.sin(target_heading)
        kappa = 2.0 * sin_th / look_d

        steer_rad = np.arctan(kappa * self.wheelbase)
        hdg_steer = float(np.clip(steer_rad / self._steer_norm_rad, -1.0, 1.0))

        # Flip: rig Y-left positive → CARLA steer-right positive
        hdg_steer = -hdg_steer

        return hdg_steer


class OfficialPIDTrajectoryFollower:
    """
    CARLA official PID follower using Alpamayo trajectory as waypoint reference.

    This mode intentionally keeps control logic simple and close to CARLA's
    built-in VehiclePIDController behavior.
    """

    STANDSTILL_NUDGE_TICKS = 1_000_000
    STANDSTILL_NUDGE_THROTTLE = 0.0

    def __init__(
        self,
        world: carla.World,
        vehicle: carla.Actor,
        output_frequency_hz: float = 10.0,
        max_speed_kmh: float = 30.0,
        min_speed_kmh: float = 0.0,
        lookahead_min_m: float = 4.0,
        lookahead_max_m: float = 12.0,
        lookahead_speed_gain: float = 0.4,
        target_speed_min_kmh: float = 10.0,
        target_speed_max_kmh: float = 35.0,
        target_speed_extent_gain: float = 0.5,
        lat_kp: float = 1.1,
        lat_ki: float = 0.02,
        lat_kd: float = 0.15,
        lon_kp: float = 0.6,
        lon_ki: float = 0.05,
        lon_kd: float = 0.0,
        max_throttle: float = 0.35,
        max_brake: float = 1.0,
    ):
        from agents.navigation.controller import VehiclePIDController

        self.world = world
        self.map = world.get_map()
        self.vehicle = vehicle

        self.max_speed_ms = max_speed_kmh / 3.6
        self.min_speed_ms = min_speed_kmh / 3.6
        self._lookahead_min_m = lookahead_min_m
        self._lookahead_max_m = lookahead_max_m
        self._lookahead_speed_gain = lookahead_speed_gain
        self._target_speed_min_kmh = target_speed_min_kmh
        self._target_speed_max_kmh = target_speed_max_kmh
        self._target_speed_extent_gain = target_speed_extent_gain

        dt = 1.0 / max(output_frequency_hz, 1e-6)
        self.pid = VehiclePIDController(
            vehicle,
            args_lateral={"K_P": lat_kp, "K_I": lat_ki, "K_D": lat_kd, "dt": dt},
            args_longitudinal={"K_P": lon_kp, "K_I": lon_ki, "K_D": lon_kd, "dt": dt},
            max_throttle=max_throttle,
            max_brake=max_brake,
            max_steering=0.8,
        )

        self.trajectory: Optional[np.ndarray] = None
        self.headings: Optional[np.ndarray] = None
        self._raw_desired_speed: float = 0.0
        self._last_desired_speed: float = 0.0
        self._last_steer: float = 0.0
        self._raw_steer: float = 0.0
        self._hdg_weight: float = 0.0
        self._pp_steer: float = 0.0
        self._hdg_steer: float = 0.0
        self._explicit_stop: bool = False
        self._last_control: Optional[carla.VehicleControl] = None
        self._standstill_ticks: int = 0
        self._curve_safe_speed: float = 999.0
        self._short_traj_count: int = 0

    def set_trajectory(self, trajectory_xy: np.ndarray, headings: np.ndarray) -> None:
        self.trajectory = trajectory_xy
        self.headings = headings

    def _trajectory_length(self) -> float:
        traj = self.trajectory
        if traj is None or len(traj) < 2:
            return 0.0
        diffs = np.diff(traj, axis=0)
        return float(np.sum(np.linalg.norm(diffs, axis=1)))

    @staticmethod
    def _alpamayo_to_carla_local(traj_xy: np.ndarray) -> np.ndarray:
        wp = np.asarray(traj_xy, dtype=np.float64).copy()
        wp[:, 1] *= -1.0  # rig y-left -> CARLA y-right
        return wp

    def _local_to_world(self, wp_local_xy: np.ndarray) -> np.ndarray:
        tr = self.vehicle.get_transform()
        wp_world = []
        for p in wp_local_xy:
            w = tr.transform(carla.Location(x=float(p[0]), y=float(p[1]), z=0.0))
            wp_world.append([w.x, w.y, w.z])
        return np.asarray(wp_world, dtype=np.float64)

    def _pick_target_index(self, wp_world_xy: np.ndarray, speed_mps: float) -> int:
        lookahead_m = float(np.clip(
            self._lookahead_min_m + self._lookahead_speed_gain * speed_mps,
            self._lookahead_min_m,
            self._lookahead_max_m,
        ))
        if len(wp_world_xy) <= 1:
            return 0
        seg = np.linalg.norm(np.diff(wp_world_xy[:, :2], axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        return int(min(np.searchsorted(cum, lookahead_m), len(wp_world_xy) - 1))

    def compute_control(self, current_speed_ms: float) -> carla.VehicleControl:
        if self.trajectory is None or len(self.trajectory) == 0:
            ctrl = carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)
            self._last_control = ctrl
            return ctrl

        if current_speed_ms < 0.1:
            self._standstill_ticks += 1
        else:
            self._standstill_ticks = 0

        traj_local = self._alpamayo_to_carla_local(self.trajectory)
        traj_world = self._local_to_world(traj_local)
        target_idx = self._pick_target_index(traj_world, current_speed_ms)
        target_loc = carla.Location(
            x=float(traj_world[target_idx, 0]),
            y=float(traj_world[target_idx, 1]),
            z=float(traj_world[target_idx, 2]),
        )
        target_wp = self.map.get_waypoint(
            target_loc, project_to_road=True, lane_type=carla.LaneType.Driving
        )
        if target_wp is None:
            ctrl = carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.3)
            self._last_control = ctrl
            return ctrl

        traj_extent = float(np.max(np.linalg.norm(traj_local[:, :2], axis=1)))
        target_speed_kmh = float(np.clip(
            self._target_speed_min_kmh + self._target_speed_extent_gain * traj_extent,
            self._target_speed_min_kmh,
            self._target_speed_max_kmh,
        ))
        target_speed_kmh = min(target_speed_kmh, self.max_speed_ms * 3.6)

        pid_ctrl = self.pid.run_step(target_speed_kmh, target_wp)
        ctrl = carla.VehicleControl(
            throttle=float(pid_ctrl.throttle),
            steer=float(pid_ctrl.steer),
            brake=float(pid_ctrl.brake),
        )

        self._raw_desired_speed = target_speed_kmh / 3.6
        self._last_desired_speed = self._raw_desired_speed
        self._raw_steer = ctrl.steer
        self._last_steer = ctrl.steer
        self._pp_steer = ctrl.steer
        self._hdg_steer = 0.0
        self._hdg_weight = 0.0
        self._explicit_stop = self._trajectory_length() < 0.5
        self._short_traj_count = 0
        self._last_control = ctrl
        return ctrl


@dataclass
class _InferenceRequest:
    """Payload sent from main thread to inference worker."""
    camera_frames: Dict[str, list]
    ego_history_xyz: torch.Tensor
    ego_history_rot: torch.Tensor
    submit_tick: int
    nav_text: Optional[str] = None


@dataclass
class _InferenceResult:
    """Payload returned from worker thread to main thread."""
    output: AlpamayoOutput
    submit_tick: int
    inference_time: float
    nav_text: Optional[str] = None


def _inference_worker(
    model: AlpamayoWrapper,
    config: AgentConfig,
    request_event: threading.Event,
    result_event: threading.Event,
    shared: dict,
    stop_event: threading.Event,
) -> None:
    """Background thread for Alpamayo inference."""
    while not stop_event.is_set():
        if not request_event.wait(timeout=0.1):
            continue
        request_event.clear()

        req: Optional[_InferenceRequest] = shared.get("request")
        if req is None:
            continue

        t0 = time.perf_counter()
        if config.use_dummy_model:
            output = model.predict_dummy()
        else:
            output = model.predict(
                camera_frames=req.camera_frames,
                ego_history_xyz=req.ego_history_xyz,
                ego_history_rot=req.ego_history_rot,
                max_generation_length=config.max_generation_length,
                diffusion_steps=config.diffusion_steps,
                nav_text=req.nav_text,
                use_cfg_nav=config.use_cfg_nav,
                cfg_nav_guidance_weight=config.cfg_nav_guidance_weight,
                use_camera_indices=config.use_camera_indices,
            )
        elapsed = time.perf_counter() - t0

        shared["result"] = _InferenceResult(
            output=output,
            submit_tick=req.submit_tick,
            inference_time=elapsed,
            nav_text=req.nav_text,
        )
        result_event.set()


# ---------------------------------------------------------------------------
# Main agent
# ---------------------------------------------------------------------------
class CarlaAlpamayoAgent:
    """
    Autonomous driving agent: Alpamayo-R1 VLA inside CARLA.

    Lifecycle::

        with CarlaAlpamayoAgent(config) as agent:
            agent.run(max_frames=500)
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()

        # CARLA handles
        self.client: Optional[carla.Client] = None
        self.world: Optional[carla.World] = None
        self.vehicle: Optional[carla.Actor] = None
        self.spectator: Optional[carla.Actor] = None
        self.traffic_manager = None

        # Sub-components
        self.sensor_manager: Optional[SensorManager] = None
        self.alpamayo: Optional[AlpamayoWrapper] = None
        self.follower: Optional[object] = None
        self.display: Optional[Display] = None
        self.traj_optimizer: Optional[TrajectoryOptimizer] = None
        self.nav_planner = None  # NavPlanner (Alpamayo 1.5 only)

        # Ego-history ring buffer (world-frame poses)
        self.ego_history: deque[EgoPose] = deque(maxlen=512)

        # Counters
        self.is_running = False
        self.inference_count = 0
        self.tick_count = 0
        self.last_output: Optional[AlpamayoOutput] = None
        self.last_inference_time: float = 0.0  # seconds (wall clock)
        self.last_result_tick: int = 0
        self.inference_busy: bool = False
        self._last_inference_submit_wall_time: float = -1e9
        self.last_opt_result: Optional[OptimizationResult] = None
        self.last_nav_text: Optional[str] = None
        self._debug_log_file = None
        self._debug_log_writer = None
        self._npc_vehicles: list = []
        self._npc_walkers: list = []       # walker actors
        self._npc_walker_ctrls: list = []  # AI controller actors

        # Inference threading (same pattern as observer)
        self._request_event = threading.Event()
        self._result_event = threading.Event()
        self._stop_event = threading.Event()
        self._shared: dict = {}
        self._worker_thread: Optional[threading.Thread] = None

    # ==================================================================
    # Initialisation helpers
    # ==================================================================

    def connect(self) -> None:
        print(f"Connecting to CARLA at {self.config.host}:{self.config.port}...")
        self.client = carla.Client(self.config.host, self.config.port)
        self.client.set_timeout(self.config.timeout)
        print(f"Connected to CARLA {self.client.get_server_version()}")

        # Load requested map if specified
        if self.config.map_name:
            current_map = self.client.get_world().get_map().name
            target = self.config.map_name
            # CARLA map names may include a path prefix (e.g. "Carla/Maps/Town03")
            if not current_map.endswith(target):
                print(f"Loading map {target} (current: {current_map})...")
                self.client.set_timeout(60.0)  # map load can be slow
                self.world = self.client.load_world(target)
                self.client.set_timeout(self.config.timeout)
            else:
                print(f"Map {target} already loaded.")
                self.world = self.client.get_world()
        else:
            self.world = self.client.get_world()

        self.spectator = self.world.get_spectator()
        print(f"Map: {self.world.get_map().name}")

        # Apply weather
        self._apply_weather(self.config.weather)

    # Supported weather presets (CARLA built-in)
    WEATHER_PRESETS = {
        "ClearNoon":      carla.WeatherParameters.ClearNoon,
        "CloudyNoon":     carla.WeatherParameters.CloudyNoon,
        "WetNoon":        carla.WeatherParameters.WetNoon,
        "WetCloudyNoon":  carla.WeatherParameters.WetCloudyNoon,
        "SoftRainNoon":   carla.WeatherParameters.SoftRainNoon,
        "MidRainyNoon":   carla.WeatherParameters.MidRainyNoon,
        "HardRainNoon":   carla.WeatherParameters.HardRainNoon,
        "ClearSunset":    carla.WeatherParameters.ClearSunset,
        "CloudySunset":   carla.WeatherParameters.CloudySunset,
        "WetSunset":      carla.WeatherParameters.WetSunset,
        "WetCloudySunset": carla.WeatherParameters.WetCloudySunset,
        "SoftRainSunset": carla.WeatherParameters.SoftRainSunset,
        "MidRainSunset":  carla.WeatherParameters.MidRainSunset,
        "HardRainSunset": carla.WeatherParameters.HardRainSunset,
    }

    def _apply_weather(self, preset_name: str) -> None:
        """Apply a weather preset and ensure dry roads (no puddles)."""
        if preset_name in self.WEATHER_PRESETS:
            weather = self.WEATHER_PRESETS[preset_name]
        else:
            available = ", ".join(sorted(self.WEATHER_PRESETS.keys()))
            raise ValueError(
                f"Unknown weather preset '{preset_name}'. "
                f"Available: {available}"
            )

        # Force-remove puddles / standing water regardless of preset.
        # precipitation_deposits controls wetness on the road surface
        # (0 = completely dry, 100 = fully wet).
        weather.precipitation_deposits = 0.0

        self.world.set_weather(weather)
        print(f"Weather: {preset_name} (roads dry, no puddles)")

    def spawn_vehicle(self) -> carla.Actor:
        import random

        bp_lib = self.world.get_blueprint_library()
        bps = bp_lib.filter(self.config.vehicle_filter)
        if not bps:
            bps = bp_lib.filter("vehicle.*")
        bp = bps[0]

        spawns = self.world.get_map().get_spawn_points()
        if not spawns:
            raise RuntimeError("No spawn points available")

        idx = self.config.spawn_point_index
        if idx < 0:
            # Random: try up to 10 different spawn points in case of collision
            random.shuffle(spawns)
            for sp in spawns[:10]:
                try:
                    self.vehicle = self.world.spawn_actor(bp, sp)
                    print(f"Spawned vehicle: {bp.id} at {sp.location} (random)")
                    return self.vehicle
                except RuntimeError:
                    continue
            raise RuntimeError("Could not find a free spawn point")
        else:
            idx = min(idx, len(spawns) - 1)
            sp = spawns[idx]
            self.vehicle = self.world.spawn_actor(bp, sp)
            print(f"Spawned vehicle: {bp.id} at {sp.location} (index={idx})")
            return self.vehicle

    @staticmethod
    def _parse_cam_resolution(value: str) -> Optional[tuple]:
        """Parse camera resolution string → (w, h) tuple or None for default."""
        if value == "full":
            return None  # Use default 1900×1080
        if value in RESOLUTION_PRESETS:
            return RESOLUTION_PRESETS[value]
        # Try "WxH" format
        if "x" in value:
            parts = value.split("x")
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                return (int(parts[0]), int(parts[1]))
        raise ValueError(
            f"Invalid cam_resolution '{value}'. "
            f"Use 'full', 'half', 'low', or 'WxH' (e.g. '640x360')."
        )

    def _spawn_npc_traffic(self) -> None:
        """Spawn NPC vehicles and pedestrians with autopilot / AI control."""
        import random

        cfg = self.config
        tm_port = 8000
        if self.traffic_manager is None:
            self.traffic_manager = self.client.get_trafficmanager(tm_port)
            self.traffic_manager.set_synchronous_mode(True)

        # ── NPC vehicles ──
        if cfg.num_npc_vehicles > 0:
            bp_lib = self.world.get_blueprint_library()
            vehicle_bps = bp_lib.filter("vehicle.*")
            # Exclude bicycles / motorcycles for stability
            vehicle_bps = [
                bp for bp in vehicle_bps
                if int(bp.get_attribute("number_of_wheels")) >= 4
            ]

            spawns = self.world.get_map().get_spawn_points()
            random.shuffle(spawns)

            count = 0
            for sp in spawns[:cfg.num_npc_vehicles + 10]:
                if count >= cfg.num_npc_vehicles:
                    break
                bp = random.choice(vehicle_bps)
                # Randomise color
                if bp.has_attribute("color"):
                    colors = bp.get_attribute("color").recommended_values
                    bp.set_attribute("color", random.choice(colors))
                bp.set_attribute("role_name", "npc")
                try:
                    npc = self.world.spawn_actor(bp, sp)
                    npc.set_autopilot(True, tm_port)
                    self._npc_vehicles.append(npc)
                    count += 1
                except RuntimeError:
                    continue

            print(f"Spawned {count} NPC vehicles")

        # ── NPC pedestrians (walkers) ──
        if cfg.num_npc_walkers > 0:
            bp_lib = self.world.get_blueprint_library()
            walker_bps = bp_lib.filter("walker.pedestrian.*")

            spawn_points = []
            for _ in range(cfg.num_npc_walkers + 20):
                loc = self.world.get_random_location_from_navigation()
                if loc is not None:
                    spawn_points.append(carla.Transform(location=loc))

            count = 0
            for sp in spawn_points[:cfg.num_npc_walkers + 10]:
                if count >= cfg.num_npc_walkers:
                    break
                bp = random.choice(walker_bps)
                if bp.has_attribute("is_invincible"):
                    bp.set_attribute("is_invincible", "false")
                try:
                    walker = self.world.spawn_actor(bp, sp)
                    self._npc_walkers.append(walker)
                    count += 1
                except RuntimeError:
                    continue

            # Spawn AI controllers for each walker
            ctrl_bp = self.world.get_blueprint_library().find("controller.ai.walker")
            for walker in self._npc_walkers:
                try:
                    ctrl = self.world.spawn_actor(ctrl_bp, carla.Transform(), walker)
                    self._npc_walker_ctrls.append(ctrl)
                except RuntimeError:
                    self._npc_walker_ctrls.append(None)

            # Need one tick for controllers to initialise
            self.world.tick()

            # Start walking to random destinations
            for ctrl in self._npc_walker_ctrls:
                if ctrl is None:
                    continue
                ctrl.start()
                dest = self.world.get_random_location_from_navigation()
                if dest is not None:
                    ctrl.go_to_location(dest)
                ctrl.set_max_speed(1.0 + random.random() * 1.5)  # 1.0–2.5 m/s

            print(f"Spawned {count} NPC walkers")

    def _destroy_npc_traffic(self) -> None:
        """Destroy all NPC actors (tolerant of already-dead actors)."""
        # Stop walker controllers first
        for ctrl in self._npc_walker_ctrls:
            try:
                if ctrl is not None and ctrl.is_alive:
                    ctrl.stop()
                    ctrl.destroy()
            except RuntimeError:
                pass
        self._npc_walker_ctrls.clear()

        for walker in self._npc_walkers:
            try:
                if walker is not None and walker.is_alive:
                    walker.destroy()
            except RuntimeError:
                pass
        self._npc_walkers.clear()

        for npc in self._npc_vehicles:
            try:
                if npc is not None and npc.is_alive:
                    npc.set_autopilot(False)
                    npc.destroy()
            except RuntimeError:
                pass
        self._npc_vehicles.clear()

    def setup_sensors(self) -> None:
        # subsample_factor: cameras fire every sim tick.
        # At 10 FPS each tick is 0.1 s, matching AR1 training data → factor=1.
        # If sim_fps is higher, subsample to maintain 0.1 s spacing.
        target_frame_dt = HISTORY_TIME_STEP          # 0.1 s (10 Hz)
        sim_dt = 1.0 / self.config.sim_fps
        subsample_factor = max(1, round(target_frame_dt / sim_dt))

        # Parse camera resolution
        cam_res = self._parse_cam_resolution(self.config.cam_resolution)

        self.sensor_manager = SensorManager(
            self.world,
            self.vehicle,
            subsample_factor=subsample_factor,
            cam_resolution=cam_res,
        )
        self.sensor_manager.spawn_default_sensors()

    def load_alpamayo(self) -> None:
        self.alpamayo = AlpamayoWrapper(
            model_name=self.config.model_name,
            num_traj_samples=self.config.num_traj_samples,
            top_p=self.config.vlm_top_p,
            temperature=self.config.vlm_temperature,
            timing_log=self.config.timing_log,
        )
        if self.config.use_dummy_model:
            print("Using dummy model for testing (no GPU required)")
        else:
            print("Loading Alpamayo model (this may take several minutes)...")
            self.alpamayo.load_model()

    def initialize(self) -> None:
        """Full initialisation sequence."""
        self.connect()
        self.spawn_vehicle()
        self.setup_sensors()
        self.load_alpamayo()

        control_mode = self.config.control_mode.replace("-", "_").lower()
        if control_mode == "official_pid":
            self.follower = OfficialPIDTrajectoryFollower(
                world=self.world,
                vehicle=self.vehicle,
                output_frequency_hz=self.config.sim_fps,
                max_speed_kmh=self.config.max_speed_kmh,
                min_speed_kmh=self.config.min_speed_kmh,
                lookahead_min_m=self.config.pid_lookahead_min_m,
                lookahead_max_m=self.config.pid_lookahead_max_m,
                lookahead_speed_gain=self.config.pid_lookahead_speed_gain,
                target_speed_min_kmh=self.config.pid_target_speed_min_kmh,
                target_speed_max_kmh=self.config.pid_target_speed_max_kmh,
                target_speed_extent_gain=self.config.pid_target_speed_extent_gain,
                lat_kp=self.config.pid_lat_kp,
                lat_ki=self.config.pid_lat_ki,
                lat_kd=self.config.pid_lat_kd,
                lon_kp=self.config.pid_lon_kp,
                lon_ki=self.config.pid_lon_ki,
                lon_kd=self.config.pid_lon_kd,
                max_throttle=self.config.pid_max_throttle,
                max_brake=self.config.pid_max_brake,
            )
            print("Control mode: official_pid (CARLA VehiclePIDController)")
        else:
            self.follower = TrajectoryFollower(
                lookahead_distance=self.config.lookahead_distance,
                max_speed_kmh=self.config.max_speed_kmh,
                min_speed_kmh=self.config.min_speed_kmh,
                steer_gain=self.config.steer_gain,
                steer_normalize_deg=self.config.steer_normalize_deg,
            )
            print("Control mode: legacy (TrajectoryFollower)")

        # Trajectory optimiser (optional post-processing)
        if self.config.traj_opt_enabled:
            self.traj_optimizer = TrajectoryOptimizer(
                smoothness_weight=self.config.traj_opt_smoothness_w,
                deviation_weight=self.config.traj_opt_deviation_w,
                comfort_weight=self.config.traj_opt_comfort_w,
                max_iterations=self.config.traj_opt_max_iter,
                enable_frenet_retiming=self.config.traj_opt_retime,
                retime_alpha=self.config.traj_opt_retime_alpha,
            )
            print("Trajectory optimiser enabled "
                  f"(smooth={self.config.traj_opt_smoothness_w}, "
                  f"dev={self.config.traj_opt_deviation_w}, "
                  f"comfort={self.config.traj_opt_comfort_w}, "
                  f"retime={self.config.traj_opt_retime})")
        else:
            self.traj_optimizer = None

        # Synchronous mode for deterministic ticking
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / self.config.sim_fps
        self.world.apply_settings(settings)

        # Spawn NPC traffic
        if self.config.num_npc_vehicles > 0 or self.config.num_npc_walkers > 0:
            self._spawn_npc_traffic()

        # Navigation planner (Alpamayo 1.5 only)
        if (
            self.config.nav_enabled
            and detect_model_version(self.config.model_name) == VERSION_15
        ):
            from .nav_planner_nxt import NavPlanner
            self.nav_planner = NavPlanner(self.world)
            start = self.vehicle.get_location()
            if self.config.nav_destination_index >= 0:
                spawns = self.world.get_map().get_spawn_points()
                idx = min(self.config.nav_destination_index, len(spawns) - 1)
                self.nav_planner.set_destination(start, spawns[idx].location)
            else:
                self.nav_planner.set_random_destination(start)
            print("Navigation planner enabled (Alpamayo 1.5)")

        # Start inference worker thread
        self._stop_event.clear()
        self._request_event.clear()
        self._result_event.clear()
        self._shared.clear()
        self.inference_busy = False
        self._worker_thread = threading.Thread(
            target=_inference_worker,
            args=(
                self.alpamayo,
                self.config,
                self._request_event,
                self._result_event,
                self._shared,
                self._stop_event,
            ),
            daemon=True,
        )
        self._worker_thread.start()
        if self.config.use_async_inference:
            print("Inference execution: async (worker thread, non-blocking)")
        else:
            print("Inference execution: sync wait (worker thread, blocking)")

        # Pygame display (optional)
        if self.config.enable_display:
            self.display = Display(
                record_path=self.config.record_path,
                record_fps=self.config.sim_fps,
                record_crf=self.config.record_crf,
                camera_downsample=self.config.display_camera_downsample,
                max_cameras=self.config.display_max_cameras,
            )

        print("Agent initialised successfully!")

    # ==================================================================
    # Ego history
    # ==================================================================

    def _record_ego_pose(self) -> None:
        """Snapshot current vehicle pose into the history buffer.

        CARLA's ``get_transform()`` returns the bounding-box center, but
        Alpamayo expects the rig origin (rear axle).  We apply the fixed
        offset ``_REAR_AXLE_OFFSET_LOCAL`` to convert.
        """
        transform = self.vehicle.get_transform()
        snap = self.world.get_snapshot()
        timestamp_us = int(snap.timestamp.elapsed_seconds * 1_000_000)

        loc = transform.location
        center_world = np.array([loc.x, loc.y, loc.z], dtype=np.float64)

        rot = transform.rotation
        # CARLA intrinsic ZYX Euler: yaw → pitch → roll (all in degrees)
        r = Rotation.from_euler(
            "ZYX", [rot.yaw, rot.pitch, rot.roll], degrees=True
        )
        rotation_matrix = r.as_matrix()

        # Convert bounding-box center → rear axle (rig origin) in world
        rear_axle_world = center_world + rotation_matrix @ _REAR_AXLE_OFFSET_LOCAL

        self.ego_history.append(
            EgoPose(
                timestamp_us=timestamp_us,
                location=rear_axle_world,
                rotation_matrix=rotation_matrix,
                rotation_quat=r.as_quat(),
            )
        )

    def _build_ego_history_tensors(
        self,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """
        Build ego-history tensors required by Alpamayo-R1.

        Returns ``None`` when there is not enough history yet.

        Returns:
            (ego_history_xyz, ego_history_rot) each float32 with shape
            ``(1, 1, 16, 3)`` and ``(1, 1, 16, 3, 3)`` respectively,
            in **rig frame** relative to the current pose.
        """
        # How many ticks correspond to HISTORY_TIME_STEP?
        step_interval = max(1, round(HISTORY_TIME_STEP * self.config.sim_fps))
        required_len = (NUM_HISTORY_STEPS - 1) * step_interval + 1

        if len(self.ego_history) < required_len:
            return None

        current = self.ego_history[-1]
        cur_rot = Rotation.from_quat(current.rotation_quat)
        cur_rot_inv = cur_rot.inv()
        R_cur_inv = current.rotation_matrix.T           # inverse rotation (for xyz transform)

        # Flip matrix: CARLA (Y right) → rig (Y left)
        F = np.diag([1.0, -1.0, 1.0])

        history_xyz = []
        history_rot = []

        for i in range(NUM_HISTORY_STEPS - 1, -1, -1):
            buf_idx = len(self.ego_history) - 1 - i * step_interval
            buf_idx = max(buf_idx, 0)
            pose = self.ego_history[buf_idx]

            # Position: world → current-vehicle-local → rig
            delta_world = pose.location - current.location
            delta_local = R_cur_inv @ delta_world
            delta_rig = F @ delta_local
            history_xyz.append(delta_rig)

            # Rotation: match alpamayo1.5 loader logic:
            # (R_t0^-1 * R_t).as_matrix(), then CARLA(Y-right)->rig(Y-left) flip.
            pose_rot = Rotation.from_quat(pose.rotation_quat)
            R_rel = (cur_rot_inv * pose_rot).as_matrix()
            R_rel_rig = F @ R_rel @ F
            history_rot.append(R_rel_rig)

        # (16, 3) → (1, 1, 16, 3)
        ego_xyz = (
            torch.from_numpy(np.array(history_xyz, dtype=np.float32))
            .unsqueeze(0)
            .unsqueeze(0)
        )
        # (16, 3, 3) → (1, 1, 16, 3, 3)
        ego_rot = (
            torch.from_numpy(np.array(history_rot, dtype=np.float32))
            .unsqueeze(0)
            .unsqueeze(0)
        )
        return ego_xyz, ego_rot

    # ==================================================================
    # Image preparation
    # ==================================================================

    def _prepare_camera_frames(self) -> Optional[Dict[str, list]]:
        """
        Collect the latest ``context_length`` frames from **all 4 cameras**.

        Returns:
            Dict mapping camera_name → list of HWC uint8 np.ndarray images
            (``context_length`` entries each), or ``None`` if any camera
            does not have enough frames yet.
        """
        all_cam_frames = self.sensor_manager.get_all_camera_frames(
            count=self.config.context_length
        )
        if all_cam_frames is None:
            return None

        # Convert TimestampedFrame objects → raw HWC images
        result: Dict[str, list] = {}
        for cam_name in ALPAMAYO_CAMERA_ORDER:
            result[cam_name] = [f.image for f in all_cam_frames[cam_name]]
        return result

    def _get_latest_camera_images(self) -> Dict[str, np.ndarray]:
        """Get the most recent single image from each camera (for display)."""
        images: Dict[str, np.ndarray] = {}
        for cam_name in ALPAMAYO_CAMERA_ORDER:
            frames = self.sensor_manager.get_buffered_frames(cam_name, count=1)
            if frames:
                images[cam_name] = frames[-1].image
        return images

    # ==================================================================
    # Vehicle state
    # ==================================================================

    def get_vehicle_state(self) -> Dict:
        vel = self.vehicle.get_velocity()
        tr = self.vehicle.get_transform()
        speed_ms = np.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
        return {
            "speed_kmh": speed_ms * 3.6,
            "speed_ms": speed_ms,
            "location": {"x": tr.location.x, "y": tr.location.y, "z": tr.location.z},
            "rotation": {
                "pitch": tr.rotation.pitch,
                "yaw": tr.rotation.yaw,
                "roll": tr.rotation.roll,
            },
        }

    # ==================================================================
    # Step / run loop
    # ==================================================================

    def _get_nav_text(self) -> Optional[str]:
        """Get the current navigation instruction from the planner."""
        if self.nav_planner is None:
            return None
        nav_text = self.nav_planner.get_instruction(self.vehicle.get_transform())
        if self.nav_planner.route_complete:
            self.nav_planner.set_random_destination(self.vehicle.get_location())
        return nav_text

    def _should_run_inference(self) -> bool:
        """Check whether we should run inference this tick."""
        sec_interval = self.config.inference_interval_sec
        if sec_interval is not None and sec_interval > 0.0:
            now = time.monotonic()
            return (now - self._last_inference_submit_wall_time) >= sec_interval
        return (self.tick_count % self.config.inference_interval) == 0

    def _apply_inference_result(self, output: AlpamayoOutput) -> None:
        """Apply inference result to active trajectory/control state."""
        self.last_output = output

        # ── Trajectory optimisation (optional) ──
        traj_xy = output.trajectory_xy
        headings = output.headings

        if (
            self.traj_optimizer is not None
            and traj_xy is not None
            and len(traj_xy) >= 2
        ):
            traj_xyh = add_heading_to_trajectory(traj_xy)
            opt_result = self.traj_optimizer.optimize(
                trajectory=traj_xyh,
                time_step=1.0 / 10.0,  # AR1 output at 10 Hz
                retime_in_frenet=self.config.traj_opt_retime,
                retime_alpha=self.config.traj_opt_retime_alpha,
            )
            self.last_opt_result = opt_result
            if opt_result.success:
                traj_xy = opt_result.trajectory[:, :2]
                headings = opt_result.trajectory[:, 2]
        else:
            self.last_opt_result = None

        self.follower.set_trajectory(traj_xy, headings)
        self.inference_count += 1

    def step(self) -> Optional[AlpamayoOutput]:
        """Execute one simulation tick; optionally run inference."""
        self._record_ego_pose()
        state = self.get_vehicle_state()
        self.tick_count += 1

        if self.config.use_async_inference:
            # --- consume completed inference result ---
            if self._result_event.is_set():
                self._result_event.clear()
                result: Optional[_InferenceResult] = self._shared.get("result")
                if result is not None:
                    self.last_inference_time = result.inference_time
                    self.last_result_tick = result.submit_tick
                    self.last_nav_text = result.nav_text
                    self.inference_busy = False
                    self._apply_inference_result(result.output)

            # --- submit inference request if idle ---
            if not self.inference_busy and self._should_run_inference():
                cam_frames = self._prepare_camera_frames()
                hist = self._build_ego_history_tensors()
                if cam_frames is not None and hist is not None:
                    ego_xyz, ego_rot = hist
                    nav_text = self.config.nav_text_override
                    if nav_text is None:
                        nav_text = self._get_nav_text()

                    self._shared["request"] = _InferenceRequest(
                        camera_frames=cam_frames,
                        ego_history_xyz=ego_xyz,
                        ego_history_rot=ego_rot,
                        submit_tick=self.tick_count,
                        nav_text=nav_text,
                    )
                    self._request_event.set()
                    self.inference_busy = True
                    self.last_nav_text = nav_text
                    self._last_inference_submit_wall_time = time.monotonic()
        else:
            # --- synchronous wait mode ---
            # In this mode inference still runs on worker thread, but
            # the main thread blocks until a result is ready.
            if not self.inference_busy and self._should_run_inference():
                cam_frames = self._prepare_camera_frames()
                hist = self._build_ego_history_tensors()
                if cam_frames is not None and hist is not None:
                    ego_xyz, ego_rot = hist
                    nav_text = self.config.nav_text_override
                    if nav_text is None:
                        nav_text = self._get_nav_text()
                    self.last_nav_text = nav_text
                    self._shared["request"] = _InferenceRequest(
                        camera_frames=cam_frames,
                        ego_history_xyz=ego_xyz,
                        ego_history_rot=ego_rot,
                        submit_tick=self.tick_count,
                        nav_text=nav_text,
                    )
                    self._request_event.set()
                    self.inference_busy = True
                    self._last_inference_submit_wall_time = time.monotonic()

            if self.inference_busy:
                got_result = self._result_event.wait(timeout=60.0)
                if got_result:
                    self._result_event.clear()
                    result: Optional[_InferenceResult] = self._shared.get("result")
                    if result is not None:
                        self.last_inference_time = result.inference_time
                        self.last_result_tick = result.submit_tick
                        self.last_nav_text = result.nav_text
                        self._apply_inference_result(result.output)
                    self.inference_busy = False
                else:
                    print("Warning: sync inference wait timed out; continuing.")
                    self.inference_busy = False

        # --- control ---
        control = self.follower.compute_control(state["speed_ms"])
        self.vehicle.apply_control(control)

        return self.last_output

    def update_spectator(self) -> None:
        """Move the CARLA spectator to follow the vehicle from behind."""
        if not (self.vehicle and self.spectator):
            return
        tr = self.vehicle.get_transform()
        yaw_rad = np.radians(tr.rotation.yaw)
        self.spectator.set_transform(
            carla.Transform(
                carla.Location(
                    x=tr.location.x - 8 * np.cos(yaw_rad),
                    y=tr.location.y - 8 * np.sin(yaw_rad),
                    z=tr.location.z + 4,
                ),
                carla.Rotation(pitch=-15, yaw=tr.rotation.yaw),
            )
        )

    def run(
        self,
        max_frames: int = 1000,
        verbose: bool = True,
    ) -> None:
        """
        Run the autonomous-driving main loop.

        Args:
            max_frames: Stop after this many **inference** steps.
            verbose: Print periodic status lines.
        """
        cfg = self.config
        if cfg.inference_interval_sec is not None and cfg.inference_interval_sec > 0.0:
            infer_hz = 1.0 / cfg.inference_interval_sec
            infer_mode = f"time-based {cfg.inference_interval_sec:.2f}s"
        else:
            infer_hz = cfg.sim_fps / cfg.inference_interval
            infer_mode = f"tick-based every {cfg.inference_interval} tick(s)"
        print(f"Starting autonomous driving  (inference ≈{infer_hz:.1f} Hz, {infer_mode})")
        print("Press Ctrl+C to stop\n")

        # --- warm-up: collect enough sensor data & ego history ---
        step_interval = max(1, round(HISTORY_TIME_STEP * cfg.sim_fps))
        subsample_factor = self.sensor_manager.subsample_factor
        # Need enough frames for subsampled context: (ctx-1)*subsample + 1
        min_cam_ticks = (cfg.context_length - 1) * subsample_factor + 2
        warmup_ticks = max(
            min_cam_ticks,
            (NUM_HISTORY_STEPS - 1) * step_interval + 4,
        )
        print(f"Warming up for {warmup_ticks} ticks "
              f"(subsample_factor={subsample_factor}) "
              f"to collect sensor data & ego history...")
        for _ in range(warmup_ticks):
            self.world.tick()
            self._record_ego_pose()
            time.sleep(0.01)
        print("Warm-up complete – entering inference loop.\n")

        # ── Debug CSV log ──
        if cfg.debug_log_path:
            import csv
            self._debug_log_file = open(cfg.debug_log_path, "w", newline="")
            self._debug_log_writer = csv.writer(self._debug_log_file)
            self._debug_log_writer.writerow([
                "tick", "inf_count", "inf_time_s",
                "speed_kmh", "speed_ms",
                # Trajectory summary
                "traj_len_m", "traj_nwp", "explicit_stop",
                "traj_wp0_x", "traj_wp0_y",
                "traj_wp5_x", "traj_wp5_y",
                "traj_wp15_x", "traj_wp15_y",
                "traj_wp30_x", "traj_wp30_y",
                # Speed estimation
                "raw_desired_spd", "ema_desired_spd", "curve_safe_spd",
                "min_speed_ms", "max_speed_ms",
                # Steering
                "pp_steer", "hdg_steer", "hdg_weight",
                "raw_steer", "ema_steer",
                # Control output
                "throttle", "brake", "steer_out",
                # Stop intent
                "standstill_ticks", "short_traj_count", "model_wants_stop",
                # Navigation
                "nav_text",
                # CoT
                "cot",
            ])
            print(f"Debug log: {cfg.debug_log_path}")

        self.is_running = True
        tick = 0

        try:
            while self.is_running and self.inference_count < max_frames:
                loop_t0 = time.perf_counter()
                t0 = time.perf_counter()
                self.world.tick()
                world_tick_ms = (time.perf_counter() - t0) * 1000.0

                t0 = time.perf_counter()
                output = self.step()
                step_ms = (time.perf_counter() - t0) * 1000.0

                t0 = time.perf_counter()
                self.update_spectator()
                spectator_ms = (time.perf_counter() - t0) * 1000.0

                # ── pygame display ──
                display_ms = 0.0
                display_fetch_ms = 0.0
                display_render_ms = 0.0
                if self.display is not None:
                    t0 = time.perf_counter()
                    state = self.get_vehicle_state()
                    cam_images = None
                    if cfg.display_fetch_cameras:
                        tf = time.perf_counter()
                        cam_images = self._get_latest_camera_images()
                        display_fetch_ms = (time.perf_counter() - tf) * 1000.0
                    # The follower holds the (possibly optimised) trajectory
                    active_traj = self.follower.trajectory
                    # Raw model output for comparison
                    raw_traj = output.trajectory_xy if output else None
                    reasoning = output.reasoning if output else None
                    meta_act = output.meta_action if output else None
                    all_traj = output.all_trajectories_xy if output else None
                    sel_idx = output.selected_index if output else 0
                    tr = time.perf_counter()
                    self.display.tick(
                        camera_images=cam_images,
                        vehicle_state=state,
                        trajectory_xy=active_traj,
                        reasoning=reasoning,
                        inference_count=self.inference_count,
                        tick_count=self.tick_count,
                        inference_time=self.last_inference_time,
                        all_trajectories_xy=all_traj,
                        selected_traj_index=sel_idx,
                        raw_trajectory_xy=raw_traj if self.traj_optimizer else None,
                        nav_text=self.last_nav_text,
                        meta_action=meta_act,
                    )
                    display_render_ms = (time.perf_counter() - tr) * 1000.0
                    if self.display.should_quit:
                        print("Display closed – stopping.")
                        break
                    display_ms = (time.perf_counter() - t0) * 1000.0
                loop_ms = (time.perf_counter() - loop_t0) * 1000.0
                if cfg.loop_timing_log:
                    print(
                        "[loop_timing] "
                        f"world_tick={world_tick_ms:.1f}ms "
                        f"step={step_ms:.1f}ms "
                        f"spectator={spectator_ms:.1f}ms "
                        f"display={display_ms:.1f}ms "
                        f"display_fetch={display_fetch_ms:.1f}ms "
                        f"display_render={display_render_ms:.1f}ms "
                        f"loop_total={loop_ms:.1f}ms"
                    )

                # ── console log ──
                if verbose:
                    st = self.get_vehicle_state()
                    # Use _last_control (set inside compute_control) to avoid
                    # the 1-tick lag of vehicle.get_control().
                    ctrl = self.follower._last_control
                    if ctrl is None:
                        ctrl = carla.VehicleControl()
                    tl = self.follower._trajectory_length() if self.follower.trajectory is not None else 0.0
                    ds = self.follower._last_desired_speed      # EMA smoothed
                    ds_raw = self.follower._raw_desired_speed    # instant (before EMA)
                    cot = ""
                    if output and output.meta_action:
                        cot = f"[{output.meta_action.strip()[:30]}] "
                    if output and output.reasoning:
                        cot += output.reasoning[:50].replace("\n", " ") + "…"
                    opt_tag = ""
                    if self.last_opt_result is not None:
                        r = self.last_opt_result
                        opt_tag = f" Opt:{'OK' if r.success else 'FAIL'}"
                    nudge_tag = ""
                    stall = self.follower._standstill_ticks
                    if stall >= self.follower.STANDSTILL_NUDGE_TICKS:
                        nudge_tag = " NUDGE"
                    elif stall > 0 and st["speed_kmh"] < 1.0:
                        nudge_tag = f" stall:{stall}"
                    steer_raw = self.follower._raw_steer
                    steer_ema = self.follower._last_steer
                    # Majority-vote info from last inference
                    vote_tag = ""
                    if output and output.all_trajectories_xy is not None:
                        from .alpamayo_wrapper_nxt import AlpamayoWrapper
                        n_go = sum(
                            1 for t in output.all_trajectories_xy
                            if AlpamayoWrapper._traj_length(t) >= AlpamayoWrapper._GO_THRESHOLD
                        )
                        n_tot = len(output.all_trajectories_xy)
                        vote_tag = f" go:{n_go}/{n_tot}"
                    curv_spd = self.follower._curve_safe_speed
                    curv_tag = f" curv:{curv_spd:.1f}m/s" if curv_spd < 90 else ""
                    hw = self.follower._hdg_weight
                    hdg_tag = f" hdg:{hw:.0%}" if hw > 0.05 else ""
                    nav_tag = ""
                    if self.last_nav_text:
                        nav_tag = f" Nav:[{self.last_nav_text}]"
                    print(
                        f"Tick {tick:5d} | Inf #{self.inference_count:4d} "
                        f"({self.last_inference_time:.2f}s) | "
                        f"Spd {st['speed_kmh']:5.1f} | "
                        f"Thr {ctrl.throttle:.2f} Brk {ctrl.brake:.2f} Str {steer_ema:+.2f} | "
                        f"Raw {ds_raw:.2f} EMA {ds:.2f}m/s TrjL {tl:.1f}m{vote_tag}{curv_tag}{hdg_tag}{opt_tag}{nudge_tag}{nav_tag} | {cot}"
                    )

                # ── Debug CSV ──
                if self._debug_log_writer is not None:
                    f = self.follower
                    traj = f.trajectory
                    nwp = len(traj) if traj is not None else 0

                    def _wp(idx):
                        if traj is not None and idx < nwp:
                            return f"{traj[idx][0]:.3f}", f"{traj[idx][1]:.3f}"
                        return "", ""

                    st2 = self.get_vehicle_state()
                    ctrl2 = f._last_control or carla.VehicleControl()
                    cot_text = ""
                    if output and output.reasoning:
                        cot_text = output.reasoning[:80].replace("\n", " ")

                    self._debug_log_writer.writerow([
                        tick, self.inference_count, f"{self.last_inference_time:.3f}",
                        f"{st2['speed_kmh']:.2f}", f"{st2['speed_ms']:.3f}",
                        f"{f._trajectory_length() if traj is not None else 0:.2f}",
                        nwp, int(f._explicit_stop),
                        *_wp(0), *_wp(5), *_wp(15), *_wp(30),
                        f"{f._raw_desired_speed:.3f}", f"{f._last_desired_speed:.3f}",
                        f"{f._curve_safe_speed:.2f}",
                        f"{f.min_speed_ms:.2f}", f"{f.max_speed_ms:.2f}",
                        f"{f._pp_steer:.4f}", f"{f._hdg_steer:.4f}", f"{f._hdg_weight:.3f}",
                        f"{f._raw_steer:.4f}", f"{f._last_steer:.4f}",
                        f"{ctrl2.throttle:.3f}", f"{ctrl2.brake:.3f}", f"{ctrl2.steer:.4f}",
                        f._standstill_ticks, f._short_traj_count,
                        int(f._short_traj_count >= 3),
                        self.last_nav_text or "",
                        cot_text,
                    ])
                    self._debug_log_file.flush()

                tick += 1

        except KeyboardInterrupt:
            print("\nStopping…")
        finally:
            self.is_running = False

    # ==================================================================
    # Cleanup / context manager
    # ==================================================================

    def cleanup(self) -> None:
        print("Cleaning up…")
        self._stop_event.set()
        self._request_event.set()
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=2.0)
            self._worker_thread = None
        self._destroy_npc_traffic()
        if self._debug_log_file:
            self._debug_log_file.close()
            self._debug_log_file = None
            self._debug_log_writer = None
            print("Debug log closed")
        if self.display:
            self.display.close()
            self.display = None
        if self.sensor_manager:
            self.sensor_manager.destroy_all()
        if self.vehicle and self.vehicle.is_alive:
            self.vehicle.destroy()
            print("Destroyed vehicle")
        if self.world:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)
        if self.traffic_manager:
            try:
                self.traffic_manager.set_synchronous_mode(False)
            except RuntimeError:
                pass
            self.traffic_manager = None
        print("Cleanup complete")

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False

