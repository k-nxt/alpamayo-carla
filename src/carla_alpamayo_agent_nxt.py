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
from typing import Dict, Optional
from dataclasses import dataclass
from collections import deque
from scipy.spatial.transform import Rotation

from .alpamayo_wrapper_nxt import AlpamayoWrapper, AlpamayoOutput
from .sensor_manager_nxt import SensorManager, ALPAMAYO_CAMERA_ORDER
from .display_nxt import Display

# ---------------------------------------------------------------------------
# Constants (must match Alpamayo-R1 expectations)
# ---------------------------------------------------------------------------
NUM_HISTORY_STEPS = 16    # ego history length
HISTORY_TIME_STEP = 0.1   # seconds between history samples


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class EgoPose:
    """Vehicle pose snapshot in CARLA world coordinates."""
    timestamp_us: int
    location: np.ndarray          # (3,) world x y z
    rotation_matrix: np.ndarray   # (3, 3) world rotation


@dataclass
class AgentConfig:
    """Configuration for the CARLA Alpamayo agent."""
    # CARLA connection
    host: str = "localhost"
    port: int = 2000
    timeout: float = 10.0

    # Vehicle
    vehicle_filter: str = "vehicle.tesla.model3"
    spawn_point_index: int = 0

    # Alpamayo model
    model_name: str = "nvidia/Alpamayo-R1-10B"
    use_dummy_model: bool = False
    context_length: int = 4       # temporal image frames
    num_traj_samples: int = 1

    # Simulation
    sim_fps: float = 10.0         # 10 Hz = 0.1 s/tick, matching AR1 training data
    inference_interval: int = 1   # run inference every tick (= every 0.1 s)

    # Control
    max_speed_kmh: float = 30.0
    lookahead_distance: float = 5.0   # pure-pursuit look-ahead (m)

    # Display
    enable_display: bool = True       # pygame dashboard window


# ---------------------------------------------------------------------------
# Trajectory → vehicle-control converter
# ---------------------------------------------------------------------------
class TrajectoryFollower:
    """
    Pure-Pursuit + proportional speed controller.

    Converts Alpamayo's predicted trajectory (rig frame waypoints)
    into a CARLA ``VehicleControl``.
    """

    def __init__(
        self,
        lookahead_distance: float = 5.0,
        max_speed_kmh: float = 30.0,
        wheelbase: float = 2.875,          # Tesla Model 3 approx
        output_frequency_hz: float = 10.0,  # Alpamayo outputs at 10 Hz
    ):
        self.lookahead_distance = lookahead_distance
        self.max_speed_ms = max_speed_kmh / 3.6
        self.wheelbase = wheelbase
        self.dt = 1.0 / output_frequency_hz

        self.trajectory: Optional[np.ndarray] = None
        self.headings: Optional[np.ndarray] = None

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
            return carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)

        traj = self.trajectory
        traj_length = self._trajectory_length()

        # ── Steering (pure pursuit) ──
        lookahead = self._find_lookahead_point()
        if lookahead is None:
            return carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.5)

        dx, dy = float(lookahead[0]), float(lookahead[1])
        dist = np.hypot(dx, dy)

        if dist < 0.1:
            steer = 0.0
        else:
            curvature = 2.0 * dy / (dist ** 2)
            steer_rad = np.arctan(curvature * self.wheelbase)
            steer = float(np.clip(steer_rad / np.radians(70), -1.0, 1.0))
            # Rig Y-left positive → CARLA steer-right positive ⇒ flip
            steer = -steer

        # ── Speed target ──
        # Use speed profile along the full trajectory, not just the first 1 s.
        desired_speed = min(self._estimate_desired_speed(), self.max_speed_ms)

        # If the trajectory is very short the model wants us to stop soon.
        # Ramp desired speed down proportionally to remaining trajectory length.
        stop_margin = self.lookahead_distance * 1.5  # begin slowing within this
        if traj_length < stop_margin:
            ratio = max(traj_length / stop_margin, 0.0)
            desired_speed *= ratio

        # ── Throttle / brake ──
        err = desired_speed - current_speed_ms

        if desired_speed < 0.3:
            # Model wants a near-stop → brake firmly
            throttle = 0.0
            brake = float(np.clip(0.5 + current_speed_ms * 0.1, 0.3, 1.0))
        elif err > 0.5:
            throttle = float(np.clip(err * 0.3, 0.0, 0.8))
            brake = 0.0
        elif err < -0.5:
            throttle = 0.0
            brake = float(np.clip(-err * 0.3, 0.1, 1.0))
        else:
            # Cruise – proportional to desired speed so we don't push
            # through a stop with constant throttle.
            throttle = float(np.clip(desired_speed * 0.06, 0.05, 0.3))
            brake = 0.0

        return carla.VehicleControl(
            throttle=throttle,
            steer=steer,
            brake=brake,
        )

    # -- helpers --

    def _trajectory_length(self) -> float:
        """Total arc length of the stored trajectory (meters)."""
        traj = self.trajectory
        if traj is None or len(traj) < 2:
            return 0.0
        diffs = np.diff(traj, axis=0)
        return float(np.sum(np.linalg.norm(diffs, axis=1)))

    def _find_lookahead_point(self) -> Optional[np.ndarray]:
        """Interpolate to find the point at ``lookahead_distance`` along the trajectory."""
        traj = self.trajectory
        if traj is None:
            return None

        diffs = np.diff(traj, axis=0)
        seg_len = np.linalg.norm(diffs, axis=1)
        cum_dist = np.concatenate([[0.0], np.cumsum(seg_len)])

        if cum_dist[-1] < self.lookahead_distance:
            return traj[-1]

        idx = int(np.searchsorted(cum_dist, self.lookahead_distance))
        if idx >= len(traj):
            return traj[-1]
        if idx == 0:
            return traj[0]

        frac = (self.lookahead_distance - cum_dist[idx - 1]) / (
            cum_dist[idx] - cum_dist[idx - 1] + 1e-6
        )
        return traj[idx - 1] + frac * (traj[idx] - traj[idx - 1])

    def _estimate_desired_speed(self) -> float:
        """
        Estimate desired speed from the trajectory's speed profile.

        Uses the speed near the lookahead region rather than just the
        first second, so deceleration commands are respected.
        """
        traj = self.trajectory
        if traj is None or len(traj) < 2:
            return 0.0

        diffs = np.diff(traj, axis=0)
        seg_speeds = np.linalg.norm(diffs, axis=1) / self.dt  # per-segment speed

        # Take the minimum of early speed and mid-trajectory speed.
        # This ensures we respond to deceleration further along.
        n = len(seg_speeds)
        early = float(np.mean(seg_speeds[: min(5, n)]))          # first 0.5 s
        mid   = float(np.mean(seg_speeds[min(5, n): min(20, n)])) if n > 5 else early
        late  = float(np.mean(seg_speeds[min(20, n):])) if n > 20 else mid

        # Use the minimum across the profile – if any section is slow,
        # we should already be decelerating.
        return min(early, mid, late)


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

        # Sub-components
        self.sensor_manager: Optional[SensorManager] = None
        self.alpamayo: Optional[AlpamayoWrapper] = None
        self.follower: Optional[TrajectoryFollower] = None
        self.display: Optional[Display] = None

        # Ego-history ring buffer (world-frame poses)
        self.ego_history: deque[EgoPose] = deque(maxlen=512)

        # Counters
        self.is_running = False
        self.inference_count = 0
        self.tick_count = 0
        self.last_output: Optional[AlpamayoOutput] = None

    # ==================================================================
    # Initialisation helpers
    # ==================================================================

    def connect(self) -> None:
        print(f"Connecting to CARLA at {self.config.host}:{self.config.port}...")
        self.client = carla.Client(self.config.host, self.config.port)
        self.client.set_timeout(self.config.timeout)
        self.world = self.client.get_world()
        self.spectator = self.world.get_spectator()
        print(f"Connected to CARLA {self.client.get_server_version()}")
        print(f"Map: {self.world.get_map().name}")

    def spawn_vehicle(self) -> carla.Actor:
        bp_lib = self.world.get_blueprint_library()
        bps = bp_lib.filter(self.config.vehicle_filter)
        if not bps:
            bps = bp_lib.filter("vehicle.*")
        bp = bps[0]

        spawns = self.world.get_map().get_spawn_points()
        if not spawns:
            raise RuntimeError("No spawn points available")
        idx = min(self.config.spawn_point_index, len(spawns) - 1)
        sp = spawns[idx]

        self.vehicle = self.world.spawn_actor(bp, sp)
        print(f"Spawned vehicle: {bp.id} at {sp.location}")
        return self.vehicle

    def setup_sensors(self) -> None:
        # subsample_factor: cameras fire every sim tick.
        # At 10 FPS each tick is 0.1 s, matching AR1 training data → factor=1.
        # If sim_fps is higher, subsample to maintain 0.1 s spacing.
        target_frame_dt = HISTORY_TIME_STEP          # 0.1 s (10 Hz)
        sim_dt = 1.0 / self.config.sim_fps
        subsample_factor = max(1, round(target_frame_dt / sim_dt))

        self.sensor_manager = SensorManager(
            self.world,
            self.vehicle,
            subsample_factor=subsample_factor,
        )
        self.sensor_manager.spawn_default_sensors()

    def load_alpamayo(self) -> None:
        self.alpamayo = AlpamayoWrapper(
            model_name=self.config.model_name,
            num_traj_samples=self.config.num_traj_samples,
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

        self.follower = TrajectoryFollower(
            lookahead_distance=self.config.lookahead_distance,
            max_speed_kmh=self.config.max_speed_kmh,
        )

        # Synchronous mode for deterministic ticking
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / self.config.sim_fps
        self.world.apply_settings(settings)

        # Pygame display (optional)
        if self.config.enable_display:
            self.display = Display()

        print("Agent initialised successfully!")

    # ==================================================================
    # Ego history
    # ==================================================================

    def _record_ego_pose(self) -> None:
        """Snapshot current vehicle pose into the history buffer."""
        transform = self.vehicle.get_transform()
        snap = self.world.get_snapshot()
        timestamp_us = int(snap.timestamp.elapsed_seconds * 1_000_000)

        loc = transform.location
        location = np.array([loc.x, loc.y, loc.z], dtype=np.float64)

        rot = transform.rotation
        # CARLA intrinsic ZYX Euler: yaw → pitch → roll (all in degrees)
        r = Rotation.from_euler(
            "ZYX", [rot.yaw, rot.pitch, rot.roll], degrees=True
        )
        rotation_matrix = r.as_matrix()

        self.ego_history.append(
            EgoPose(
                timestamp_us=timestamp_us,
                location=location,
                rotation_matrix=rotation_matrix,
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
        R_cur_inv = current.rotation_matrix.T           # inverse rotation

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

            # Rotation: relative to current, then flip to rig
            R_rel = R_cur_inv @ pose.rotation_matrix
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

    def step(self) -> Optional[AlpamayoOutput]:
        """Execute one simulation tick; optionally run inference."""
        self._record_ego_pose()
        state = self.get_vehicle_state()
        self.tick_count += 1

        # --- periodic inference ---
        if self.tick_count % self.config.inference_interval == 0:
            cam_frames = self._prepare_camera_frames()
            hist = self._build_ego_history_tensors()

            if cam_frames is not None and hist is not None:
                ego_xyz, ego_rot = hist

                if self.config.use_dummy_model:
                    output = self.alpamayo.predict_dummy()
                else:
                    output = self.alpamayo.predict(
                        camera_frames=cam_frames,
                        ego_history_xyz=ego_xyz,
                        ego_history_rot=ego_rot,
                    )

                self.last_output = output
                self.follower.set_trajectory(
                    output.trajectory_xy, output.headings
                )
                self.inference_count += 1

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
        infer_hz = cfg.sim_fps / cfg.inference_interval
        print(f"Starting autonomous driving  (inference ≈{infer_hz:.1f} Hz)")
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

        self.is_running = True
        tick = 0

        try:
            while self.is_running and self.inference_count < max_frames:
                self.world.tick()
                output = self.step()
                self.update_spectator()

                # ── pygame display ──
                if self.display is not None:
                    state = self.get_vehicle_state()
                    traj = output.trajectory_xy if output else None
                    reasoning = output.reasoning if output else None
                    self.display.tick(
                        camera_images=self._get_latest_camera_images(),
                        vehicle_state=state,
                        trajectory_xy=traj,
                        reasoning=reasoning,
                        inference_count=self.inference_count,
                        tick_count=self.tick_count,
                    )
                    if self.display.should_quit:
                        print("Display closed – stopping.")
                        break

                # ── console log ──
                if verbose and tick % 20 == 0:
                    st = self.get_vehicle_state()
                    cot = ""
                    if output and output.reasoning:
                        cot = output.reasoning[:80].replace("\n", " ") + "…"
                    print(
                        f"Tick {tick:5d} | Inf #{self.inference_count:4d} | "
                        f"Speed {st['speed_kmh']:5.1f} km/h | {cot}"
                    )
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
        print("Cleanup complete")

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False

