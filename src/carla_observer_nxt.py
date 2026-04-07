"""
CARLA Alpamayo Observer (NXT) — Open-Loop Evaluation Mode

Runs the CARLA autopilot to drive the vehicle while simultaneously
feeding the camera images to Alpamayo-R1 for inference.  The model's
predicted trajectory and chain-of-thought reasoning are displayed
alongside the actual driving behaviour, but *never* used for control.

Key characteristics:
  - Vehicle is driven by CARLA's TrafficManager autopilot
  - Alpamayo inference runs in a background thread (non-blocking)
  - Simulation time advances continuously during inference
  - Results are displayed with a "delay" label showing how many ticks
    have elapsed since the input was captured

This allows qualitative evaluation of Alpamayo's perception and
planning ability without any feedback from the model to the vehicle.
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
from .nav_planner_nxt import nav_text_from_traffic_manager

# ---------------------------------------------------------------------------
# Constants (same as agent — must match Alpamayo-R1 expectations)
# ---------------------------------------------------------------------------
NUM_HISTORY_STEPS = 16    # ego history length
HISTORY_TIME_STEP = 0.1   # seconds between history samples

# CARLA rear-axle offset from bounding-box center in vehicle-local frame.
_REAR_AXLE_OFFSET_LOCAL = np.array([-1.389, 0.0, 0.0])


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class EgoPose:
    timestamp_us: int
    location: np.ndarray          # (3,) world x y z
    rotation_matrix: np.ndarray   # (3, 3) world rotation


@dataclass
class ObserverConfig:
    """Configuration for the open-loop observer."""
    # CARLA connection
    host: str = "localhost"
    port: int = 2000
    timeout: float = 10.0
    map_name: Optional[str] = None
    weather: str = "ClearNoon"

    # Vehicle
    vehicle_filter: str = "vehicle.mercedes.coupe_2020"
    spawn_point_index: int = -1   # -1 = random

    # Autopilot
    autopilot_speed_pct: float = -20.0
    # Negative → faster than limit, positive → slower.
    # TrafficManager's vehicle_percentage_speed_difference semantics:
    #   0 → limit speed, -20 → 20% above limit, 30 → 30% below limit.

    # NPC traffic
    num_npc_vehicles: int = 0     # number of NPC vehicles to spawn
    num_npc_walkers: int = 0      # number of NPC pedestrians to spawn

    # Alpamayo model
    model_name: str = "nvidia/Alpamayo-1.5-10B"
    use_dummy_model: bool = False
    context_length: int = 4
    num_traj_samples: int = 6
    max_generation_length: int = 256
    diffusion_steps: int = 5
    cam_resolution: str = "full"
    vlm_temperature: float = 0.6
    vlm_top_p: float = 0.98

    # Navigation (Alpamayo 1.5)
    nav_enabled: bool = True              # enable nav instructions (auto-disabled for R1)
    use_cfg_nav: bool = False             # use classifier-free guidance for nav
    cfg_nav_guidance_weight: Optional[float] = None
    nav_text_override: Optional[str] = None  # fixed nav text for debugging (overrides TM nav)
    use_camera_indices: bool = True          # include camera indices in 1.5 prompt construction

    # Simulation
    sim_fps: float = 10.0

    # Display
    enable_display: bool = True
    record_path: Optional[str] = None
    record_crf: int = 23


# ---------------------------------------------------------------------------
# Inference worker (runs in background thread)
# ---------------------------------------------------------------------------
@dataclass
class _InferenceRequest:
    """Payload sent from main thread → inference thread."""
    camera_frames: Dict[str, list]    # camera_name → list of HWC images
    ego_history_xyz: torch.Tensor     # (1,1,16,3)
    ego_history_rot: torch.Tensor     # (1,1,16,3,3)
    submit_tick: int                  # simulation tick when this was captured
    nav_text: Optional[str] = None    # navigation instruction (1.5 only)


@dataclass
class _InferenceResult:
    """Payload returned from inference thread → main thread."""
    output: AlpamayoOutput
    submit_tick: int                  # tick when input was captured
    inference_time: float             # wall-clock seconds
    nav_text: Optional[str] = None


def _inference_worker(
    model: AlpamayoWrapper,
    config: ObserverConfig,
    request_event: threading.Event,
    result_event: threading.Event,
    shared: dict,
    stop_event: threading.Event,
) -> None:
    """Background thread that runs Alpamayo inference.

    Communication via ``shared`` dict protected by events:
      - Main sets shared["request"] and signals request_event
      - Worker reads request, runs inference, sets shared["result"],
        signals result_event
    """
    while not stop_event.is_set():
        # Wait for a request (with timeout to allow stop check)
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
# Supported weather presets (same as agent)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Main observer class
# ---------------------------------------------------------------------------
class CarlaObserver:
    """
    Open-loop observer: CARLA autopilot drives, Alpamayo watches.

    Lifecycle::

        with CarlaObserver(config) as obs:
            obs.run(max_ticks=3000)
    """

    def __init__(self, config: Optional[ObserverConfig] = None):
        self.config = config or ObserverConfig()

        # CARLA handles
        self.client: Optional[carla.Client] = None
        self.world: Optional[carla.World] = None
        self.vehicle: Optional[carla.Actor] = None
        self.spectator: Optional[carla.Actor] = None
        self.traffic_manager: Optional[carla.TrafficManager] = None

        # NPC actors (cleaned up on exit)
        self._npc_vehicles: list = []
        self._npc_walkers: list = []       # walker actors
        self._npc_walker_ctrls: list = []  # AI controller actors

        # Sub-components
        self.sensor_manager: Optional[SensorManager] = None
        self.alpamayo: Optional[AlpamayoWrapper] = None
        self.display: Optional[Display] = None
        self._nav_enabled = False

        # Ego-history ring buffer (world-frame poses)
        self.ego_history: deque[EgoPose] = deque(maxlen=512)

        # Tick → EgoPose mapping for building comparison paths
        self._pose_by_tick: dict[int, EgoPose] = {}
        _POSE_HISTORY_LIMIT = 200   # keep at most this many ticks
        self._pose_history_limit = _POSE_HISTORY_LIMIT

        # Inference state
        self.inference_count = 0
        self.tick_count = 0
        self.last_output: Optional[AlpamayoOutput] = None
        self.last_inference_time: float = 0.0
        self.last_result_tick: int = 0          # tick when the last result input was captured
        self.inference_busy: bool = False

        self.last_nav_text: Optional[str] = None

        # Frozen comparison snapshot (updated only when inference completes)
        self._frozen_actual_rig: Optional[np.ndarray] = None   # (N,2) actual path in rig
        self._frozen_traj_xy: Optional[np.ndarray] = None       # (T,2) selected traj
        self._frozen_all_traj: Optional[np.ndarray] = None      # (N,T,2) all candidates
        self._frozen_sel_idx: int = 0
        self._frozen_reasoning: Optional[str] = None
        self._frozen_meta_action: Optional[str] = None
        self._frozen_delay_ticks: int = -1

        # Threading
        self._request_event = threading.Event()
        self._result_event = threading.Event()
        self._stop_event = threading.Event()
        self._shared: dict = {}
        self._worker_thread: Optional[threading.Thread] = None

    # ==================================================================
    # Initialisation
    # ==================================================================

    def connect(self) -> None:
        cfg = self.config
        print(f"Connecting to CARLA at {cfg.host}:{cfg.port}...")
        self.client = carla.Client(cfg.host, cfg.port)
        self.client.set_timeout(cfg.timeout)
        print(f"Connected to CARLA {self.client.get_server_version()}")

        if cfg.map_name:
            current_map = self.client.get_world().get_map().name
            if not current_map.endswith(cfg.map_name):
                print(f"Loading map {cfg.map_name} (current: {current_map})...")
                self.client.set_timeout(60.0)
                self.world = self.client.load_world(cfg.map_name)
                self.client.set_timeout(cfg.timeout)
            else:
                print(f"Map {cfg.map_name} already loaded.")
                self.world = self.client.get_world()
        else:
            self.world = self.client.get_world()

        self.spectator = self.world.get_spectator()
        print(f"Map: {self.world.get_map().name}")

        # Weather
        self._apply_weather(cfg.weather)

    def _apply_weather(self, preset_name: str) -> None:
        if preset_name in WEATHER_PRESETS:
            weather = WEATHER_PRESETS[preset_name]
        else:
            available = ", ".join(sorted(WEATHER_PRESETS.keys()))
            raise ValueError(
                f"Unknown weather preset '{preset_name}'. Available: {available}"
            )
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

    def _enable_autopilot(self) -> None:
        """Enable CARLA TrafficManager autopilot on the ego vehicle."""
        tm_port = 8000
        self.traffic_manager = self.client.get_trafficmanager(tm_port)
        self.traffic_manager.set_synchronous_mode(True)

        self.vehicle.set_autopilot(True, tm_port)

        # Speed offset: negative = faster than speed limit
        self.traffic_manager.vehicle_percentage_speed_difference(
            self.vehicle, self.config.autopilot_speed_pct,
        )

        # Safe defaults
        self.traffic_manager.auto_lane_change(self.vehicle, True)
        self.traffic_manager.distance_to_leading_vehicle(self.vehicle, 3.0)

        print(f"Autopilot enabled (speed offset: {self.config.autopilot_speed_pct}%)")

    def _spawn_npc_traffic(self) -> None:
        """Spawn NPC vehicles and pedestrians with autopilot / AI control."""
        import random

        cfg = self.config
        tm_port = 8000

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

            # Use SpawnActor batch for walkers
            spawn_points = []
            for _ in range(cfg.num_npc_walkers + 20):
                loc = self.world.get_random_location_from_navigation()
                if loc is not None:
                    spawn_points.append(
                        carla.Transform(location=loc)
                    )

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
            ctrl_bp = self.world.get_blueprint_library().find(
                "controller.ai.walker"
            )
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

        print("Destroyed NPC traffic")

    @staticmethod
    def _parse_cam_resolution(value: str) -> Optional[tuple]:
        if value == "full":
            return None
        if value in RESOLUTION_PRESETS:
            return RESOLUTION_PRESETS[value]
        if "x" in value:
            parts = value.split("x")
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                return (int(parts[0]), int(parts[1]))
        raise ValueError(
            f"Invalid cam_resolution '{value}'. "
            f"Use 'full', 'half', 'low', or 'WxH' (e.g. '640x360')."
        )

    def setup_sensors(self) -> None:
        target_frame_dt = HISTORY_TIME_STEP
        sim_dt = 1.0 / self.config.sim_fps
        subsample_factor = max(1, round(target_frame_dt / sim_dt))

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

        # Synchronous mode
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / self.config.sim_fps
        self.world.apply_settings(settings)

        # Enable autopilot AFTER synchronous mode is set
        self._enable_autopilot()

        # Spawn NPC traffic
        if self.config.num_npc_vehicles > 0 or self.config.num_npc_walkers > 0:
            self._spawn_npc_traffic()

        # Navigation via TrafficManager (Alpamayo 1.5 only)
        if (
            self.config.nav_enabled
            and detect_model_version(self.config.model_name) == VERSION_15
        ):
            self._nav_enabled = True
            print("Navigation enabled via TrafficManager (Alpamayo 1.5)")

        # Start inference worker thread
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

        # Pygame display
        if self.config.enable_display:
            self.display = Display(
                record_path=self.config.record_path,
                record_fps=self.config.sim_fps,
                record_crf=self.config.record_crf,
            )

        print("Observer initialised successfully!")

    # ==================================================================
    # Ego history (same logic as agent)
    # ==================================================================

    def _record_ego_pose(self) -> None:
        transform = self.vehicle.get_transform()
        snap = self.world.get_snapshot()
        timestamp_us = int(snap.timestamp.elapsed_seconds * 1_000_000)

        loc = transform.location
        center_world = np.array([loc.x, loc.y, loc.z], dtype=np.float64)

        rot = transform.rotation
        r = Rotation.from_euler(
            "ZYX", [rot.yaw, rot.pitch, rot.roll], degrees=True
        )
        rotation_matrix = r.as_matrix()

        rear_axle_world = center_world + rotation_matrix @ _REAR_AXLE_OFFSET_LOCAL

        pose = EgoPose(
            timestamp_us=timestamp_us,
            location=rear_axle_world,
            rotation_matrix=rotation_matrix,
        )
        self.ego_history.append(pose)

        # Also record tick → pose for comparison path building
        self._pose_by_tick[self.tick_count] = pose

        # Prune old entries to avoid unbounded growth
        if len(self._pose_by_tick) > self._pose_history_limit:
            oldest = min(self._pose_by_tick.keys())
            del self._pose_by_tick[oldest]

    def _build_ego_history_tensors(
        self,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        step_interval = max(1, round(HISTORY_TIME_STEP * self.config.sim_fps))
        required_len = (NUM_HISTORY_STEPS - 1) * step_interval + 1

        if len(self.ego_history) < required_len:
            return None

        current = self.ego_history[-1]
        R_cur_inv = current.rotation_matrix.T

        F = np.diag([1.0, -1.0, 1.0])

        history_xyz = []
        history_rot = []

        for i in range(NUM_HISTORY_STEPS - 1, -1, -1):
            buf_idx = len(self.ego_history) - 1 - i * step_interval
            buf_idx = max(buf_idx, 0)
            pose = self.ego_history[buf_idx]

            delta_world = pose.location - current.location
            delta_local = R_cur_inv @ delta_world
            delta_rig = F @ delta_local
            history_xyz.append(delta_rig)

            R_rel = R_cur_inv @ pose.rotation_matrix
            R_rel_rig = F @ R_rel @ F
            history_rot.append(R_rel_rig)

        ego_xyz = (
            torch.from_numpy(np.array(history_xyz, dtype=np.float32))
            .unsqueeze(0)
            .unsqueeze(0)
        )
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
        all_cam_frames = self.sensor_manager.get_all_camera_frames(
            count=self.config.context_length
        )
        if all_cam_frames is None:
            return None

        result: Dict[str, list] = {}
        for cam_name in ALPAMAYO_CAMERA_ORDER:
            result[cam_name] = [f.image for f in all_cam_frames[cam_name]]
        return result

    def _get_latest_camera_images(self) -> Dict[str, np.ndarray]:
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
        ctrl = self.vehicle.get_control()
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
            "throttle": ctrl.throttle,
            "brake": ctrl.brake,
            "steer": ctrl.steer,
        }

    # ==================================================================
    # Build comparison path: actual route from submit_tick → now,
    # in the rig frame of the submit_tick pose.
    # ==================================================================

    def _build_actual_path_from_submit(
        self, submit_tick: int,
    ) -> Optional[np.ndarray]:
        """Build the actual vehicle path from *submit_tick* to now,
        transformed into the rig frame of the submit_tick pose.

        This makes the actual path directly comparable with Alpamayo's
        prediction, which is also in that same rig frame.

        Returns (N, 2) array in rig frame (X forward, Y left) or None.
        """
        ref_pose = self._pose_by_tick.get(submit_tick)
        if ref_pose is None:
            return None

        R_inv = ref_pose.rotation_matrix.T
        F = np.diag([1.0, -1.0, 1.0])

        rig_points = []
        for t in range(submit_tick, self.tick_count + 1):
            pose = self._pose_by_tick.get(t)
            if pose is None:
                continue
            delta_world = pose.location - ref_pose.location
            delta_local = R_inv @ delta_world
            delta_rig = F @ delta_local
            rig_points.append(delta_rig[:2])

        if len(rig_points) < 2:
            return None
        return np.array(rig_points)

    # ==================================================================
    # Main loop
    # ==================================================================

    def run(
        self,
        max_ticks: int = 3000,
        verbose: bool = True,
    ) -> None:
        cfg = self.config

        print(f"Starting open-loop observation (autopilot driving, {cfg.sim_fps} Hz)")
        print("Press Ctrl+C to stop\n")

        # --- warm-up: collect enough sensor data & ego history ---
        step_interval = max(1, round(HISTORY_TIME_STEP * cfg.sim_fps))
        subsample_factor = self.sensor_manager.subsample_factor
        min_cam_ticks = (cfg.context_length - 1) * subsample_factor + 2
        warmup_ticks = max(
            min_cam_ticks,
            (NUM_HISTORY_STEPS - 1) * step_interval + 4,
        )
        print(f"Warming up for {warmup_ticks} ticks...")
        for _ in range(warmup_ticks):
            self.world.tick()
            self._record_ego_pose()
            time.sleep(0.01)
        print("Warm-up complete — entering observation loop.\n")

        try:
            for tick in range(max_ticks):
                try:
                    self.world.tick()
                except RuntimeError as e:
                    print(f"\nCARLA tick error: {e}")
                    print("Server may have disconnected — stopping.")
                    break
                self._record_ego_pose()
                self.tick_count += 1

                # --- Check for completed inference result ---
                if self._result_event.is_set():
                    self._result_event.clear()
                    result: Optional[_InferenceResult] = self._shared.get("result")
                    if result is not None:
                        self.last_output = result.output
                        self.last_inference_time = result.inference_time
                        self.last_result_tick = result.submit_tick
                        self.last_nav_text = result.nav_text
                        self.inference_count += 1
                        self.inference_busy = False

                        # ── Snapshot frozen comparison ──
                        # Actual path from submit_tick → now, in submit rig frame
                        self._frozen_actual_rig = (
                            self._build_actual_path_from_submit(result.submit_tick)
                        )
                        self._frozen_traj_xy = result.output.trajectory_xy
                        self._frozen_all_traj = result.output.all_trajectories_xy
                        self._frozen_sel_idx = result.output.selected_index
                        self._frozen_reasoning = result.output.reasoning
                        self._frozen_meta_action = result.output.meta_action
                        self._frozen_delay_ticks = (
                            self.tick_count - result.submit_tick
                        )

                # --- Submit new inference if idle and data ready ---
                if not self.inference_busy:
                    cam_frames = self._prepare_camera_frames()
                    hist = self._build_ego_history_tensors()
                    if cam_frames is not None and hist is not None:
                        ego_xyz, ego_rot = hist
                        nav_text = self.config.nav_text_override
                        if (
                            nav_text is None
                            and self._nav_enabled
                            and self.traffic_manager is not None
                        ):
                            nav_text = nav_text_from_traffic_manager(
                                self.traffic_manager, self.vehicle,
                            )
                        self._shared["request"] = _InferenceRequest(
                            camera_frames=cam_frames,
                            ego_history_xyz=ego_xyz,
                            ego_history_rot=ego_rot,
                            submit_tick=self.tick_count,
                            nav_text=nav_text,
                        )
                        self._request_event.set()
                        self.inference_busy = True

                # --- Update spectator ---
                self._update_spectator()

                # --- Display (always show frozen snapshot) ---
                if self.display is not None:
                    state = self.get_vehicle_state()

                    self.display.tick(
                        camera_images=self._get_latest_camera_images(),
                        vehicle_state=state,
                        trajectory_xy=self._frozen_traj_xy,
                        reasoning=self._frozen_reasoning,
                        inference_count=self.inference_count,
                        tick_count=self.tick_count,
                        inference_time=self.last_inference_time,
                        all_trajectories_xy=self._frozen_all_traj,
                        selected_traj_index=self._frozen_sel_idx,
                        observer_mode=True,
                        delay_ticks=self._frozen_delay_ticks,
                        actual_path_rig=self._frozen_actual_rig,
                        autopilot_state=state,
                        nav_text=self.last_nav_text,
                        meta_action=self._frozen_meta_action,
                    )
                    if self.display.should_quit:
                        print("Display closed — stopping.")
                        break

                # --- Console log ---
                if verbose and self.tick_count % 5 == 0:
                    st = self.get_vehicle_state()
                    delay = (
                        self.tick_count - self.last_result_tick
                        if self.last_result_tick > 0 else -1
                    )
                    busy_tag = " [inferring]" if self.inference_busy else ""
                    cot = ""
                    if self.last_output and self.last_output.meta_action:
                        cot = f"[{self.last_output.meta_action.strip()[:30]}] "
                    if self.last_output and self.last_output.reasoning:
                        cot += self.last_output.reasoning[:50].replace("\n", " ") + "…"
                    nav = ""
                    if self.last_nav_text:
                        nav = f" | Nav: {self.last_nav_text}"
                    print(
                        f"Tick {tick:5d} | Inf #{self.inference_count:4d} "
                        f"({self.last_inference_time:.2f}s) | "
                        f"Spd {st['speed_kmh']:5.1f} | "
                        f"Thr {st['throttle']:.2f} Brk {st['brake']:.2f} "
                        f"Str {st['steer']:+.2f} | "
                        f"Delay {delay:3d}tick{busy_tag}{nav} | {cot}"
                    )

        except KeyboardInterrupt:
            print("\nStopping…")
        finally:
            pass

    def _update_spectator(self) -> None:
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

    # ==================================================================
    # Cleanup / context manager
    # ==================================================================

    def cleanup(self) -> None:
        print("Cleaning up…")

        # Stop inference thread
        self._stop_event.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)

        if self.display:
            self.display.close()
            self.display = None
        if self.sensor_manager:
            self.sensor_manager.destroy_all()
        # Destroy NPC traffic before ego vehicle
        self._destroy_npc_traffic()
        try:
            if self.vehicle and self.vehicle.is_alive:
                self.vehicle.set_autopilot(False)
                self.vehicle.destroy()
                print("Destroyed vehicle")
        except RuntimeError:
            pass
        try:
            if self.world:
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                self.world.apply_settings(settings)
        except RuntimeError:
            pass
        try:
            if self.traffic_manager:
                self.traffic_manager.set_synchronous_mode(False)
        except RuntimeError:
            pass
        print("Cleanup complete")

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False

