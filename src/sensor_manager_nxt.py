"""
CARLA Sensor Manager (NXT)
Handles sensor spawning, data collection, and temporal frame buffering
for Alpamayo-R1 inference.

Alpamayo-R1 expects **4 cameras × 4 temporal frames = 16 images**.
The cameras must match the model's training configuration:
  - camera_cross_left_120fov   (index 0)  120° FOV, facing left
  - camera_front_wide_120fov   (index 1)  120° FOV, facing front
  - camera_cross_right_120fov  (index 2)  120° FOV, facing right
  - camera_front_tele_30fov    (index 6)   30° FOV, facing front (telephoto)
"""

import carla
import numpy as np
import queue
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import deque


# ---------------------------------------------------------------------------
# Alpamayo camera-index mapping (must match ar1_model.py / training data)
# ---------------------------------------------------------------------------
ALPAMAYO_CAMERA_INDEX = {
    "camera_cross_left_120fov": 0,
    "camera_front_wide_120fov": 1,
    "camera_cross_right_120fov": 2,
    "camera_front_tele_30fov": 6,
}

# Sorted camera names by model index – this is the order images must
# be arranged when passed to ``helper.create_message()``.
ALPAMAYO_CAMERA_ORDER: List[str] = sorted(
    ALPAMAYO_CAMERA_INDEX, key=ALPAMAYO_CAMERA_INDEX.get  # type: ignore[arg-type]
)


@dataclass
class SensorConfig:
    """Configuration for a sensor"""
    sensor_type: str
    transform: carla.Transform
    attributes: Dict[str, str] = field(default_factory=dict)


@dataclass
class TimestampedFrame:
    """A camera frame paired with its simulation timestamp."""
    timestamp_us: int       # Simulation time in microseconds
    image: np.ndarray       # HWC uint8 RGB image


# Default camera image size – matches training data (1900×1080).
# Lower resolution can be used at the cost of accuracy.
_CAM_W = "1900"
_CAM_H = "1080"


class SensorManager:
    """
    Manages CARLA sensors with temporal frame buffering.

    Alpamayo-R1 requires 4 cameras × 4 temporal frames = 16 images.
    This manager spawns the correct camera rig and automatically buffers
    incoming frames with timestamps.
    """

    # -- Camera configs matching Alpamayo's training rig ----------------
    # Approximate positions on a sedan (Tesla Model 3 scale).
    # "cross" cameras are angled ≈60° to the side.
    CAMERA_CONFIGS = {
        "camera_front_wide_120fov": SensorConfig(
            sensor_type="sensor.camera.rgb",
            transform=carla.Transform(
                carla.Location(x=2.0, z=1.4),
                carla.Rotation(pitch=0),
            ),
            attributes={
                "image_size_x": _CAM_W,
                "image_size_y": _CAM_H,
                "fov": "120",
            },
        ),
        "camera_front_tele_30fov": SensorConfig(
            sensor_type="sensor.camera.rgb",
            transform=carla.Transform(
                carla.Location(x=2.0, z=1.4),
                carla.Rotation(pitch=0),
            ),
            attributes={
                "image_size_x": _CAM_W,
                "image_size_y": _CAM_H,
                "fov": "30",
            },
        ),
        "camera_cross_left_120fov": SensorConfig(
            sensor_type="sensor.camera.rgb",
            transform=carla.Transform(
                carla.Location(x=0.5, y=-0.4, z=1.4),
                carla.Rotation(yaw=-60),
            ),
            attributes={
                "image_size_x": _CAM_W,
                "image_size_y": _CAM_H,
                "fov": "120",
            },
        ),
        "camera_cross_right_120fov": SensorConfig(
            sensor_type="sensor.camera.rgb",
            transform=carla.Transform(
                carla.Location(x=0.5, y=0.4, z=1.4),
                carla.Rotation(yaw=60),
            ),
            attributes={
                "image_size_x": _CAM_W,
                "image_size_y": _CAM_H,
                "fov": "120",
            },
        ),
    }

    # -- Non-camera sensors --------------------------------------------
    OTHER_CONFIGS = {
        "gnss": SensorConfig(
            sensor_type="sensor.other.gnss",
            transform=carla.Transform(
                carla.Location(x=0.0, z=0.0),
            ),
            attributes={},
        ),
        "imu": SensorConfig(
            sensor_type="sensor.other.imu",
            transform=carla.Transform(
                carla.Location(x=0.0, z=0.0),
            ),
            attributes={},
        ),
    }

    # Merge for lookup
    DEFAULT_CONFIGS = {**CAMERA_CONFIGS, **OTHER_CONFIGS}

    def __init__(
        self,
        world: carla.World,
        vehicle: carla.Actor,
        frame_buffer_size: int = 32,
    ):
        """
        Args:
            world: CARLA world instance.
            vehicle: Vehicle to attach sensors to.
            frame_buffer_size: Max frames to keep per camera (ring buffer).
        """
        self.world = world
        self.vehicle = vehicle
        self.blueprint_library = world.get_blueprint_library()

        self.sensors: Dict[str, carla.Actor] = {}
        self.data_queues: Dict[str, queue.Queue] = {}
        self.latest_data: Dict[str, object] = {}

        # Per-camera ring buffers for temporal context
        self.frame_buffer_size = frame_buffer_size
        self.frame_buffers: Dict[str, deque] = {}

    # ------------------------------------------------------------------
    # Sensor lifecycle
    # ------------------------------------------------------------------

    def spawn_sensor(
        self,
        name: str,
        config: Optional[SensorConfig] = None,
    ) -> carla.Actor:
        """Spawn a sensor and attach to vehicle."""
        if config is None:
            if name not in self.DEFAULT_CONFIGS:
                raise ValueError(f"Unknown sensor: {name}. Provide custom config.")
            config = self.DEFAULT_CONFIGS[name]

        blueprint = self.blueprint_library.find(config.sensor_type)
        for attr, value in config.attributes.items():
            if blueprint.has_attribute(attr):
                blueprint.set_attribute(attr, value)

        sensor = self.world.spawn_actor(
            blueprint,
            config.transform,
            attach_to=self.vehicle,
        )

        self.data_queues[name] = queue.Queue()

        # Create frame buffer for RGB camera sensors
        if "camera.rgb" in config.sensor_type:
            self.frame_buffers[name] = deque(maxlen=self.frame_buffer_size)

        sensor.listen(lambda data, n=name: self._on_sensor_data(n, data))
        self.sensors[name] = sensor
        print(f"Spawned sensor: {name} ({config.sensor_type})")
        return sensor

    def spawn_default_sensors(self) -> Dict[str, carla.Actor]:
        """Spawn the full Alpamayo camera rig (4 cameras) + GNSS + IMU."""
        for cam_name in ALPAMAYO_CAMERA_ORDER:
            self.spawn_sensor(cam_name)
        for other_name in self.OTHER_CONFIGS:
            self.spawn_sensor(other_name)
        return self.sensors

    def destroy_all(self) -> None:
        """Destroy all sensors and clear buffers."""
        for name, sensor in self.sensors.items():
            if sensor.is_alive:
                sensor.stop()
                sensor.destroy()
                print(f"Destroyed sensor: {name}")
        self.sensors.clear()
        self.data_queues.clear()
        self.latest_data.clear()
        self.frame_buffers.clear()

    # ------------------------------------------------------------------
    # Internal callback
    # ------------------------------------------------------------------

    def _on_sensor_data(self, name: str, data: object) -> None:
        """Callback invoked by CARLA on each sensor tick."""
        self.latest_data[name] = data

        # Append to frame buffer for camera sensors
        if name in self.frame_buffers:
            image = self._process_camera_image(data)
            timestamp_us = int(data.timestamp * 1_000_000)
            self.frame_buffers[name].append(
                TimestampedFrame(timestamp_us=timestamp_us, image=image)
            )

        try:
            self.data_queues[name].put_nowait(data)
        except queue.Full:
            try:
                self.data_queues[name].get_nowait()
                self.data_queues[name].put_nowait(data)
            except queue.Empty:
                pass

    # ------------------------------------------------------------------
    # Data retrieval
    # ------------------------------------------------------------------

    def get_camera_image(
        self,
        name: str = "front_camera",
        timeout: float = 1.0,
    ) -> Optional[np.ndarray]:
        """Get the latest single camera image as HWC uint8 RGB array."""
        try:
            data = self.data_queues[name].get(timeout=timeout)
            return self._process_camera_image(data)
        except queue.Empty:
            if name in self.latest_data:
                return self._process_camera_image(self.latest_data[name])
            return None

    def get_buffered_frames(
        self,
        name: str = "camera_front_wide_120fov",
        count: int = 4,
    ) -> Optional[List[TimestampedFrame]]:
        """
        Get the latest ``count`` frames from a single camera's ring buffer.

        Returns:
            List of TimestampedFrame (oldest first), or None if not
            enough frames have been collected yet.
        """
        if name not in self.frame_buffers:
            return None
        buf = self.frame_buffers[name]
        if len(buf) < count:
            return None
        return list(buf)[-count:]

    def get_all_camera_frames(
        self,
        count: int = 4,
    ) -> Optional[Dict[str, List[TimestampedFrame]]]:
        """
        Get the latest ``count`` frames from **every** Alpamayo camera.

        Returns:
            Dict mapping camera name → list of TimestampedFrame,
            or None if ANY camera has fewer than ``count`` frames.
        """
        result: Dict[str, List[TimestampedFrame]] = {}
        for cam_name in ALPAMAYO_CAMERA_ORDER:
            frames = self.get_buffered_frames(cam_name, count)
            if frames is None:
                return None
            result[cam_name] = frames
        return result

    def get_gnss_data(self, name: str = "gnss") -> Optional[Dict]:
        """Get latest GNSS data."""
        if name in self.latest_data:
            data = self.latest_data[name]
            return {
                "latitude": data.latitude,
                "longitude": data.longitude,
                "altitude": data.altitude,
            }
        return None

    def get_imu_data(self, name: str = "imu") -> Optional[Dict]:
        """Get latest IMU data."""
        if name in self.latest_data:
            data = self.latest_data[name]
            return {
                "accelerometer": {
                    "x": data.accelerometer.x,
                    "y": data.accelerometer.y,
                    "z": data.accelerometer.z,
                },
                "gyroscope": {
                    "x": data.gyroscope.x,
                    "y": data.gyroscope.y,
                    "z": data.gyroscope.z,
                },
                "compass": data.compass,
            }
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _process_camera_image(image: carla.Image) -> np.ndarray:
        """Convert a CARLA image (BGRA) to a uint8 RGB numpy array."""
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        # BGRA → RGB
        return array[:, :, :3][:, :, ::-1].copy()

