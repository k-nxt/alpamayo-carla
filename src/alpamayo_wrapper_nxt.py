"""
Alpamayo Model Wrapper (NXT)
Proper integration with NVIDIA Alpamayo-R1 VLA model using official alpamayo_r1 API.

Key differences from original:
- Uses AlpamayoR1.from_pretrained() instead of AutoModelForVision2Seq
- Uses helper.get_processor() (Qwen3-VL base) instead of AutoProcessor on Alpamayo model
- Input: 4 cameras × 4 frames (16 images) + ego history → Output: trajectory waypoints
"""

import torch
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class AlpamayoOutput:
    """Output from Alpamayo model inference"""
    trajectory_xy: np.ndarray          # Predicted waypoints (T, 2) in rig frame
    headings: np.ndarray               # Heading angles (T,) in radians
    reasoning: Optional[str] = None    # Chain-of-thought reasoning text


class AlpamayoWrapper:
    """
    Wrapper for NVIDIA Alpamayo-R1 VLA model.
    Uses the official alpamayo_r1 package for correct model loading and inference.
    """

    DTYPE = torch.bfloat16

    def __init__(
        self,
        model_name: str = "nvidia/Alpamayo-R1-10B",
        device: str = "cuda",
        num_traj_samples: int = 1,
        top_p: float = 0.98,
        temperature: float = 0.6,
    ):
        """
        Args:
            model_name: HuggingFace model ID or local path.
            device: Device for inference ("cuda" or "cpu").
            num_traj_samples: Number of trajectory samples per inference.
            top_p: Top-p sampling for VLM generation.
            temperature: Temperature for VLM sampling.
        """
        self.model_name = model_name
        self.device = device
        self.num_traj_samples = num_traj_samples
        self.top_p = top_p
        self.temperature = temperature

        self.model = None
        self.processor = None
        self.is_loaded = False
        self.pred_num_waypoints: Optional[int] = None

    def load_model(self) -> None:
        """Load Alpamayo model using official alpamayo_r1 API."""
        try:
            from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
            from alpamayo_r1 import helper

            print(f"Loading Alpamayo model: {self.model_name}")
            print("This may take a few minutes (~22GB)...")

            # Load with custom AlpamayoR1 class (NOT AutoModelForVision2Seq)
            self.model = AlpamayoR1.from_pretrained(
                self.model_name, dtype=self.DTYPE
            ).to(self.device)

            # Get processor from Qwen3-VL base model (NOT from Alpamayo model)
            self.processor = helper.get_processor(self.model.tokenizer)

            # Get number of output waypoints from model config
            output_shape = self.model.action_space.get_action_space_dims()
            self.pred_num_waypoints, _ = output_shape

            self.model.eval()
            self.is_loaded = True
            print(f"Model loaded successfully! Output waypoints: {self.pred_num_waypoints}")

        except ImportError as e:
            raise ImportError(
                "alpamayo_r1 package not found. "
                "Ensure it is installed or accessible via PYTHONPATH.\n"
                "  e.g.: pip install -e /path/to/alpamayo\n"
                f"Error: {e}"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to load Alpamayo model: {e}") from e

    @torch.no_grad()
    def predict(
        self,
        camera_frames: Dict[str, List[np.ndarray]],
        ego_history_xyz: torch.Tensor,
        ego_history_rot: torch.Tensor,
        max_generation_length: int = 256,
    ) -> AlpamayoOutput:
        """
        Run Alpamayo inference.

        Args:
            camera_frames: dict mapping camera name →
                list of HWC uint8 RGB images (``context_length`` frames each).
                Camera names must be keys from ``ALPAMAYO_CAMERA_ORDER``.
            ego_history_xyz: Ego position history (1, 1, 16, 3) float, in rig frame
                             relative to current pose.
            ego_history_rot: Ego rotation history (1, 1, 16, 3, 3) float, in rig frame
                             relative to current pose.
            max_generation_length: Max tokens for VLM text generation.

        Returns:
            AlpamayoOutput with predicted trajectory and reasoning.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        from alpamayo_r1 import helper

        # Arrange images in model-index order and convert to torch (C, H, W) uint8
        from .sensor_manager_nxt import ALPAMAYO_CAMERA_ORDER

        all_frames: List[torch.Tensor] = []
        for cam_name in ALPAMAYO_CAMERA_ORDER:
            for img_hwc in camera_frames[cam_name]:
                # HWC uint8 → CHW uint8 tensor (do NOT normalise – the
                # Qwen3-VL processor handles that internally).
                t = torch.from_numpy(img_hwc).permute(2, 0, 1)  # (3, H, W)
                all_frames.append(t)

        # Stack to (N_cameras * context_length, C, H, W)
        image_tensor = torch.stack(all_frames, dim=0)

        # Build chat messages from image frames
        messages = helper.create_message(image_tensor)

        # Tokenize via the processor's chat template
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )

        # Assemble model inputs
        model_inputs = {
            "tokenized_data": inputs,
            "ego_history_xyz": ego_history_xyz,
            "ego_history_rot": ego_history_rot,
        }

        # Move everything to device
        model_inputs = helper.to_device(model_inputs, self.device)

        # Run inference with autocast
        with torch.autocast(self.device, dtype=self.DTYPE):
            pred_xyz, pred_rot, extra = (
                self.model.sample_trajectories_from_data_with_vlm_rollout(
                    data=model_inputs,
                    top_p=self.top_p,
                    temperature=self.temperature,
                    num_traj_samples=self.num_traj_samples,
                    max_generation_length=max_generation_length,
                    return_extra=True,
                )
            )

        # Extract trajectory: (batch, num_traj_sets, num_traj_samples, T, 3) → (T, 2)
        traj = pred_xyz[0, 0, 0].cpu().numpy()  # (T, 3)
        trajectory_xy = traj[:, :2]               # (T, 2) x,y only

        # Compute headings from trajectory
        headings = self._compute_headings(trajectory_xy)

        # Extract chain-of-thought reasoning
        reasoning = None
        if "cot" in extra and extra["cot"].size > 0:
            reasoning = str(extra["cot"].flat[0])

        return AlpamayoOutput(
            trajectory_xy=trajectory_xy,
            headings=headings,
            reasoning=reasoning,
        )

    @staticmethod
    def _compute_headings(trajectory_xy: np.ndarray) -> np.ndarray:
        """Compute heading angles from sequential waypoints."""
        headings = np.zeros(len(trajectory_xy))
        for i in range(len(trajectory_xy) - 1):
            dx = trajectory_xy[i + 1, 0] - trajectory_xy[i, 0]
            dy = trajectory_xy[i + 1, 1] - trajectory_xy[i, 1]
            headings[i] = np.arctan2(dy, dx)
        if len(headings) > 1:
            headings[-1] = headings[-2]
        return headings

    def predict_dummy(self, **kwargs) -> AlpamayoOutput:
        """Dummy prediction for testing without GPU / model."""
        n_wp = self.pred_num_waypoints or 40
        # Straight trajectory: ~5 m/s for 4 seconds
        t = np.linspace(0, 4.0, n_wp)
        trajectory_xy = np.stack([t * 5.0, np.zeros(n_wp)], axis=1)
        headings = np.zeros(n_wp)
        return AlpamayoOutput(
            trajectory_xy=trajectory_xy,
            headings=headings,
            reasoning="[DUMMY MODE] Driving straight at low speed.",
        )

