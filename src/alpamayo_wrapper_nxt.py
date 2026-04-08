"""
Alpamayo Model Wrapper (NXT)

Loads NVIDIA Alpamayo VLA model (R1 or 1.5) and runs trajectory
inference from multi-camera images + ego history.

Supports both Alpamayo-R1 (alpamayo_r1 package) and Alpamayo-1.5
(alpamayo1_5 package), auto-detecting the version from the model name.

Input:  4 cameras × 4 temporal frames (16 images) + ego pose history
        + optional navigation text (Alpamayo 1.5 only)
Output: trajectory waypoints in rig frame + chain-of-thought reasoning
"""

import torch
import numpy as np
import time
from typing import Dict, List, Optional
from dataclasses import dataclass
from types import ModuleType


@dataclass
class AlpamayoOutput:
    """Output from Alpamayo model inference"""
    trajectory_xy: np.ndarray          # Selected waypoints (T, 2) in rig frame
    headings: np.ndarray               # Heading angles (T,) for selected trajectory
    reasoning: Optional[str] = None    # Chain-of-thought reasoning text (for SELECTED trajectory)
    meta_action: Optional[str] = None  # Meta-action text (for SELECTED trajectory)
    all_trajectories_xy: Optional[np.ndarray] = None  # All candidates (N, T, 2)
    all_headings: Optional[np.ndarray] = None          # All headings (N, T)
    selected_index: int = 0            # Index of the selected trajectory in all_*
    all_reasoning: Optional[List[str]] = None  # CoT per candidate (N,)
    all_meta_actions: Optional[List[str]] = None  # Meta-actions per candidate (N,)


# Version constants
VERSION_R1 = "r1"
VERSION_15 = "1.5"

# Known model-name → version mappings
_KNOWN_MODELS: Dict[str, str] = {
    "nvidia/Alpamayo-R1-10B": VERSION_R1,
    "nvidia/Alpamayo-1.5-10B": VERSION_15,
}


def detect_model_version(model_name: str) -> str:
    """Auto-detect model version from the model name / path."""
    if model_name in _KNOWN_MODELS:
        return _KNOWN_MODELS[model_name]
    lower = model_name.lower()
    if "1.5" in lower or "1_5" in lower:
        return VERSION_15
    return VERSION_R1


class AlpamayoWrapper:
    """
    Wrapper for NVIDIA Alpamayo VLA models (R1 and 1.5).

    Automatically detects the model version from ``model_name`` and
    loads the corresponding package (``alpamayo_r1`` or ``alpamayo1_5``).
    """

    DTYPE = torch.bfloat16

    def __init__(
        self,
        model_name: str = "nvidia/Alpamayo-1.5-10B",
        device: str = "cuda",
        num_traj_samples: int = 6,
        top_p: float = 0.98,
        temperature: float = 0.6,
        timing_log: bool = False,
    ):
        self.model_name = model_name
        self.device = device
        self.num_traj_samples = num_traj_samples
        self.top_p = top_p
        self.temperature = temperature
        self.timing_log = timing_log

        self.model_version: str = detect_model_version(model_name)
        self.model = None
        self.processor = None
        self._helper: Optional[ModuleType] = None
        self.is_loaded = False
        self.pred_num_waypoints: Optional[int] = None

    @property
    def is_v15(self) -> bool:
        return self.model_version == VERSION_15

    def load_model(self) -> None:
        """Load Alpamayo model using the official package API."""
        try:
            if self.is_v15:
                from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
                from alpamayo1_5 import helper

                print(f"Loading Alpamayo 1.5 model: {self.model_name}")
                print("This may take a few minutes (~22GB)...")
                self.model = Alpamayo1_5.from_pretrained(
                    self.model_name, dtype=self.DTYPE
                ).to(self.device)
            else:
                from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
                from alpamayo_r1 import helper

                print(f"Loading Alpamayo R1 model: {self.model_name}")
                print("This may take a few minutes (~22GB)...")
                self.model = AlpamayoR1.from_pretrained(
                    self.model_name, dtype=self.DTYPE
                ).to(self.device)

            self._helper = helper
            self.processor = helper.get_processor(self.model.tokenizer)

            output_shape = self.model.action_space.get_action_space_dims()
            self.pred_num_waypoints, _ = output_shape

            self.model.eval()
            self.is_loaded = True
            print(
                f"Model loaded successfully! "
                f"(version={self.model_version}, waypoints={self.pred_num_waypoints})"
            )

        except ImportError as e:
            pkg = "alpamayo1_5" if self.is_v15 else "alpamayo_r1"
            raise ImportError(
                f"{pkg} package not found. "
                "Ensure it is installed or accessible via PYTHONPATH.\n"
                f"  e.g.: pip install -e /path/to/{pkg.replace('_', '')}\n"
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
        diffusion_steps: Optional[int] = None,
        nav_text: Optional[str] = None,
        use_cfg_nav: bool = False,
        cfg_nav_guidance_weight: Optional[float] = None,
        use_camera_indices: bool = True,
    ) -> AlpamayoOutput:
        """
        Run Alpamayo inference.

        Args:
            camera_frames: dict mapping camera name →
                list of HWC uint8 RGB images (``context_length`` frames each).
            ego_history_xyz: Ego position history (1, 1, 16, 3) in rig frame.
            ego_history_rot: Ego rotation history (1, 1, 16, 3, 3) in rig frame.
            max_generation_length: Max tokens for VLM text generation.
            diffusion_steps: Override diffusion denoising steps.
            nav_text: Navigation instruction (Alpamayo 1.5 only).
                e.g. ``"Turn left in 40m"``.
            use_cfg_nav: Use classifier-free guidance for navigation
                conditioning (Alpamayo 1.5 only).  Requires ``nav_text``.
            cfg_nav_guidance_weight: CFG guidance weight (alpha).
                Only used when ``use_cfg_nav=True``.
            use_camera_indices: Pass camera-index annotations to
                ``helper.create_message`` (Alpamayo 1.5 only). Disable
                this for A/B comparison against index-free prompting.

        Returns:
            AlpamayoOutput with predicted trajectory and reasoning.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        t_start = time.perf_counter()

        helper = self._helper
        from .sensor_manager_nxt import ALPAMAYO_CAMERA_ORDER, ALPAMAYO_CAMERA_INDEX

        t_prep_start = time.perf_counter()
        all_frames: List[torch.Tensor] = []
        for cam_name in ALPAMAYO_CAMERA_ORDER:
            for img_hwc in camera_frames[cam_name]:
                t = torch.from_numpy(img_hwc).permute(2, 0, 1)
                all_frames.append(t)

        image_tensor = torch.stack(all_frames, dim=0)

        # Build chat messages (version-dependent)
        if self.is_v15:
            create_message_kwargs = {
                "nav_text": nav_text,
                # In no-nav mode, keep the nav-style prompt format for a fair
                # baseline instead of switching to a different instruction style.
                "use_nav_prompt": (nav_text is None),
            }
            if use_camera_indices:
                # Alpamayo 1.5 expects camera-index annotations in the prompt
                # to match the training message format.
                create_message_kwargs["camera_indices"] = torch.tensor(
                    [ALPAMAYO_CAMERA_INDEX[name] for name in ALPAMAYO_CAMERA_ORDER],
                    dtype=torch.long,
                )
                create_message_kwargs["num_frames_per_camera"] = len(
                    camera_frames[ALPAMAYO_CAMERA_ORDER[0]]
                )
            messages = helper.create_message(image_tensor, **create_message_kwargs)
        else:
            messages = helper.create_message(image_tensor)

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )

        model_inputs = {
            "tokenized_data": inputs,
            "ego_history_xyz": ego_history_xyz,
            "ego_history_rot": ego_history_rot,
        }
        model_inputs = helper.to_device(model_inputs, self.device)
        t_prep_end = time.perf_counter()

        diffusion_kwargs = None
        if diffusion_steps is not None:
            diffusion_kwargs = {"inference_step": diffusion_steps}

        # Choose inference method
        use_cfg = (
            use_cfg_nav
            and self.is_v15
            and nav_text is not None
        )

        t_model_start = time.perf_counter()
        with torch.autocast(self.device, dtype=self.DTYPE):
            if use_cfg:
                if diffusion_kwargs is None:
                    diffusion_kwargs = {}
                if cfg_nav_guidance_weight is not None:
                    diffusion_kwargs["inference_guidance_weight"] = cfg_nav_guidance_weight
                pred_xyz, pred_rot, extra = (
                    self.model.sample_trajectories_from_data_with_vlm_rollout_cfg_nav(
                        data=model_inputs,
                        top_p=self.top_p,
                        temperature=self.temperature,
                        num_traj_samples=self.num_traj_samples,
                        max_generation_length=max_generation_length,
                        diffusion_kwargs=diffusion_kwargs,
                        return_extra=True,
                    )
                )
            else:
                pred_xyz, pred_rot, extra = (
                    self.model.sample_trajectories_from_data_with_vlm_rollout(
                        data=model_inputs,
                        top_p=self.top_p,
                        temperature=self.temperature,
                        num_traj_samples=self.num_traj_samples,
                        max_generation_length=max_generation_length,
                        diffusion_kwargs=diffusion_kwargs,
                        return_extra=True,
                    )
                )
        t_model_end = time.perf_counter()

        output = self._postprocess(pred_xyz, extra)
        t_end = time.perf_counter()

        if self.timing_log:
            prep_ms = (t_prep_end - t_prep_start) * 1000.0
            model_ms = (t_model_end - t_model_start) * 1000.0
            post_ms = (t_end - t_model_end) * 1000.0
            total_ms = (t_end - t_start) * 1000.0
            print(
                "[timing] "
                f"prep={prep_ms:.1f}ms "
                f"model={model_ms:.1f}ms "
                f"post={post_ms:.1f}ms "
                f"total={total_ms:.1f}ms "
                f"samples={self.num_traj_samples} "
                f"max_gen={max_generation_length} "
                f"diff_steps={diffusion_steps if diffusion_steps is not None else 'default'} "
                f"cfg_nav={use_cfg}"
            )

        return output

    def _postprocess(
        self,
        pred_xyz: torch.Tensor,
        extra: dict,
    ) -> AlpamayoOutput:
        """Convert raw model output to AlpamayoOutput."""
        all_traj = pred_xyz[0, 0].cpu().numpy()   # (N, T, 3)
        all_traj_xy = all_traj[:, :, :2]           # (N, T, 2)

        all_heads = np.array([
            self._compute_headings(t) for t in all_traj_xy
        ])

        selected_idx = self._select_medoid(all_traj_xy)
        trajectory_xy = all_traj_xy[selected_idx]
        headings = all_heads[selected_idx]

        all_reasoning: Optional[List[str]] = None
        reasoning: Optional[str] = None
        if "cot" in extra and extra["cot"].size > 0:
            cot_arr = extra["cot"][0, 0]
            all_reasoning = [str(c) for c in cot_arr]
            reasoning = all_reasoning[selected_idx]

        all_meta_actions: Optional[List[str]] = None
        meta_action: Optional[str] = None

        if "meta_action" in extra and extra["meta_action"].size > 0:
            ma_arr = extra["meta_action"][0, 0]
            all_meta_actions = [str(m) for m in ma_arr]
            meta_action = all_meta_actions[selected_idx]
            if not meta_action:
                all_meta_actions = None
                meta_action = None

        return AlpamayoOutput(
            trajectory_xy=trajectory_xy,
            headings=headings,
            reasoning=reasoning,
            meta_action=meta_action,
            all_trajectories_xy=all_traj_xy,
            all_headings=all_heads,
            selected_index=selected_idx,
            all_reasoning=all_reasoning,
            all_meta_actions=all_meta_actions,
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

    _GO_THRESHOLD = 3.0

    @staticmethod
    def _traj_length(traj_xy: np.ndarray) -> float:
        """Arc-length of a (T, 2) trajectory."""
        if len(traj_xy) < 2:
            return 0.0
        return float(np.sum(np.linalg.norm(np.diff(traj_xy, axis=0), axis=1)))

    @classmethod
    def _select_medoid(cls, all_traj_xy: np.ndarray) -> int:
        """
        Majority-vote trajectory selection.

        1. Classify each candidate as "go" (arc-length ≥ threshold) or "stop".
        2. Pick the majority group; compute medoid within that group.
        """
        n = all_traj_xy.shape[0]
        if n <= 1:
            return 0

        lengths = np.array([cls._traj_length(t) for t in all_traj_xy])
        go_mask = lengths >= cls._GO_THRESHOLD
        n_go = int(go_mask.sum())

        if n_go > 0 and n_go >= n - n_go:
            subset_indices = np.where(go_mask)[0]
        else:
            subset_indices = np.where(~go_mask)[0]

        subset = all_traj_xy[subset_indices]
        mean_traj = subset.mean(axis=0)
        diffs = subset - mean_traj[None]
        sq_dist = (diffs ** 2).sum(axis=(1, 2))
        best_local = int(np.argmin(sq_dist))
        return int(subset_indices[best_local])

    def predict_dummy(self, **kwargs) -> AlpamayoOutput:
        """Dummy prediction for testing without GPU / model."""
        n_wp = self.pred_num_waypoints or 40
        n_samples = self.num_traj_samples

        rng = np.random.default_rng()
        all_traj_xy = np.zeros((n_samples, n_wp, 2))
        for i in range(n_samples):
            t = np.linspace(0, 4.0, n_wp)
            speed = 5.0 + rng.normal(0, 0.5)
            lateral = rng.normal(0, 0.3, size=n_wp).cumsum() * 0.05
            all_traj_xy[i] = np.stack([t * speed, lateral], axis=1)

        all_heads = np.array([
            self._compute_headings(t) for t in all_traj_xy
        ])

        selected_idx = self._select_medoid(all_traj_xy)

        return AlpamayoOutput(
            trajectory_xy=all_traj_xy[selected_idx],
            headings=all_heads[selected_idx],
            reasoning="[DUMMY MODE] Driving straight at low speed.",
            all_trajectories_xy=all_traj_xy,
            all_headings=all_heads,
            selected_index=selected_idx,
        )

