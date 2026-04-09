#!/usr/bin/env python3
"""
Run CARLA Alpamayo Observer (NXT) — Open-Loop Evaluation Mode

The CARLA autopilot drives the vehicle while Alpamayo observes and
produces trajectory predictions and chain-of-thought reasoning.  The
model output is displayed alongside the actual driving but is NEVER
used for vehicle control.

Supports both Alpamayo R1 and Alpamayo 1.5.

Usage:
    # Alpamayo R1 (default)
    source /work/yasu/program/alpamayo/alpamayo/ar1_venv/bin/activate
    python run_observer_nxt.py --ticks 3000 --map Town03

    # Alpamayo 1.5 (with navigation)
    source /work/yasu/program/alpamayo/alpamayo1.5/a1_5_venv/bin/activate
    python run_observer_nxt.py --model nvidia/Alpamayo-1.5-10B --ticks 3000

    # Dummy mode (no GPU, for testing)
    python run_observer_nxt.py --dummy --ticks 500
"""

import argparse
import sys

sys.path.insert(0, "..")
sys.path.insert(0, "../src")

from src.carla_observer_nxt import CarlaObserver, ObserverConfig


def main():
    parser = argparse.ArgumentParser(
        description="Run CARLA Alpamayo Observer — Open-Loop Evaluation"
    )
    parser.add_argument("--host", default="localhost", help="CARLA server host")
    parser.add_argument("--port", type=int, default=2000, help="CARLA server port")
    parser.add_argument(
        "--map", type=str, default=None, metavar="NAME",
        help="CARLA map to load (e.g. Town01..Town15, Town10HD). "
             "Omit to keep the current map.",
    )
    parser.add_argument(
        "--weather", type=str, default="ClearNoon",
        help="Weather preset: ClearNoon (default), CloudyNoon, WetNoon, "
             "SoftRainNoon, ClearSunset, etc.",
    )
    parser.add_argument(
        "--ticks", type=int, default=3000,
        help="Max simulation ticks to run (default 3000 = 5 min at 10 Hz)",
    )
    parser.add_argument(
        "--dummy", action="store_true",
        help="Use dummy model (no GPU required)",
    )
    parser.add_argument(
        "--spawn", type=int, default=-1,
        help="Spawn point index (-1 = random)",
    )
    parser.add_argument(
        "--vehicle", default="vehicle.tesla.model3",
        help="Vehicle blueprint filter",
    )
    parser.add_argument(
        "--model", default="nvidia/Alpamayo-1.5-10B",
        help="HuggingFace model ID or local path (e.g. nvidia/Alpamayo-R1-10B for R1)",
    )
    parser.add_argument(
        "--autopilot-speed", type=float, default=-20.0,
        help="Autopilot speed offset %% relative to speed limit "
             "(negative = faster, positive = slower; default -20)",
    )

    # NPC traffic
    parser.add_argument(
        "--npc-vehicles", type=int, default=0, metavar="N",
        help="Number of NPC vehicles to spawn (default 0)",
    )
    parser.add_argument(
        "--npc-walkers", type=int, default=0, metavar="N",
        help="Number of NPC pedestrians to spawn (default 0)",
    )

    # ── Navigation (Alpamayo 1.5) ──
    parser.add_argument(
        "--no-nav", action="store_true",
        help="Disable navigation instructions (Alpamayo 1.5 only; ignored for R1)",
    )
    parser.add_argument(
        "--cfg-nav", action="store_true",
        help="Use classifier-free guidance for navigation conditioning (1.5 only)",
    )
    parser.add_argument(
        "--cfg-nav-weight", type=float, default=None, metavar="ALPHA",
        help="CFG guidance weight for navigation (1.5 only; None = model default)",
    )
    parser.add_argument(
        "--no-camera-indices", action="store_true",
        help="Do not pass camera_indices into Alpamayo 1.5 prompt construction "
             "(for A/B testing)",
    )
    parser.add_argument(
        "--nav-text", type=str, default=None, metavar="TEXT",
        help="Force a fixed navigation instruction text (1.5 only). "
             "If set, this overrides TrafficManager-derived navigation.",
    )

    # Alpamayo inference parameters
    parser.add_argument(
        "--max-gen-len", type=int, default=256,
        help="VLM max generation tokens (default 256; use 64 for faster but weaker reasoning)",
    )
    parser.add_argument(
        "--num-traj-samples", type=int, default=6,
        help="Number of trajectory candidates per inference (default 6)",
    )
    parser.add_argument(
        "--diffusion-steps", type=int, default=5,
        help="Flow-matching diffusion steps (default 5)",
    )
    parser.add_argument(
        "--cam-res", type=str, default="full",
        help="Camera resolution: 'full', 'half', 'low', or 'WxH'",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.6,
        help="VLM text-generation temperature",
    )
    parser.add_argument(
        "--top-p", type=float, default=0.98,
        help="VLM nucleus-sampling threshold",
    )
    parser.add_argument(
        "--timing-log", action="store_true",
        help="Print per-inference timing breakdown (prep/model/post/total)",
    )
    parser.add_argument(
        "--sim-fps", type=float, default=10.0,
        help="Simulation FPS (default 10.0)",
    )

    # Display / recording
    parser.add_argument(
        "--no-display", action="store_true",
        help="Disable pygame dashboard window",
    )
    parser.add_argument(
        "--record", type=str, default=None, metavar="PATH",
        help="Record the dashboard to an MP4 file",
    )
    parser.add_argument(
        "--crf", type=int, default=23,
        help="H.264 CRF for recording (0=lossless, 23=default)",
    )
    args = parser.parse_args()

    if args.record and args.no_display:
        print("Warning: --record requires the display. Enabling display.")
        args.no_display = False

    config = ObserverConfig(
        host=args.host,
        port=args.port,
        map_name=args.map,
        weather=args.weather,
        spawn_point_index=args.spawn,
        vehicle_filter=args.vehicle,
        use_dummy_model=args.dummy,
        model_name=args.model,
        autopilot_speed_pct=args.autopilot_speed,
        num_npc_vehicles=args.npc_vehicles,
        num_npc_walkers=args.npc_walkers,
        nav_enabled=not args.no_nav,
        use_cfg_nav=args.cfg_nav,
        cfg_nav_guidance_weight=args.cfg_nav_weight,
        use_camera_indices=not args.no_camera_indices,
        nav_text_override=args.nav_text,
        num_traj_samples=args.num_traj_samples,
        max_generation_length=args.max_gen_len,
        diffusion_steps=args.diffusion_steps,
        cam_resolution=args.cam_res,
        vlm_temperature=args.temperature,
        vlm_top_p=args.top_p,
        timing_log=args.timing_log,
        sim_fps=args.sim_fps,
        enable_display=not args.no_display,
        record_path=args.record,
        record_crf=args.crf,
    )

    with CarlaObserver(config) as observer:
        observer.run(
            max_ticks=args.ticks,
            verbose=True,
        )


if __name__ == "__main__":
    main()

