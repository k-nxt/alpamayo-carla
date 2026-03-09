#!/usr/bin/env python3
"""
Run CARLA Alpamayo Observer (NXT) — Open-Loop Evaluation Mode

The CARLA autopilot drives the vehicle while Alpamayo-R1 observes and
produces trajectory predictions and chain-of-thought reasoning.  The
model output is displayed alongside the actual driving but is NEVER
used for vehicle control.

Usage:
    # Activate the venv
    source /work/yasu/program/alpamayo/alpamayo/ar1_venv/bin/activate

    # Dummy mode (no GPU, for testing)
    python run_observer_nxt.py --dummy --ticks 500

    # Full model
    python run_observer_nxt.py --ticks 3000 --map Town03

    # With recording
    python run_observer_nxt.py --ticks 3000 --map Town01 --record obs01.mp4
"""

import argparse
import sys

sys.path.insert(0, "..")

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
        "--vehicle", default="vehicle.mercedes.coupe_2020",
        help="Vehicle blueprint filter",
    )
    parser.add_argument(
        "--model", default="nvidia/Alpamayo-R1-10B",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--autopilot-speed", type=float, default=-20.0,
        help="Autopilot speed offset %% relative to speed limit "
             "(negative = faster, positive = slower; default -20)",
    )

    # Alpamayo inference parameters
    parser.add_argument(
        "--max-gen-len", type=int, default=64,
        help="VLM max generation tokens (default 64; 256 for full CoT)",
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
        num_traj_samples=args.num_traj_samples,
        max_generation_length=args.max_gen_len,
        diffusion_steps=args.diffusion_steps,
        cam_resolution=args.cam_res,
        vlm_temperature=args.temperature,
        vlm_top_p=args.top_p,
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

