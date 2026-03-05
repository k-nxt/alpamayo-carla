#!/usr/bin/env python3
"""
Run CARLA Alpamayo Agent (NXT)

Usage:
    # Activate the venv that has alpamayo_r1, transformers, carla
    source /work/yasu/program/alpamayo/alpamayo/ar1_venv/bin/activate

    # Dummy mode (no GPU, for testing CARLA integration)
    python run_agent_nxt.py --dummy --frames 200

    # Full model
    python run_agent_nxt.py --frames 500
"""

import argparse
import sys

sys.path.insert(0, "..")

from src.carla_alpamayo_agent_nxt import CarlaAlpamayoAgent, AgentConfig


def main():
    parser = argparse.ArgumentParser(description="Run CARLA Alpamayo Agent (NXT)")
    parser.add_argument("--host", default="localhost", help="CARLA server host")
    parser.add_argument("--port", type=int, default=2000, help="CARLA server port")
    parser.add_argument(
        "--frames", type=int, default=500,
        help="Max inference frames to run",
    )
    parser.add_argument(
        "--dummy", action="store_true",
        help="Use dummy model (no GPU required)",
    )
    parser.add_argument(
        "--spawn", type=int, default=-1,
        help="Spawn point index (-1 = random, 0..N = fixed index)",
    )
    parser.add_argument(
        "--vehicle", default="vehicle.mercedes.coupe_2020",
        help="Vehicle blueprint filter (e.g. vehicle.mercedes.coupe_2020, vehicle.lincoln.mkz_2020)",
    )
    parser.add_argument(
        "--model", default="nvidia/Alpamayo-R1-10B",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--max-speed", type=float, default=30.0,
        help="Max speed in km/h",
    )
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
        help="Flow-matching diffusion steps (default 5; original 10)",
    )
    parser.add_argument(
        "--cam-res", type=str, default="full",
        help="Camera resolution: 'full' (1900x1080), 'half' (960x540), 'low' (640x360), or 'WxH'",
    )
    parser.add_argument(
        "--inference-interval", type=int, default=1,
        help="Run inference every N simulation ticks (default 1 → every tick)",
    )
    parser.add_argument(
        "--sim-fps", type=float, default=10.0,
        help="Simulation FPS (default 10.0 = 0.1s/tick, matching AR1 training)",
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Disable pygame dashboard window",
    )
    parser.add_argument(
        "--record", type=str, default=None, metavar="PATH",
        help="Record the dashboard to an MP4 file (e.g. --record out.mp4)",
    )
    parser.add_argument(
        "--crf", type=int, default=23,
        help="H.264 CRF for recording (0=lossless, 23=default, 51=worst)",
    )
    args = parser.parse_args()

    # Recording requires the display to be enabled
    if args.record and args.no_display:
        print("Warning: --record requires the display. Enabling display.")
        args.no_display = False

    config = AgentConfig(
        host=args.host,
        port=args.port,
        spawn_point_index=args.spawn,
        vehicle_filter=args.vehicle,
        use_dummy_model=args.dummy,
        model_name=args.model,
        max_speed_kmh=args.max_speed,
        num_traj_samples=args.num_traj_samples,
        max_generation_length=args.max_gen_len,
        diffusion_steps=args.diffusion_steps,
        cam_resolution=args.cam_res,
        sim_fps=args.sim_fps,
        inference_interval=args.inference_interval,
        enable_display=not args.no_display,
        record_path=args.record,
        record_crf=args.crf,
    )

    with CarlaAlpamayoAgent(config) as agent:
        agent.run(
            max_frames=args.frames,
            verbose=True,
        )


if __name__ == "__main__":
    main()

