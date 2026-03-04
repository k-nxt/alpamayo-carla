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
        "--vehicle", default="vehicle.tesla.model3",
        help="Vehicle blueprint filter",
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
    args = parser.parse_args()

    config = AgentConfig(
        host=args.host,
        port=args.port,
        spawn_point_index=args.spawn,
        vehicle_filter=args.vehicle,
        use_dummy_model=args.dummy,
        model_name=args.model,
        max_speed_kmh=args.max_speed,
        sim_fps=args.sim_fps,
        inference_interval=args.inference_interval,
        enable_display=not args.no_display,
    )

    with CarlaAlpamayoAgent(config) as agent:
        agent.run(
            max_frames=args.frames,
            verbose=True,
        )


if __name__ == "__main__":
    main()

