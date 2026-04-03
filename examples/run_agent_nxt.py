#!/usr/bin/env python3
"""
Run CARLA Alpamayo Agent (NXT)

Supports both Alpamayo R1 and Alpamayo 1.5.  The model version is
auto-detected from the --model argument.

Usage:
    # Alpamayo R1 (default)
    source /work/yasu/program/alpamayo/alpamayo/ar1_venv/bin/activate
    python run_agent_nxt.py --frames 500

    # Alpamayo 1.5 (with navigation)
    source /work/yasu/program/alpamayo/alpamayo1.5/a1_5_venv/bin/activate
    python run_agent_nxt.py --model nvidia/Alpamayo-1.5-10B --frames 500

    # Dummy mode (no GPU, for testing CARLA integration)
    python run_agent_nxt.py --dummy --frames 200
"""

import argparse
import sys

sys.path.insert(0, "..")
sys.path.insert(0, "../src")

from src.carla_alpamayo_agent_nxt import CarlaAlpamayoAgent, AgentConfig


def main():
    parser = argparse.ArgumentParser(description="Run CARLA Alpamayo Agent (NXT)")
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
             "SoftRainNoon, MidRainyNoon, HardRainNoon, ClearSunset, etc.",
    )
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
        "--model", default="nvidia/Alpamayo-1.5-10B",
        help="HuggingFace model ID or local path (e.g. nvidia/Alpamayo-R1-10B for R1)",
    )
    parser.add_argument(
        "--max-speed", type=float, default=30.0,
        help="Max speed in km/h",
    )
    parser.add_argument(
        "--min-speed", type=float, default=0.0,
        help="Min cruise speed in km/h (0=no minimum). "
             "Applies as a floor on desired speed except when the model "
             "explicitly commands a stop.  Useful for maps with domain gap "
             "where the model produces consistently short trajectories.",
    )
    parser.add_argument(
        "--steer-gain", type=float, default=1.0,
        help="Steering gain multiplier (default 1.0; >1 = sharper turns). "
             "Useful when the car cannot make curves at higher min-speed.",
    )
    parser.add_argument(
        "--steer-norm-deg", type=float, default=70.0,
        help="Steering normalization angle in degrees (default 70). "
             "Lower values increase steering sensitivity.",
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
        "--temperature", type=float, default=0.6,
        help="VLM text-generation temperature (default 0.6; lower → more deterministic CoT & trajectory)",
    )
    parser.add_argument(
        "--top-p", type=float, default=0.98,
        help="VLM nucleus-sampling threshold (default 0.98)",
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
        "--nav-dest", type=int, default=-1, metavar="IDX",
        help="Spawn-point index for route destination (-1 = random, auto-replan on arrival)",
    )

    # ── Debug ──
    parser.add_argument(
        "--debug-log", type=str, default=None, metavar="PATH",
        help="Write per-tick debug CSV log (trajectory, steering, speed, control)",
    )

    # ── Trajectory optimizer ──
    parser.add_argument(
        "--traj-opt", action="store_true",
        help="Enable trajectory optimiser (smoothness + comfort post-processing)",
    )
    parser.add_argument(
        "--traj-opt-smooth", type=float, default=1.0,
        help="Optimiser smoothness weight (default 1.0)",
    )
    parser.add_argument(
        "--traj-opt-deviation", type=float, default=0.1,
        help="Optimiser deviation weight (default 0.1)",
    )
    parser.add_argument(
        "--traj-opt-comfort", type=float, default=2.0,
        help="Optimiser comfort penalty weight (default 2.0)",
    )
    parser.add_argument(
        "--traj-opt-iter", type=int, default=50,
        help="Optimiser max iterations (default 50)",
    )
    parser.add_argument(
        "--no-retime", action="store_true",
        help="Disable Frenet retiming in trajectory optimiser",
    )
    parser.add_argument(
        "--retime-alpha", type=float, default=0.25,
        help="Retiming strength [0..1] (default 0.25)",
    )
    args = parser.parse_args()

    # Recording requires the display to be enabled
    if args.record and args.no_display:
        print("Warning: --record requires the display. Enabling display.")
        args.no_display = False

    config = AgentConfig(
        host=args.host,
        port=args.port,
        map_name=args.map,
        weather=args.weather,
        spawn_point_index=args.spawn,
        vehicle_filter=args.vehicle,
        use_dummy_model=args.dummy,
        model_name=args.model,
        max_speed_kmh=args.max_speed,
        min_speed_kmh=args.min_speed,
        steer_gain=args.steer_gain,
        steer_normalize_deg=args.steer_norm_deg,
        num_traj_samples=args.num_traj_samples,
        max_generation_length=args.max_gen_len,
        diffusion_steps=args.diffusion_steps,
        cam_resolution=args.cam_res,
        vlm_temperature=args.temperature,
        vlm_top_p=args.top_p,
        sim_fps=args.sim_fps,
        inference_interval=args.inference_interval,
        enable_display=not args.no_display,
        record_path=args.record,
        record_crf=args.crf,
        nav_enabled=not args.no_nav,
        use_cfg_nav=args.cfg_nav,
        cfg_nav_guidance_weight=args.cfg_nav_weight,
        nav_destination_index=args.nav_dest,
        debug_log_path=args.debug_log,
        traj_opt_enabled=args.traj_opt,
        traj_opt_smoothness_w=args.traj_opt_smooth,
        traj_opt_deviation_w=args.traj_opt_deviation,
        traj_opt_comfort_w=args.traj_opt_comfort,
        traj_opt_max_iter=args.traj_opt_iter,
        traj_opt_retime=not args.no_retime,
        traj_opt_retime_alpha=args.retime_alpha,
    )

    with CarlaAlpamayoAgent(config) as agent:
        agent.run(
            max_frames=args.frames,
            verbose=True,
        )


if __name__ == "__main__":
    main()

