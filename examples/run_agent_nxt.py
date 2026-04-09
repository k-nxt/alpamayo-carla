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
        "--vehicle", default="vehicle.tesla.model3",
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
        "--max-gen-len", type=int, default=256,
        help="VLM max generation tokens (default 256; use 64 for faster but weaker reasoning)",
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
        "--timing-log", action="store_true",
        help="Print per-inference timing breakdown (prep/model/post/total)",
    )
    parser.add_argument(
        "--loop-timing-log", action="store_true",
        help="Print per-loop timing breakdown (world tick/step/display/total)",
    )
    parser.add_argument(
        "--inference-interval", type=int, default=1,
        help="Run inference every N simulation ticks (default 1 → every tick)",
    )
    parser.add_argument(
        "--inference-interval-sec", type=float, default=None,
        help="Run inference every N wall-clock seconds (overrides --inference-interval)",
    )
    parser.add_argument(
        "--sim-fps", type=float, default=10.0,
        help="Simulation FPS (default 10.0 = 0.1s/tick, matching AR1 training)",
    )
    parser.add_argument(
        "--control-mode", type=str, default="legacy", choices=["legacy", "official-pid"],
        help="Control mode: legacy follower or CARLA official PID follower",
    )
    parser.add_argument(
        "--npc-vehicles", type=int, default=0, metavar="N",
        help="Number of NPC vehicles to spawn (default 0)",
    )
    parser.add_argument(
        "--npc-walkers", type=int, default=0, metavar="N",
        help="Number of NPC pedestrians to spawn (default 0)",
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Disable pygame dashboard window",
    )
    parser.add_argument(
        "--display-cam-downsample", type=int, default=1, metavar="N",
        help="Display-only camera downsample factor (1=no downsample, 2=half in each axis, ...)",
    )
    parser.add_argument(
        "--display-max-cameras", type=int, default=4, metavar="N",
        help="Maximum number of camera panels to render in display (0-4)",
    )
    parser.add_argument(
        "--display-no-camera-fetch", action="store_true",
        help="Do not fetch camera images for display (HUD/BEV text only)",
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
        "--no-camera-indices", action="store_true",
        help="Do not pass camera_indices into Alpamayo 1.5 prompt construction "
             "(for A/B testing)",
    )
    parser.add_argument(
        "--nav-dest", type=int, default=-1, metavar="IDX",
        help="Spawn-point index for route destination (-1 = random, auto-replan on arrival)",
    )
    parser.add_argument(
        "--nav-text", type=str, default=None, metavar="TEXT",
        help="Force a fixed navigation instruction text (1.5 only). "
             "If set, this overrides planner-generated navigation.",
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
    parser.add_argument(
        "--pid-lookahead-min", type=float, default=4.0,
        help="Official PID: minimum lookahead distance [m]",
    )
    parser.add_argument(
        "--pid-lookahead-max", type=float, default=12.0,
        help="Official PID: maximum lookahead distance [m]",
    )
    parser.add_argument(
        "--pid-lookahead-speed-gain", type=float, default=0.4,
        help="Official PID: lookahead increase per speed [m/(m/s)]",
    )
    parser.add_argument(
        "--pid-target-speed-min", type=float, default=10.0,
        help="Official PID: minimum target speed [km/h]",
    )
    parser.add_argument(
        "--pid-target-speed-max", type=float, default=35.0,
        help="Official PID: maximum target speed [km/h]",
    )
    parser.add_argument(
        "--pid-target-speed-extent-gain", type=float, default=0.5,
        help="Official PID: target speed gain from trajectory extent",
    )
    parser.add_argument("--pid-lat-kp", type=float, default=1.1, help="Official PID lateral Kp")
    parser.add_argument("--pid-lat-ki", type=float, default=0.02, help="Official PID lateral Ki")
    parser.add_argument("--pid-lat-kd", type=float, default=0.15, help="Official PID lateral Kd")
    parser.add_argument("--pid-lon-kp", type=float, default=0.6, help="Official PID longitudinal Kp")
    parser.add_argument("--pid-lon-ki", type=float, default=0.05, help="Official PID longitudinal Ki")
    parser.add_argument("--pid-lon-kd", type=float, default=0.0, help="Official PID longitudinal Kd")
    parser.add_argument(
        "--pid-max-throttle", type=float, default=0.35,
        help="Official PID max throttle",
    )
    parser.add_argument(
        "--pid-max-brake", type=float, default=1.0,
        help="Official PID max brake",
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
        control_mode=args.control_mode.replace("-", "_"),
        steer_gain=args.steer_gain,
        steer_normalize_deg=args.steer_norm_deg,
        num_traj_samples=args.num_traj_samples,
        max_generation_length=args.max_gen_len,
        diffusion_steps=args.diffusion_steps,
        cam_resolution=args.cam_res,
        vlm_temperature=args.temperature,
        vlm_top_p=args.top_p,
        timing_log=args.timing_log,
        loop_timing_log=args.loop_timing_log,
        sim_fps=args.sim_fps,
        num_npc_vehicles=args.npc_vehicles,
        num_npc_walkers=args.npc_walkers,
        inference_interval=args.inference_interval,
        inference_interval_sec=args.inference_interval_sec,
        enable_display=not args.no_display,
        display_camera_downsample=max(1, args.display_cam_downsample),
        display_max_cameras=max(0, min(4, args.display_max_cameras)),
        display_fetch_cameras=not args.display_no_camera_fetch,
        record_path=args.record,
        record_crf=args.crf,
        nav_enabled=not args.no_nav,
        use_cfg_nav=args.cfg_nav,
        cfg_nav_guidance_weight=args.cfg_nav_weight,
        use_camera_indices=not args.no_camera_indices,
        nav_destination_index=args.nav_dest,
        nav_text_override=args.nav_text,
        debug_log_path=args.debug_log,
        traj_opt_enabled=args.traj_opt,
        traj_opt_smoothness_w=args.traj_opt_smooth,
        traj_opt_deviation_w=args.traj_opt_deviation,
        traj_opt_comfort_w=args.traj_opt_comfort,
        traj_opt_max_iter=args.traj_opt_iter,
        traj_opt_retime=not args.no_retime,
        traj_opt_retime_alpha=args.retime_alpha,
        pid_lookahead_min_m=args.pid_lookahead_min,
        pid_lookahead_max_m=args.pid_lookahead_max,
        pid_lookahead_speed_gain=args.pid_lookahead_speed_gain,
        pid_target_speed_min_kmh=args.pid_target_speed_min,
        pid_target_speed_max_kmh=args.pid_target_speed_max,
        pid_target_speed_extent_gain=args.pid_target_speed_extent_gain,
        pid_lat_kp=args.pid_lat_kp,
        pid_lat_ki=args.pid_lat_ki,
        pid_lat_kd=args.pid_lat_kd,
        pid_lon_kp=args.pid_lon_kp,
        pid_lon_ki=args.pid_lon_ki,
        pid_lon_kd=args.pid_lon_kd,
        pid_max_throttle=args.pid_max_throttle,
        pid_max_brake=args.pid_max_brake,
    )

    with CarlaAlpamayoAgent(config) as agent:
        agent.run(
            max_frames=args.frames,
            verbose=True,
        )


if __name__ == "__main__":
    main()

