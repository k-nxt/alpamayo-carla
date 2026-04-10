"""
Navigation Planner for CARLA Alpamayo Agent (NXT)

Generates natural-language navigation instructions compatible with
Alpamayo 1.5's navigation-conditioned trajectory prediction.

Uses ``agents.navigation.global_route_planner.GlobalRoutePlanner``
(CARLA 0.9.16, copied into ``src/agents/``).

Public API
----------
NavPlanner (Agent mode)
    Plans a route via ``GlobalRoutePlanner.trace_route`` and emits
    instructions by scanning ahead for LEFT / RIGHT / STRAIGHT actions.

nav_text_from_traffic_manager (Observer mode)
    Converts ``TrafficManager.get_all_actions()`` output into a
    navigation instruction string.  No route planning needed.
"""

from __future__ import annotations

import random
import time
from typing import Optional, List, Tuple

import carla

from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.local_planner import RoadOption


# ------------------------------------------------------------------
# RoadOption → natural-language mapping
# ------------------------------------------------------------------
_ROAD_OPTION_TEXT = {
    RoadOption.LEFT: "Turn left",
    RoadOption.RIGHT: "Turn right",
    RoadOption.STRAIGHT: "Continue straight",
}

# TrafficManager.get_next_action / get_all_actions returns strings
_TM_ACTION_TEXT = {
    "Left": "Turn left",
    "Right": "Turn right",
    "Straight": "Continue straight",
}

_TURN_OPTIONS = {RoadOption.LEFT, RoadOption.RIGHT, RoadOption.STRAIGHT}


def _road_option_label(opt: RoadOption) -> Optional[str]:
    return _ROAD_OPTION_TEXT.get(opt)


# ==================================================================
# Agent mode — GlobalRoutePlanner-based NavPlanner
# ==================================================================

class NavPlanner:
    """Route-based navigation planner for Agent mode.

    Usage::

        planner = NavPlanner(world)
        planner.set_destination(vehicle.get_location(), dest_location)

        # Each tick:
        nav_text = planner.get_instruction(vehicle.get_transform())
        if planner.route_complete:
            planner.set_random_destination(vehicle.get_location())
    """

    def __init__(
        self,
        world: carla.World,
        sampling_resolution: float = 2.0,
    ):
        self._world = world
        self._map = world.get_map()
        self._grp = GlobalRoutePlanner(self._map, sampling_resolution)
        self._sampling_resolution = sampling_resolution

        self._route: List[Tuple[carla.Waypoint, RoadOption]] = []
        self._route_idx: int = 0
        self._destination: Optional[carla.Location] = None
        self._stagnant_calls: int = 0
        self._last_reroute_time: float = -1e9
        self._off_route_distance_m: float = 15.0
        self._stagnant_replan_calls: int = 30
        self._reroute_cooldown_sec: float = 3.0

    # ------------------------------------------------------------------
    # Route setup
    # ------------------------------------------------------------------

    def set_destination(
        self, start: carla.Location, destination: carla.Location,
    ) -> None:
        """Plan a route from *start* to *destination*."""
        self._route = self._grp.trace_route(start, destination)
        self._route_idx = 0
        self._destination = destination
        self._stagnant_calls = 0
        print(
            f"NavPlanner: route planned "
            f"({len(self._route)} waypoints → "
            f"dest≈{destination.x:.0f},{destination.y:.0f})"
        )

    def set_random_destination(
        self, start: carla.Location, min_distance: float = 100.0,
    ) -> Optional[carla.Location]:
        """Pick a random spawn point as destination and plan a route."""
        spawns = self._map.get_spawn_points()
        candidates = [
            sp for sp in spawns
            if sp.location.distance(start) >= min_distance
        ]
        if not candidates:
            candidates = spawns
        dest = random.choice(candidates).location
        self.set_destination(start, dest)
        return dest

    # ------------------------------------------------------------------
    # Instruction generation
    # ------------------------------------------------------------------

    def get_instruction(
        self, vehicle_transform: carla.Transform,
    ) -> Optional[str]:
        """Return a nav instruction string, or ``None`` if route is empty."""
        if not self._route:
            return None

        veh_loc = vehicle_transform.location
        min_dist, advanced = self._advance(veh_loc)
        self._maybe_reroute(veh_loc, min_dist=min_dist, progressed=advanced)

        if self.route_complete:
            return None

        for i in range(self._route_idx, len(self._route)):
            wp, opt = self._route[i]
            if opt not in _TURN_OPTIONS:
                continue

            label = _road_option_label(opt)
            if label is None:
                continue

            dist = veh_loc.distance(wp.transform.location)
            return f"{label} in {dist:.0f}m"

        return "Continue straight"

    # ------------------------------------------------------------------
    # Route tracking
    # ------------------------------------------------------------------

    @property
    def route_complete(self) -> bool:
        if not self._route:
            return True
        return self._route_idx >= len(self._route) - 5

    @property
    def route_remaining(self) -> int:
        return max(0, len(self._route) - self._route_idx)

    @property
    def destination(self) -> Optional[carla.Location]:
        return self._destination

    def _advance(self, location: carla.Location) -> None:
        """Move route index forward to nearest upcoming waypoint.

        Returns:
            (min_dist, progressed) where min_dist is the nearest distance
            to searched route points, and progressed indicates whether
            route_idx advanced this call.
        """
        min_dist = float("inf")
        min_idx = self._route_idx
        search_end = min(self._route_idx + 60, len(self._route))

        for i in range(self._route_idx, search_end):
            wp, _ = self._route[i]
            dist = location.distance(wp.transform.location)
            if dist < min_dist:
                min_dist = dist
                min_idx = i

        progressed = False
        if min_idx > self._route_idx:
            self._route_idx = min_idx
            progressed = True

        return min_dist, progressed

    def _maybe_reroute(
        self,
        location: carla.Location,
        min_dist: float,
        progressed: bool,
    ) -> None:
        """Replan to destination if we appear off-route or stalled."""
        if self._destination is None or not self._route:
            return

        if progressed:
            self._stagnant_calls = 0
        else:
            self._stagnant_calls += 1

        off_route = min_dist > self._off_route_distance_m
        stagnant = self._stagnant_calls >= self._stagnant_replan_calls
        if not (off_route or stagnant):
            return

        now = time.monotonic()
        if now - self._last_reroute_time < self._reroute_cooldown_sec:
            return

        new_route = self._grp.trace_route(location, self._destination)
        if not new_route:
            return

        reason = "off-route" if off_route else "stalled"
        self._route = new_route
        self._route_idx = 0
        self._stagnant_calls = 0
        self._last_reroute_time = now
        print(
            f"NavPlanner: reroute ({reason}, nearest={min_dist:.1f}m), "
            f"new_len={len(self._route)}"
        )


# ==================================================================
# Observer mode — TrafficManager-based nav text
# ==================================================================

def nav_text_from_traffic_manager(
    traffic_manager: carla.TrafficManager,
    vehicle: carla.Actor,
) -> Optional[str]:
    """Generate a nav instruction from the TrafficManager's planned actions.

    Calls ``traffic_manager.get_all_actions(vehicle)`` and returns the
    first actionable instruction (Left / Right / Straight) with distance.

    Returns ``None`` if no meaningful action is found.
    """
    try:
        actions = traffic_manager.get_all_actions(vehicle)
    except RuntimeError:
        return None

    if not actions:
        return None

    veh_loc = vehicle.get_location()

    for action_str, waypoint in actions:
        label = _TM_ACTION_TEXT.get(action_str)
        if label is None:
            continue
        dist = veh_loc.distance(waypoint.transform.location)
        return f"{label} in {dist:.0f}m"

    return None
