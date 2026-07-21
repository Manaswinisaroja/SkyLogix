"""
drone_agent.py — SkyLogix UAV Research Platform v3.0
═══════════════════════════════════════════════════════════════════
Real robotics control algorithms (fully self-contained, no numpy req):

1.  PID Controller          — velocity / altitude regulation (cascaded)
2.  A* Path Planner         — grid-based, 8-directional, NFZ-aware
3.  Kalman Filter (6-DOF)   — GPS + IMU sensor fusion (pure Python)
4.  ORCA Collision Avoidance— multi-drone reciprocal avoidance
5.  MAVLink-style Protocol  — structured heartbeat / mission messages
6.  Fleet Coordinator       — Hungarian-inspired greedy assignment
7.  Swarm State Machine     — full autonomous mission lifecycle

Public API consumed by flask_server_robotics.py:
  - GPSCoord, WeatherData, Mission, FlightState, MissionStatus
  - DroneAgent
  - FleetManager   (replaces FleetCoordinator for Flask compat)
"""

import math
import time
import threading
import random
import json
import heapq
import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from collections import defaultdict
from enum import Enum

log = logging.getLogger("DroneAgent")

# ══════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════

EARTH_R      = 6_371_000.0   # metres
DEG2RAD      = math.pi / 180
RAD2DEG      = 180 / math.pi
MAX_SPEED    = 15.0           # m/s cruise
MAX_VSPEED   = 3.0            # m/s vertical
CRUISE_ALT   = 80.0           # metres AGL
TAKEOFF_ALT  = 20.0           # initial climb target
CELL_SIZE    = 0.0002         # ~22 m grid cells (movement resolution)
ASTAR_CELL   = 0.0004         # ~44 m A* grid cells (coarser for faster search)
SOFT_NFZ_ALT = 150.0          # metres — altitude boost over soft NFZ zones
NFZ_OVER_ALT = 160.0          # metres — altitude to clear hard NFZ
ORCA_RADIUS  = 15.0           # collision radius per drone (m)
ORCA_HORIZON = 8.0            # look-ahead window (s)
TICK_HZ      = 10             # simulation frequency
DT           = 1.0 / TICK_HZ  # seconds per tick
GPS_SIGMA    = 3.0            # GPS noise standard deviation (m)
SIM_SPEEDUP  = 8              # visual speedup multiplier (matches JS frontend — was 80, reduced for visible flight)


# ══════════════════════════════════════════════════════════════════
#  DATA CLASSES  (Flask-compatible public types)
# ══════════════════════════════════════════════════════════════════

@dataclass
class GPSCoord:
    lat: float
    lng: float
    alt: float = 0.0

    def distance_to(self, other: "GPSCoord") -> float:
        """Great-circle distance in metres."""
        return haversine(self.lat, self.lng, other.lat, other.lng)

    def to_dict(self) -> dict:
        return {"lat": round(self.lat, 7), "lng": round(self.lng, 7), "alt": round(self.alt, 1)}


@dataclass
class WeatherData:
    wind_speed_kmh:      float = 0.0
    wind_direction_deg:  float = 0.0
    temperature_c:       float = 25.0
    humidity_pct:        float = 50.0
    visibility_km:       float = 10.0
    weather_code:        int   = 0
    is_dangerous:        bool  = False
    is_caution:          bool  = False

    @property
    def wind_speed_ms(self) -> float:
        return self.wind_speed_kmh / 3.6

    @property
    def speed_modifier(self) -> float:
        if self.is_dangerous:
            return 0.0
        if self.is_caution or self.wind_speed_kmh > 25:
            return 0.70
        if self.wind_speed_kmh > 15:
            return 0.85
        return 1.0

    @property
    def battery_drain_modifier(self) -> float:
        if self.is_caution:
            return 1.30
        if self.wind_speed_kmh > 15:
            return 1.15
        if self.humidity_pct > 80:
            return 1.10
        return 1.0

    @property
    def wind_drift(self) -> float:
        return self.wind_speed_ms * 0.1


class FlightState(Enum):
    IDLE           = "IDLE"
    TAKEOFF        = "TAKEOFF"
    CLIMB          = "CLIMB"
    CRUISE_PICKUP  = "CRUISE_PICKUP"
    HOVER_PICKUP   = "HOVER_PICKUP"
    CRUISE_DELIVER = "CRUISE_DELIVER"
    HOVER_DELIVER  = "HOVER_DELIVER"
    RTB            = "RTB"
    LAND           = "LAND"
    CHARGING       = "CHARGING"
    FAULT_RTH      = "FAULT_RTH"
    FAULT_LAND     = "FAULT_LAND"


class MissionStatus(Enum):
    PENDING    = "pending"
    INFLIGHT   = "inflight"
    DELIVERED  = "delivered"
    ABORTED    = "aborted"


@dataclass
class Mission:
    id:           int
    pickup:       GPSCoord
    delivery:     GPSCoord
    pickup_name:  str   = "Pickup"
    deliver_name: str   = "Destination"
    payload_kg:   float = 1.0
    priority:     str   = "standard"   # standard / express / urgent
    status:       MissionStatus = MissionStatus.PENDING
    assigned_to:  Optional[str] = None
    created_at:   float = field(default_factory=time.time)
    started_at:   Optional[float] = None
    delivered_at: Optional[float] = None

    @property
    def priority_rank(self) -> int:
        return {"urgent": 0, "express": 1, "standard": 2}.get(self.priority, 2)

    @property
    def elapsed_s(self) -> Optional[float]:
        if self.started_at is None:
            return None
        end = self.delivered_at or time.time()
        return round(end - self.started_at, 1)

    def to_dict(self) -> dict:
        return {
            "id":           self.id,
            "pickup":       self.pickup.to_dict(),
            "delivery":     self.delivery.to_dict(),
            "pickup_name":  self.pickup_name,
            "deliver_name": self.deliver_name,
            "payload_kg":   self.payload_kg,
            "priority":     self.priority,
            "status":       self.status.value,
            "assigned_to":  self.assigned_to,
            "elapsed_s":    self.elapsed_s,
        }


# ══════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════

def haversine(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Great-circle distance in metres."""
    dlat = (lat2 - lat1) * DEG2RAD
    dlng = (lng2 - lng1) * DEG2RAD
    a = (math.sin(dlat / 2) ** 2
         + math.cos(lat1 * DEG2RAD) * math.cos(lat2 * DEG2RAD) * math.sin(dlng / 2) ** 2)
    return 2 * EARTH_R * math.asin(math.sqrt(max(0.0, a)))


def bearing(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """True bearing in degrees (0 = North, 90 = East)."""
    y = math.sin((lng2 - lng1) * DEG2RAD) * math.cos(lat2 * DEG2RAD)
    x = (math.cos(lat1 * DEG2RAD) * math.sin(lat2 * DEG2RAD)
         - math.sin(lat1 * DEG2RAD) * math.cos(lat2 * DEG2RAD) * math.cos((lng2 - lng1) * DEG2RAD))
    return (math.atan2(y, x) * RAD2DEG + 360) % 360


def move_along_bearing(lat: float, lng: float, dist_m: float, bearing_deg: float) -> Tuple[float, float]:
    """Compute new (lat, lng) after moving dist_m metres on a bearing."""
    d  = dist_m / EARTH_R
    b  = bearing_deg * DEG2RAD
    la = lat * DEG2RAD
    lo = lng * DEG2RAD
    la2 = math.asin(math.sin(la) * math.cos(d) + math.cos(la) * math.sin(d) * math.cos(b))
    lo2 = lo + math.atan2(math.sin(b) * math.sin(d) * math.cos(la),
                          math.cos(d) - math.sin(la) * math.sin(la2))
    return la2 * RAD2DEG, lo2 * RAD2DEG


def gauss(sigma: float) -> float:
    """Box-Muller Gaussian sample (no numpy needed)."""
    u1 = max(1e-10, random.random())
    u2 = random.random()
    return sigma * math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)


# ══════════════════════════════════════════════════════════════════
#  1. PID CONTROLLER
# ══════════════════════════════════════════════════════════════════

class PIDController:
    """
    Standard discrete PID with anti-windup clamp.

        u(t) = Kp·e + Ki·∫e dt + Kd·(de/dt)

    Used for:
      - Lateral velocity control  (outer loop → position error → velocity cmd)
      - Altitude hold             (position PID on altitude)
      - Speed regulation          (inner loop limiter)
    """

    def __init__(self, kp: float, ki: float, kd: float,
                 out_min: float = -1.0, out_max: float = 1.0,
                 integral_limit: float = 50.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.out_min, self.out_max = out_min, out_max
        self.integral_limit = integral_limit
        self._integral   = 0.0
        self._prev_error = 0.0
        self._prev_time  = time.time()

    def reset(self) -> None:
        self._integral   = 0.0
        self._prev_error = 0.0
        self._prev_time  = time.time()

    def compute(self, setpoint: float, measurement: float) -> float:
        now = time.time()
        dt  = max(now - self._prev_time, 1e-6)
        self._prev_time = now

        error = setpoint - measurement

        # Integrator with anti-windup clamp
        self._integral += error * dt
        self._integral  = max(-self.integral_limit,
                               min(self.integral_limit, self._integral))

        derivative      = (error - self._prev_error) / dt
        self._prev_error = error

        u = self.kp * error + self.ki * self._integral + self.kd * derivative
        return max(self.out_min, min(self.out_max, u))


class DroneFlightController:
    """
    Cascaded flight controller per drone.

    Outer loop: (lat/lng/alt error) → velocity setpoint  [position PID]
    Inner loop: velocity → throttle/attitude              [speed limiter PID]
    Altitude  : separate Z-axis PID

    Returns velocity commands (vx_ms, vy_ms, vz_ms) in m/s
    where vx = North, vy = East, vz = Up.
    """

    def __init__(self, drone_id: str):
        self.drone_id = drone_id
        # Outer position loops (lat/lng error in metres → velocity)
        self.pid_lat = PIDController(kp=1.8,  ki=0.05, kd=0.40,
                                     out_min=-MAX_SPEED, out_max=MAX_SPEED)
        self.pid_lng = PIDController(kp=1.8,  ki=0.05, kd=0.40,
                                     out_min=-MAX_SPEED, out_max=MAX_SPEED)
        # Altitude PID
        self.pid_alt = PIDController(kp=2.5,  ki=0.10, kd=0.80,
                                     out_min=-MAX_VSPEED, out_max=MAX_VSPEED)
        # Speed-limiting inner PID
        self.pid_spd = PIDController(kp=1.2,  ki=0.02, kd=0.30,
                                     out_min=0.0, out_max=MAX_SPEED)

    def compute_velocity(self,
                         cur_lat: float, cur_lng: float, cur_alt: float,
                         tgt_lat: float, tgt_lng: float, tgt_alt: float
                         ) -> Tuple[float, float, float]:
        """
        Convert position error to velocity commands.

        Returns:
            (vx, vy, vz) in m/s  — North, East, Up
        """
        # Convert angular error to metres
        dlat_m = (tgt_lat - cur_lat) * EARTH_R * DEG2RAD
        dlng_m = ((tgt_lng - cur_lng) * EARTH_R
                  * math.cos(cur_lat * DEG2RAD) * DEG2RAD)
        dalt_m = tgt_alt - cur_alt

        # PID outputs in m/s
        vx = self.pid_lat.compute(0.0, -dlat_m)
        vy = self.pid_lng.compute(0.0, -dlng_m)
        vz = self.pid_alt.compute(0.0, -dalt_m)

        # Scale horizontal speed to not exceed MAX_SPEED
        speed_2d = math.sqrt(vx ** 2 + vy ** 2)
        if speed_2d > MAX_SPEED:
            scale = MAX_SPEED / speed_2d
            vx *= scale
            vy *= scale

        return vx, vy, vz

    def reset(self) -> None:
        self.pid_lat.reset()
        self.pid_lng.reset()
        self.pid_alt.reset()


# ══════════════════════════════════════════════════════════════════
#  2. A* PATH PLANNER
# ══════════════════════════════════════════════════════════════════

class AStarPlanner:
    """
    Grid-based A* path planner on a geographic lat/lng grid.

    Grid is offset-relative to the start point so all coordinates remain
    near the actual location (not drifting toward 0,0 / Gulf of Guinea).

    Grid cell size: ASTAR_CELL degrees (~44 m per cell at equator).
    Movement      : 8-directional (diagonal allowed).
    Heuristic     : Euclidean distance in grid units (admissible).
    No-fly zones  : modelled as blocked cells with a 15 m buffer.
    Path smoothing: direction-change filter + density guarantee.

    Returns a list of (lat, lng) waypoints from start to goal.
    Falls back to straight-line + NFZ deflection if budget exhausted.
    """

    def __init__(self, nfz_list: Optional[List[dict]] = None):
        self.nfz_list: List[dict] = nfz_list or []

    # ── Grid helpers (offset from start) ─────────────────────────

    @staticmethod
    def _to_grid(lat: float, lng: float,
                 origin_lat: float, origin_lng: float) -> Tuple[int, int]:
        return (round((lat - origin_lat) / ASTAR_CELL),
                round((lng - origin_lng) / ASTAR_CELL))

    @staticmethod
    def _from_grid(gx: int, gy: int,
                   origin_lat: float, origin_lng: float) -> Tuple[float, float]:
        return (origin_lat + gx * ASTAR_CELL,
                origin_lng + gy * ASTAR_CELL)

    def _is_blocked(self, lat: float, lng: float) -> bool:
        for nfz in self.nfz_list:
            if haversine(lat, lng, nfz["lat"], nfz["lng"]) < nfz["radius_m"] + 15:
                return True
        return False

    @staticmethod
    def _heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        # Euclidean distance in grid units — admissible heuristic
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    # ── Core search ───────────────────────────────────────────────

    def plan(self,
             start_lat: float, start_lng: float,
             goal_lat:  float, goal_lng:  float,
             max_cells: int = 15000) -> List[Tuple[float, float]]:
        """
        Run A* from (start_lat, start_lng) to (goal_lat, goal_lng).
        Grid is offset so start = (0, 0) in grid space.
        """
        # Use start as the grid origin
        origin_lat, origin_lng = start_lat, start_lng

        start = (0, 0)
        goal  = self._to_grid(goal_lat, goal_lng, origin_lat, origin_lng)

        if start == goal:
            return [(goal_lat, goal_lng)]

        dist_m = haversine(start_lat, start_lng, goal_lat, goal_lng)
        budget = min(max_cells, max(800, int(dist_m / (ASTAR_CELL * EARTH_R * DEG2RAD)) * 10))

        open_heap: List = []
        heapq.heappush(open_heap, (0.0, 0.0, start, None))

        came_from: Dict[Tuple, Optional[Tuple]] = {}
        g_score: Dict[Tuple, float] = defaultdict(lambda: float("inf"))
        g_score[start] = 0.0
        visited: set = set()
        iters = 0

        DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1),
                (-1,-1), (-1, 1), (1,-1), (1, 1)]
        COSTS = [1.0, 1.0, 1.0, 1.0, 1.414, 1.414, 1.414, 1.414]

        while open_heap and iters < budget:
            f, g, current, parent = heapq.heappop(open_heap)
            iters += 1

            if current in visited:
                continue
            visited.add(current)
            came_from[current] = parent

            if current == goal:
                # Reconstruct path in real coordinates
                path: List[Tuple[float, float]] = []
                node: Optional[Tuple] = goal
                while node is not None:
                    lat, lng = self._from_grid(node[0], node[1], origin_lat, origin_lng)
                    path.append((lat, lng))
                    node = came_from.get(node)
                path.reverse()
                # Ensure goal is exactly the requested coordinate
                if path[-1] != (goal_lat, goal_lng):
                    path.append((goal_lat, goal_lng))
                log.debug("A* solved in %d iters, %d waypoints", iters, len(path))
                return self._smooth_path(path)

            for i, (dx, dy) in enumerate(DIRS):
                nb = (current[0] + dx, current[1] + dy)
                if nb in visited:
                    continue
                nb_lat, nb_lng = self._from_grid(nb[0], nb[1], origin_lat, origin_lng)
                if self._is_blocked(nb_lat, nb_lng):
                    continue
                tentative_g = g + COSTS[i]
                if tentative_g < g_score[nb]:
                    g_score[nb] = tentative_g
                    h = self._heuristic(nb, goal)
                    heapq.heappush(open_heap, (tentative_g + h, tentative_g, nb, current))

        log.warning("A* budget exhausted (%d iters) — falling back to deflected straight line", iters)
        return self._straight_with_deflection(start_lat, start_lng, goal_lat, goal_lng)

    # ── Path smoothing ────────────────────────────────────────────

    def _smooth_path(self, path: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Light smoothing: keep direction-change waypoints, skip only truly
        collinear points. Preserves enough points for visible A* curves.
        """
        if len(path) <= 3:
            return path
        smoothed = [path[0]]
        for i in range(1, len(path) - 1):
            ax, ay = path[i - 1]
            bx, by = path[i]
            cx, cy = path[i + 1]
            # Keep if cross-product shows a real direction change
            cross = abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))
            if cross > 1e-10:
                smoothed.append(path[i])
        smoothed.append(path[-1])
        # Ensure minimum 6 points so the curve is visually smooth
        if len(smoothed) < 6:
            dense = []
            for i in range(len(smoothed) - 1):
                dense.append(smoothed[i])
                dense.append(((smoothed[i][0] + smoothed[i+1][0]) / 2,
                               (smoothed[i][1] + smoothed[i+1][1]) / 2))
            dense.append(smoothed[-1])
            return dense
        return smoothed

    # ── Fallback ──────────────────────────────────────────────────

    def _straight_with_deflection(self,
                                   s_lat: float, s_lng: float,
                                   e_lat: float, e_lng: float,
                                   steps: int = 24) -> List[Tuple[float, float]]:
        """Straight line with potential-field repulsion around NFZs."""
        pts: List[Tuple[float, float]] = []
        for i in range(steps + 1):
            t   = i / steps
            la  = s_lat + (e_lat - s_lat) * t
            lo  = s_lng + (e_lng - s_lng) * t
            for nfz in self.nfz_list:
                d = haversine(la, lo, nfz["lat"], nfz["lng"])
                if d < nfz["radius_m"] + 40:
                    ang   = math.atan2(la - nfz["lat"], lo - nfz["lng"])
                    push  = (nfz["radius_m"] + 60) / (EARTH_R * DEG2RAD)
                    la   += push * math.cos(ang)
                    lo   += push * math.sin(ang)
            pts.append((la, lo))
        return pts


# ══════════════════════════════════════════════════════════════════
#  2b. THETA* ADVANCED A* PATH PLANNER (any-angle, line-of-sight)
# ══════════════════════════════════════════════════════════════════

class ThetaStarPlanner(AStarPlanner):
    """
    Theta* — any-angle extension of A* using line-of-sight checks.

    Key difference from Basic A*:
      When relaxing a neighbour, Theta* checks if the *grandparent*
      has direct line-of-sight to the neighbour. If yes, it skips
      the intermediate node entirely, allowing paths at any angle
      rather than being constrained to 45° grid increments.

    Result: shorter, smoother paths with fewer waypoints. Slightly
    more compute per node (LoS walk) but far fewer waypoints overall.

    Algorithm: Nash et al., "Theta*: Any-Angle Path Planning on Grids"
               AAAI 2007 / Journal of Artificial Intelligence Research 2010
    """

    def _line_of_sight(self,
                       ax: int, ay: int,
                       bx: int, by: int,
                       origin_lat: float, origin_lng: float) -> bool:
        """
        Walk the grid segment (ax,ay)→(bx,by) checking for NFZ blocks.
        Uses Bresenham-style sampling at twice the cell density.
        """
        steps = max(abs(bx - ax), abs(by - ay)) * 2 or 1
        for t in range(steps + 1):
            frac = t / steps
            lat = origin_lat + (ax + (bx - ax) * frac) * ASTAR_CELL
            lng = origin_lng + (ay + (by - ay) * frac) * ASTAR_CELL
            if self._is_blocked(lat, lng):
                return False
        return True

    def plan(self,
             start_lat: float, start_lng: float,
             goal_lat:  float, goal_lng:  float,
             max_cells: int = 15000) -> List[Tuple[float, float]]:
        """
        Theta* search from start to goal.
        Falls back to straight deflection if budget exhausted.
        """
        origin_lat, origin_lng = start_lat, start_lng

        start = (0, 0)
        goal  = self._to_grid(goal_lat, goal_lng, origin_lat, origin_lng)

        if start == goal:
            return [(goal_lat, goal_lng)]

        dist_m = haversine(start_lat, start_lng, goal_lat, goal_lng)
        budget = min(max_cells, max(800, int(dist_m / (ASTAR_CELL * EARTH_R * DEG2RAD)) * 10))

        # parent map: node → parent node key (or None for start)
        parent: Dict[Tuple, Optional[Tuple]] = {start: None}
        g_score: Dict[Tuple, float] = defaultdict(lambda: float("inf"))
        g_score[start] = 0.0
        visited: set = set()

        # heap entries: (f, g, node)
        open_heap: List = []
        heapq.heappush(open_heap, (self._heuristic(start, goal), 0.0, start))

        DIRS  = [(-1, 0), (1, 0), (0, -1), (0, 1),
                 (-1,-1), (-1, 1), (1,-1), (1, 1)]
        COSTS = [1.0, 1.0, 1.0, 1.0, 1.414, 1.414, 1.414, 1.414]
        iters = 0

        while open_heap and iters < budget:
            f, g, current = heapq.heappop(open_heap)
            iters += 1

            if current in visited:
                continue
            visited.add(current)

            if current == goal:
                # Reconstruct path via parent map
                path: List[Tuple[float, float]] = []
                node: Optional[Tuple] = goal
                while node is not None:
                    lat, lng = self._from_grid(node[0], node[1], origin_lat, origin_lng)
                    path.append((lat, lng))
                    node = parent.get(node)
                path.reverse()
                if path[-1] != (goal_lat, goal_lng):
                    path.append((goal_lat, goal_lng))
                log.debug("Theta* solved in %d iters, %d waypoints", iters, len(path))
                return self._smooth_path(path)

            cur_par = parent.get(current)  # grandparent of neighbours

            for i, (dx, dy) in enumerate(DIRS):
                nb = (current[0] + dx, current[1] + dy)
                if nb in visited:
                    continue
                nb_lat, nb_lng = self._from_grid(nb[0], nb[1], origin_lat, origin_lng)
                if self._is_blocked(nb_lat, nb_lng):
                    continue

                # Default: path through current node
                best_g = g + COSTS[i]
                best_par = current

                # Theta* relaxation: try path through grandparent if LoS exists
                if cur_par is not None:
                    pp = cur_par
                    pp_g = g_score[pp]
                    ddx = nb[0] - pp[0]
                    ddy = nb[1] - pp[1]
                    direct_cost = math.sqrt(ddx * ddx + ddy * ddy)
                    g_via_par = pp_g + direct_cost
                    if (g_via_par < best_g and
                            self._line_of_sight(pp[0], pp[1], nb[0], nb[1],
                                                origin_lat, origin_lng)):
                        best_g   = g_via_par
                        best_par = pp

                if best_g < g_score[nb]:
                    g_score[nb] = best_g
                    parent[nb]  = best_par
                    h = self._heuristic(nb, goal)
                    heapq.heappush(open_heap, (best_g + h, best_g, nb))

        log.warning("Theta* budget exhausted (%d iters) — falling back", iters)
        return self._straight_with_deflection(start_lat, start_lng, goal_lat, goal_lng)



class KalmanFilter6DOF:
    """
    6-state discrete Kalman filter for drone state estimation.

    State vector:   x = [lat, lng, alt, v_lat, v_lng, v_alt]ᵀ

    Measurement:    GPS gives (lat, lng, alt) with ~3 m noise.
    Prediction:     Constant-velocity model + acceleration inputs from IMU.

    Equations:
        Predict:  x̂_{k|k-1} = F · x̂_{k-1|k-1}
                  P_{k|k-1}  = F · P_{k-1|k-1} · Fᵀ + Q
        Update:   y  = z - H · x̂_{k|k-1}          (innovation)
                  S  = H · P · Hᵀ + R               (innov. covariance)
                  K  = P · Hᵀ · S⁻¹                 (Kalman gain)
                  x̂  = x̂ + K · y
                  P  = (I - K·H) · P
    """

    def __init__(self, lat0: float, lng0: float, alt0: float):
        # State [lat, lng, alt, v_lat, v_lng, v_alt]
        self._x = [lat0, lng0, alt0, 0.0, 0.0, 0.0]

        # 6×6 state transition (constant-velocity)
        # x_new = F · x_old  →  pos += vel * dt
        self._F = self._eye(6)
        self._F[0][3] = DT
        self._F[1][4] = DT
        self._F[2][5] = DT

        # Covariance P (6×6)
        self._P = self._diag([1e-8, 1e-8, 1.0, 1e-4, 1e-4, 0.1])

        # Process noise Q (vibration, wind)
        self._Q = self._diag([1e-10, 1e-10, 0.01, 1e-6, 1e-6, 0.001])

        # GPS measurement noise R (σ_pos ≈ 3 m expressed in degrees/m)
        self._R = self._diag([2e-9, 2e-9, 1.0])

        # Observation matrix H (3×6): GPS sees lat, lng, alt only
        self._H = [[0.0] * 6 for _ in range(3)]
        self._H[0][0] = 1.0  # lat
        self._H[1][1] = 1.0  # lng
        self._H[2][2] = 1.0  # alt

    # ── Public interface ──────────────────────────────────────────

    def predict(self,
                accel_lat: float = 0.0,
                accel_lng: float = 0.0,
                accel_alt: float = 0.0) -> None:
        """IMU predict step."""
        # Inject acceleration into velocity states
        self._x[3] += accel_lat * DT
        self._x[4] += accel_lng * DT
        self._x[5] += accel_alt * DT
        # Propagate state
        self._x = self._mv(self._F, self._x)
        # Propagate covariance  P = F·P·Fᵀ + Q
        FP  = self._mm(self._F, self._P)
        FPFt = self._mm(FP, self._transpose(self._F))
        self._P = self._madd(FPFt, self._Q)

    def update_gps(self, gps_lat: float, gps_lng: float, gps_alt: float) -> None:
        """GPS measurement update."""
        z = [gps_lat, gps_lng, gps_alt]
        # Innovation  y = z - H·x
        Hx = self._mv(self._H, self._x)
        y  = [z[i] - Hx[i] for i in range(3)]
        # Innovation covariance  S = H·P·Hᵀ + R
        HP   = self._mm(self._H, self._P)
        HPHt = self._mm(HP, self._transpose(self._H))
        S    = self._madd(HPHt, self._R)
        # Kalman gain  K = P·Hᵀ·S⁻¹
        PHt = self._mm(self._P, self._transpose(self._H))
        K   = self._mm(PHt, self._inv3(S))   # 6×3
        # State update  x += K·y
        Ky = self._mv(K, y)
        self._x = [self._x[i] + Ky[i] for i in range(6)]
        # Covariance update  P = (I - K·H)·P
        KH     = self._mm(K, self._H)
        ImKH   = self._msub(self._eye(6), KH)
        self._P = self._mm(ImKH, self._P)

    @property
    def position(self) -> Tuple[float, float, float]:
        return self._x[0], self._x[1], self._x[2]

    @property
    def velocity(self) -> Tuple[float, float, float]:
        return self._x[3], self._x[4], self._x[5]

    @property
    def uncertainty_m(self) -> float:
        """Horizontal position uncertainty in metres."""
        return math.sqrt(max(0, self._P[0][0] + self._P[1][1])) * EARTH_R * DEG2RAD

    # ── Pure-Python matrix helpers (no numpy) ────────────────────

    @staticmethod
    def _eye(n: int) -> List[List[float]]:
        return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    @staticmethod
    def _diag(v: List[float]) -> List[List[float]]:
        n = len(v)
        return [[v[i] if i == j else 0.0 for j in range(n)] for i in range(n)]

    @staticmethod
    def _transpose(A: List[List[float]]) -> List[List[float]]:
        rows, cols = len(A), len(A[0])
        return [[A[r][c] for r in range(rows)] for c in range(cols)]

    @staticmethod
    def _mm(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
        """Matrix multiplication A × B."""
        ra, ca = len(A), len(A[0])
        rb, cb = len(B), len(B[0])
        assert ca == rb, f"Shape mismatch {ra}×{ca} × {rb}×{cb}"
        C = [[0.0] * cb for _ in range(ra)]
        for i in range(ra):
            for j in range(cb):
                C[i][j] = sum(A[i][k] * B[k][j] for k in range(ca))
        return C

    @staticmethod
    def _mv(A: List[List[float]], v: List[float]) -> List[float]:
        """Matrix-vector product A × v."""
        return [sum(A[i][j] * v[j] for j in range(len(v))) for i in range(len(A))]

    @staticmethod
    def _madd(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
        return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

    @staticmethod
    def _msub(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
        return [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

    @staticmethod
    def _inv3(M: List[List[float]]) -> List[List[float]]:
        """3×3 matrix inverse using Cramer's rule."""
        a, b, c = M[0][0], M[0][1], M[0][2]
        d, e, f = M[1][0], M[1][1], M[1][2]
        g, h, k = M[2][0], M[2][1], M[2][2]
        det = a*(e*k - f*h) - b*(d*k - f*g) + c*(d*h - e*g)
        if abs(det) < 1e-18:
            return KalmanFilter6DOF._eye(3)
        inv_det = 1.0 / det
        return [
            [(e*k - f*h)*inv_det, -(b*k - c*h)*inv_det,  (b*f - c*e)*inv_det],
            [-(d*k - f*g)*inv_det,  (a*k - c*g)*inv_det, -(a*f - c*d)*inv_det],
            [(d*h - e*g)*inv_det, -(a*h - b*g)*inv_det,  (a*e - b*d)*inv_det],
        ]


# ══════════════════════════════════════════════════════════════════
#  4. ORCA COLLISION AVOIDANCE
# ══════════════════════════════════════════════════════════════════

class ORCAAgent:
    """
    Optimal Reciprocal Collision Avoidance (ORCA) — 2D horizontal plane.

    Algorithm (van den Berg et al., "Reciprocal n-Body Collision Avoidance"):
      For each neighbour B of agent A:
        1. Compute relative position p = p_B - p_A  and velocity v = v_A - v_B.
        2. If |p| > combined_radius + max_speed × horizon: safe, skip.
        3. Solve quadratic for time-to-collision t_min.
        4. Derive avoidance half-plane normal from VO boundary.
        5. Accumulate half-plane constraints.
      Solve LP / project preferred velocity onto feasible region.

    Simplified here: weighted constraint accumulation (sufficient for
    sparse aerial fleets; full LP is needed for ground swarms with
    dense packing).
    """

    def __init__(self,
                 radius:       float = ORCA_RADIUS,
                 max_speed:    float = MAX_SPEED,
                 time_horizon: float = ORCA_HORIZON):
        self.radius       = radius
        self.max_speed    = max_speed
        self.time_horizon = time_horizon

    def compute_avoidance_velocity(
        self,
        pos:      Tuple[float, float],
        vel:      Tuple[float, float],
        pref_vel: Tuple[float, float],
        others:   List[Dict]          # [{"pos":(x,y), "vel":(vx,vy)}]
    ) -> Tuple[float, float]:
        """
        pos, vel, pref_vel : (x_m, y_m) in local metric frame.
        Returns adjusted (vx, vy) in m/s.
        """
        px, py = pos
        vx, vy = vel
        pvx, pvy = pref_vel

        constraints: List[Tuple[float, float, float]] = []  # (avoid_x, avoid_y, urgency)

        for other in others:
            opx, opy = other["pos"]
            ovx, ovy = other["vel"]

            rpx = opx - px
            rpy = opy - py
            rvx = vx  - ovx
            rvy = vy  - ovy

            dist = math.sqrt(rpx ** 2 + rpy ** 2)
            combined_r = self.radius * 2

            # Early exit if no collision possible within horizon
            if dist > combined_r + self.max_speed * self.time_horizon:
                continue
            if dist < 1e-6:
                continue

            # Time to collision via quadratic  |rel_pos + rel_vel·t|² = r²
            a = rvx ** 2 + rvy ** 2
            b = 2.0 * (rpx * rvx + rpy * rvy)
            c = dist ** 2 - combined_r ** 2

            if a < 1e-8:
                continue

            discriminant = b ** 2 - 4 * a * c
            if discriminant < 0:
                continue   # No collision

            t_min = (-b - math.sqrt(discriminant)) / (2 * a)
            if t_min < 0 or t_min > self.time_horizon:
                continue

            # Avoidance direction: push velocity away from relative velocity
            avoid_x = -rvx / max(dist, 1e-6)
            avoid_y = -rvy / max(dist, 1e-6)
            urgency = max(0.0, 1.0 - t_min / self.time_horizon)
            constraints.append((avoid_x, avoid_y, urgency))

        if not constraints:
            return pref_vel

        # Project preferred velocity by applying all half-plane constraints
        adj_vx, adj_vy = pvx, pvy
        for ax, ay, urgency in constraints:
            strength  = self.max_speed * urgency * 0.5
            adj_vx   += ax * strength
            adj_vy   += ay * strength

        # Clip to max speed
        spd = math.sqrt(adj_vx ** 2 + adj_vy ** 2)
        if spd > self.max_speed:
            adj_vx = adj_vx / spd * self.max_speed
            adj_vy = adj_vy / spd * self.max_speed

        return adj_vx, adj_vy


# ══════════════════════════════════════════════════════════════════
#  5. MAVLink-STYLE MESSAGE PROTOCOL
# ══════════════════════════════════════════════════════════════════

class MAVMessage:
    """
    Lightweight MAVLink-inspired message protocol.

    Real MAVLink uses binary-packed structs over serial/UDP.
    We use JSON dicts for human-readability in a research context.

    Message IDs mirror MAVLink v2:
      0   HEARTBEAT
      39  MISSION_ITEM
      76  COMMAND_LONG
      147 BATTERY_STATUS
      253 STATUSTEXT
    """

    _seq = 0  # global sequence counter

    @classmethod
    def _next_seq(cls) -> int:
        cls._seq += 1
        return cls._seq

    @classmethod
    def heartbeat(cls, drone_id: str, state: str,
                  battery: float, lat: float, lng: float, alt: float) -> dict:
        return {
            "msgid":   0,
            "seq":     cls._next_seq(),
            "sysid":   cls._drone_sysid(drone_id),
            "type":    "MAV_TYPE_QUADROTOR",
            "autopilot": "MAV_AUTOPILOT_GENERIC",
            "state":   state,
            "battery": round(battery, 1),
            "lat":     round(lat, 7),
            "lng":     round(lng, 7),
            "alt":     round(alt, 1),
            "ts":      time.time(),
        }

    @classmethod
    def mission_item(cls, drone_id: str, item_seq: int,
                     lat: float, lng: float, alt: float,
                     command: str = "NAV_WAYPOINT") -> dict:
        return {
            "msgid":    39,
            "seq":      cls._next_seq(),
            "drone_id": drone_id,
            "item_seq": item_seq,
            "command":  command,
            "frame":    "MAV_FRAME_GLOBAL_RELATIVE_ALT",
            "lat":      round(lat, 7),
            "lng":      round(lng, 7),
            "alt":      round(alt, 1),
            "ts":       time.time(),
        }

    @classmethod
    def command_long(cls, drone_id: str, command: str, params: Optional[dict] = None) -> dict:
        return {
            "msgid":    76,
            "seq":      cls._next_seq(),
            "drone_id": drone_id,
            "command":  command,
            "params":   params or {},
            "ts":       time.time(),
        }

    @classmethod
    def battery_status(cls, drone_id: str,
                       voltage_v: float, current_a: float, pct: float) -> dict:
        return {
            "msgid":      147,
            "seq":        cls._next_seq(),
            "drone_id":   drone_id,
            "voltage_mv": round(voltage_v * 1000),
            "current_ca": round(current_a * 100),
            "battery_pct": round(pct, 1),
            "ts":         time.time(),
        }

    @classmethod
    def statustext(cls, drone_id: str, severity: str, text: str) -> dict:
        return {
            "msgid":    253,
            "seq":      cls._next_seq(),
            "drone_id": drone_id,
            "severity": severity,   # DEBUG / INFO / WARNING / ERROR / CRITICAL
            "text":     text,
            "ts":       time.time(),
        }

    @staticmethod
    def _drone_sysid(drone_id: str) -> int:
        """Extract a small integer sysid from drone name (e.g. 'Alpha' → 1)."""
        names = ["Alpha", "Beta", "Gamma", "Delta", "Echo", "Foxtrot",
                 "Sigma", "Omega"]
        try:
            return names.index(drone_id) + 1
        except ValueError:
            return hash(drone_id) % 255 + 1


# ══════════════════════════════════════════════════════════════════
#  6. DRONE AGENT  (full autonomous state machine)
# ══════════════════════════════════════════════════════════════════

class DroneAgent:
    """
    Fully autonomous drone agent.

    State machine lifecycle:
        IDLE → TAKEOFF → CLIMB → CRUISE_PICKUP → HOVER_PICKUP →
        CRUISE_DELIVER → HOVER_DELIVER → RTB → LAND → CHARGING → IDLE

    Each state uses real control algorithms:
      - PID flight controller (position → velocity)
      - Kalman filter         (GPS + IMU fusion)
      - ORCA                  (multi-drone avoidance)
      - MAVLink               (telemetry log)
    """

    # Ticks required in hover states
    HOVER_PICKUP_TICKS  = 30    # ~3 s at 10 Hz
    HOVER_DELIVER_TICKS = 25    # ~2.5 s

    def __init__(self,
                 drone_id:   str,
                 home:       GPSCoord,
                 battery:    float = 100.0,
                 color:      str   = "#00e5ff"):
        self.drone_id = drone_id
        self.id       = drone_id      # alias for Flask compat
        self.color    = color
        self.home     = home

        # Physical state (Kalman-estimated)
        self.lat  = home.lat
        self.lng  = home.lng
        self.alt  = 0.0
        self.v_lat = 0.0   # velocity m/s (North component)
        self.v_lng = 0.0   # velocity m/s (East component)
        self.v_alt = 0.0

        # Battery model
        self.battery    = battery
        self.soh        = 100.0   # State of Health %
        self.cycle_count = 0

        # Mission state
        self.state: FlightState = FlightState.IDLE
        self.mission: Optional[Mission] = None
        self.path: List[Tuple[float, float]] = []
        self.path_idx: int = 0
        self.store_waypoint_idx: int = 0   # index where leg 1 ends / leg 2 starts
        self.target_alt: float = 0.0
        self.hover_ticks: int = 0
        self._fleet_nfz: List[dict] = []   # set by FleetManager before each mission

        # Sub-systems
        self.kalman = KalmanFilter6DOF(home.lat, home.lng, 0.0)
        self.fc     = DroneFlightController(drone_id)
        self.orca   = ORCAAgent()

        # Telemetry
        self.telemetry_log: List[dict] = []
        self.mavlink_log:   List[dict] = []
        self._last_heartbeat = 0.0

        # Research metrics
        self.total_distance_m    = 0.0
        self.km_flown            = 0.0
        self.deliveries_completed = 0
        self.missions_aborted     = 0
        self.energy_used_wh       = 0.0
        self.collisions_avoided   = 0
        self.total_hover_s        = 0.0

        # GPS noise model (3 m standard deviation in degrees)
        self._gps_sigma = GPS_SIGMA / (EARTH_R * DEG2RAD)

        # Fault state
        self.fault_type: Optional[str] = None

        # For battery monitoring
        self._charge_rate_map = {(0, 80): 1.2, (80, 95): 0.4, (95, 101): 0.1}

        # Smooth interpolation state
        self.path_frac: float = 0.0        # fraction through current segment (0.0–1.0)
        self.replan_cooldown: int = 0      # ticks before mid-flight replan allowed again
        self.replan_count: int = 0
        self._live_dynamic_obs: List[dict] = []  # updated by FleetManager when obstacles move

    # ── Mission assignment ────────────────────────────────────────

    def assign_mission(self,
                       mission: Mission,
                       path:    List[Tuple[float, float]],
                       store_idx: int) -> None:
        """Called by FleetCoordinator after A* path is planned."""
        self.mission         = mission
        self.path            = path
        self.path_idx        = 0
        self.path_frac       = 0.0
        self.replan_cooldown = 0
        self.store_waypoint_idx = store_idx
        self.target_alt      = CRUISE_ALT
        self.state           = FlightState.TAKEOFF
        self.hover_ticks     = 0
        self.fc.reset()

        mission.status      = MissionStatus.INFLIGHT
        mission.assigned_to = self.drone_id
        mission.started_at  = time.time()

        # MAVLink: ARM command
        self._send_mavlink(MAVMessage.command_long(
            self.drone_id, "MAV_CMD_COMPONENT_ARM_DISARM", {"arm": 1}
        ))
        # MAVLink: upload mission items
        for i, (wlat, wlng) in enumerate(path):
            self._send_mavlink(MAVMessage.mission_item(
                self.drone_id, i, wlat, wlng, CRUISE_ALT
            ))
        self._send_mavlink(MAVMessage.statustext(
            self.drone_id, "INFO",
            f"Mission {mission.id} started: {mission.pickup_name} → {mission.deliver_name}"
        ))
        log.info("[%s] Mission %d assigned: %d waypoints", self.drone_id, mission.id, len(path))

    # ── Main tick  (called at TICK_HZ by FleetCoordinator._loop) ─

    def tick(self, other_agents: List["DroneAgent"], weather: WeatherData) -> None:
        """Full control loop: Kalman → state machine → battery → faults → telemetry."""

        if self.state == FlightState.IDLE:
            return

        if self.state == FlightState.CHARGING:
            self._do_charging()
            return

        # ── 1. Kalman predict (IMU integration) ──
        self.kalman.predict()

        # ── 2. Simulate GPS measurement with noise ──
        gps_lat = self.lat + gauss(self._gps_sigma)
        gps_lng = self.lng + gauss(self._gps_sigma * 0.8)
        gps_alt = self.alt + gauss(0.5)

        # ── 3. Kalman update with GPS ──
        self.kalman.update_gps(gps_lat, gps_lng, gps_alt)
        est_lat, est_lng, est_alt = self.kalman.position

        # ── 4. State machine ──
        self._weather_dangerous = weather.is_dangerous   # used by _check_faults
        self._run_state_machine(est_lat, est_lng, est_alt, other_agents, weather)

        # ── 5. Battery drain (physics model, scaled for demo speed) ──
        speed = math.sqrt(self.v_lat ** 2 + self.v_lng ** 2)
        payload_factor = 1.0 + (self.mission.payload_kg if self.mission else 0.0) * 0.08
        # Divide by SIM_SPEEDUP so battery drains at real-world rate visually
        drain = (0.008 + speed * 0.002) * payload_factor * DT * weather.battery_drain_modifier / SIM_SPEEDUP
        self.battery = max(0.0, self.battery - drain)
        self.energy_used_wh += drain * 0.1

        # ── 6. Fault checks ──
        self._check_faults()

        # ── 7. Heartbeat (1 Hz) ──
        if time.time() - self._last_heartbeat > 1.0:
            self._last_heartbeat = time.time()
            self._send_mavlink(MAVMessage.heartbeat(
                self.drone_id, self.state.value, self.battery,
                self.lat, self.lng, self.alt
            ))
            # Battery status MAVLink message
            voltage = 3.7 * (self.battery / 100) * 4   # 4S LiPo ~14.8 V full
            current = 5.0 + speed * 0.5
            self._send_mavlink(MAVMessage.battery_status(
                self.drone_id, voltage, current, self.battery
            ))

        # ── 8. Telemetry snapshot ──
        self._snapshot()

    # ── State machine ─────────────────────────────────────────────

    def _run_state_machine(self,
                           lat: float, lng: float, alt: float,
                           others: List["DroneAgent"],
                           weather: WeatherData) -> None:
        s = self.state
        if s == FlightState.TAKEOFF:
            self._do_takeoff(alt, weather)
        elif s == FlightState.CLIMB:
            self._do_climb(alt)
        elif s in (FlightState.CRUISE_PICKUP, FlightState.CRUISE_DELIVER):
            self._do_cruise(lat, lng, alt, others, weather)
        elif s in (FlightState.HOVER_PICKUP, FlightState.HOVER_DELIVER):
            self._do_hover(lat, lng)
        elif s == FlightState.RTB:
            self._do_rtb(lat, lng, alt, others, weather)
        elif s == FlightState.LAND:
            self._do_land(alt)
        elif s == FlightState.FAULT_RTH:
            self._do_fault_rth(lat, lng, alt, weather)
        elif s == FlightState.FAULT_LAND:
            self._do_fault_land(alt)

    def _do_takeoff(self, alt: float, weather: WeatherData) -> None:
        self.target_alt = TAKEOFF_ALT
        self.alt = min(self.alt + MAX_VSPEED * DT * weather.speed_modifier, self.target_alt)
        self.v_alt = MAX_VSPEED * weather.speed_modifier
        if abs(self.alt - self.target_alt) < 1.0:
            self.state = FlightState.CLIMB
            self._send_mavlink(MAVMessage.statustext(self.drone_id, "INFO", "TAKEOFF complete → CLIMB"))

    def _do_climb(self, alt: float) -> None:
        self.target_alt = CRUISE_ALT
        self.alt = min(self.alt + MAX_VSPEED * DT * 1.5, self.target_alt)
        if abs(self.alt - self.target_alt) < 2.0:
            self.state = FlightState.CRUISE_PICKUP
            self.path_idx = 0
            self._send_mavlink(MAVMessage.statustext(self.drone_id, "INFO", "CLIMB complete → CRUISE_PICKUP"))

    def _do_cruise(self,
                   lat: float, lng: float, alt: float,
                   others: List["DroneAgent"],
                   weather: WeatherData) -> None:
        # Early guard: check if we've reached the relevant end of path
        pickup_boundary_reached = (
            self.state == FlightState.CRUISE_PICKUP and
            self.store_waypoint_idx > 0 and
            self.path_idx >= self.store_waypoint_idx - 1
        )
        if not self.path or pickup_boundary_reached or self.path_idx >= len(self.path) - 1:
            if self.state == FlightState.CRUISE_DELIVER:
                self.state = FlightState.HOVER_DELIVER
                self.hover_ticks = 0
                self._send_mavlink(MAVMessage.statustext(
                    self.drone_id, "INFO",
                    f"Arrived at delivery: {self.mission.deliver_name if self.mission else '?'}"
                ))
            elif self.state == FlightState.CRUISE_PICKUP:
                self.state = FlightState.HOVER_PICKUP
                self.hover_ticks = 0
                self._send_mavlink(MAVMessage.statustext(
                    self.drone_id, "INFO",
                    f"Arrived at pickup: {self.mission.pickup_name if self.mission else '?'}"
                ))
            return

        tgt_lat, tgt_lng = self.path[self.path_idx]

        # ── Dynamic NFZ altitude ──
        target_cruise_alt = CRUISE_ALT + getattr(self, '_alt_offset', 0.0)
        for nfz in self._fleet_nfz:
            dist_to_nfz = haversine(lat, lng, nfz["lat"], nfz["lng"])
            nfz_r = nfz.get("radius_m", 100)
            if dist_to_nfz < nfz_r * 2.5:
                target_cruise_alt = NFZ_OVER_ALT if nfz.get("hard") else SOFT_NFZ_ALT
                break
        # Smooth altitude transition ±3 m per tick
        if self.alt < target_cruise_alt - 3.0:
            self.alt = min(self.alt + 3.0, target_cruise_alt)
        elif self.alt > target_cruise_alt + 3.0:
            self.alt = max(self.alt - 3.0, target_cruise_alt)
        else:
            self.alt = target_cruise_alt + gauss(0.3)

        # ── Mid-flight obstacle replanning ──
        if self.replan_cooldown > 0:
            self.replan_cooldown -= 1
        else:
            # Get fresh obstacle list from fleet (dynamic obstacles may have moved)
            live_obstacles = list(self._fleet_nfz)  # base NFZs
            # Add any dynamic obstacles tracked on this agent
            if hasattr(self, '_live_dynamic_obs'):
                live_obstacles += self._live_dynamic_obs
            # Look 12 waypoints ahead
            lookahead = 12
            obstacle_ahead = False
            for k in range(self.path_idx, min(self.path_idx + lookahead, len(self.path))):
                wp_lat, wp_lng = self.path[k][0], self.path[k][1]
                for obs in live_obstacles:
                    r = obs.get("radius_m", 50)
                    if haversine(wp_lat, wp_lng, obs["lat"], obs["lng"]) < r + 30:
                        obstacle_ahead = True
                        break
                if obstacle_ahead:
                    break
            if obstacle_ahead and len(self.path) > self.path_idx + 1:
                dest = self.path[-1]
                planner = AStarPlanner(live_obstacles)
                new_path = planner.plan(self.lat, self.lng, dest[0], dest[1])
                if len(new_path) > 1:
                    self.path = new_path
                    self.path_idx = 0
                    self.path_frac = 0.0
                    self.replan_cooldown = 8
                    self.replan_count += 1
                    log.info("[%s] Mid-flight replan #%d — obstacle in path",
                             self.drone_id, self.replan_count)
                    self._send_mavlink(MAVMessage.statustext(
                        self.drone_id, "WARN",
                        f"Mid-flight reroute #{self.replan_count} — obstacle detected"
                    ))

        # ── PID: compute preferred velocity ──
        vx, vy, _ = self.fc.compute_velocity(lat, lng, self.alt,
                                              tgt_lat, tgt_lng, target_cruise_alt)
        vx *= weather.speed_modifier
        vy *= weather.speed_modifier

        # ── ORCA: collision avoidance ──
        other_data = []
        for other in others:
            if other.drone_id == self.drone_id or other.state == FlightState.IDLE:
                continue
            opx = (other.lat - lat) * EARTH_R * DEG2RAD
            opy = (other.lng - lng) * EARTH_R * math.cos(lat * DEG2RAD) * DEG2RAD
            other_data.append({"pos": (opx, opy), "vel": (other.v_lat, other.v_lng)})

        adj_vel = self.orca.compute_avoidance_velocity(
            (0.0, 0.0), (self.v_lat, self.v_lng), (vx, vy), other_data
        )
        if len(adj_vel) == 2:
            vx, vy = adj_vel

        # ── Wind drift ──
        drift_lat = gauss(weather.wind_drift * 0.00003)
        drift_lng = gauss(weather.wind_drift * 0.00003)

        # ── Smooth interpolated movement (no teleporting between waypoints) ──
        # Advance a continuous fraction along each segment, consuming the full
        # speed budget this tick and rolling into the next segment if needed.
        spd = math.sqrt(vx ** 2 + vy ** 2)
        if spd > 0:
            # Distance budget this tick in metres (weather modifier already in vx/vy)
            budget_m = spd * DT * SIM_SPEEDUP
            prev_lat, prev_lng = self.lat, self.lng

            # During CRUISE_PICKUP, do not advance past the store_waypoint_idx boundary
            pickup_stop = (self.store_waypoint_idx - 1) if (
                self.state == FlightState.CRUISE_PICKUP and
                self.store_waypoint_idx > 0
            ) else len(self.path) - 1

            while budget_m > 0 and self.path_idx < pickup_stop:
                cur_lat, cur_lng = self.path[self.path_idx][0], self.path[self.path_idx][1]
                nxt_lat, nxt_lng = self.path[self.path_idx + 1][0], self.path[self.path_idx + 1][1]
                seg_m = haversine(cur_lat, cur_lng, nxt_lat, nxt_lng)
                if seg_m < 1e-3:
                    self.path_idx += 1
                    self.path_frac = 0.0
                    continue
                remaining_m = (1.0 - self.path_frac) * seg_m
                if budget_m >= remaining_m:
                    budget_m -= remaining_m
                    self.path_idx += 1
                    self.path_frac = 0.0
                    idx = min(self.path_idx, len(self.path) - 1)
                    self.lat = self.path[idx][0] + drift_lat
                    self.lng = self.path[idx][1] + drift_lng
                else:
                    self.path_frac += budget_m / seg_m
                    brg = bearing(cur_lat, cur_lng, nxt_lat, nxt_lng)
                    adv = self.path_frac * seg_m
                    new_lat, new_lng = move_along_bearing(cur_lat, cur_lng, adv, brg)
                    self.lat = new_lat + drift_lat
                    self.lng = new_lng + drift_lng
                    budget_m = 0

            dist_moved = haversine(prev_lat, prev_lng, self.lat, self.lng)
            self.total_distance_m += dist_moved
            self.km_flown = self.total_distance_m / 1000

        self.v_lat = vx
        self.v_lng = vy

        # ── End-of-path / pickup boundary: transition state ──
        # For CRUISE_PICKUP: stop at store_waypoint_idx (end of leg 1), not end of full path.
        # For CRUISE_DELIVER: stop at end of full path.
        if self.state == FlightState.CRUISE_PICKUP:
            # Reached the pickup boundary (leg1 end) or ran out of path
            at_pickup_boundary = (
                self.store_waypoint_idx > 0 and
                self.path_idx >= self.store_waypoint_idx - 1
            )
            at_path_end = self.path_idx >= len(self.path) - 1
            if at_pickup_boundary or at_path_end:
                self.state = FlightState.HOVER_PICKUP
                self.hover_ticks = 0
                self._send_mavlink(MAVMessage.statustext(
                    self.drone_id, "INFO",
                    f"Arrived at pickup: {self.mission.pickup_name if self.mission else '?'}"
                ))
                return
        elif self.state == FlightState.CRUISE_DELIVER:
            if self.path_idx >= len(self.path) - 1:
                self.state = FlightState.HOVER_DELIVER
                self.hover_ticks = 0
                self._send_mavlink(MAVMessage.statustext(
                    self.drone_id, "INFO",
                    f"Arrived at delivery: {self.mission.deliver_name if self.mission else '?'}"
                ))
                return

    def _do_hover(self, lat: float, lng: float) -> None:
        self.hover_ticks += 1
        self.v_lat = 0.0
        self.v_lng = 0.0
        # Slight position drift during hover (realistic)
        self.lat += gauss(0.3) / (EARTH_R * DEG2RAD)
        self.lng += gauss(0.3) / (EARTH_R * DEG2RAD)
        self.total_hover_s += DT

        if self.state == FlightState.HOVER_PICKUP:
            if self.hover_ticks >= self.HOVER_PICKUP_TICKS:
                # Plan leg 2: current position → delivery, A* NFZ-aware
                if self.mission:
                    planner = AStarPlanner(self._fleet_nfz)
                    leg2 = planner.plan(
                        self.lat, self.lng,   # current pos at pickup
                        self.mission.delivery.lat, self.mission.delivery.lng
                    )
                    self.path      = leg2 if leg2 else [(self.mission.delivery.lat,
                                                          self.mission.delivery.lng)]
                    self.path_idx  = 0
                    self.path_frac = 0.0
                    self.store_waypoint_idx = 0  # reset — no longer needed
                self.state = FlightState.CRUISE_DELIVER
                self._send_mavlink(MAVMessage.statustext(
                    self.drone_id, "INFO", "Package collected → CRUISE_DELIVER"
                ))
        else:  # HOVER_DELIVER
            if self.hover_ticks >= self.HOVER_DELIVER_TICKS:
                # Mission complete
                self.deliveries_completed += 1
                if self.mission:
                    self.mission.status       = MissionStatus.DELIVERED
                    self.mission.delivered_at = time.time()
                    self._send_mavlink(MAVMessage.statustext(
                        self.drone_id, "INFO",
                        f"Delivered mission {self.mission.id} to {self.mission.deliver_name}"
                    ))
                    # Register in fleet completed list if fleet reference available
                    try:
                        import drone_agent as _da
                        if _da._coordinator:
                            with _da._coordinator._lock:
                                _da._coordinator.completed_missions.append(self.mission)
                    except Exception:
                        pass
                self.mission  = None
                self.state    = FlightState.RTB
                # Plan A* route home to avoid NFZs on return
                _rtb_planner = AStarPlanner(self._fleet_nfz)
                rtb_path = _rtb_planner.plan(self.lat, self.lng, self.home.lat, self.home.lng)
                self.path     = rtb_path if rtb_path else [(self.home.lat, self.home.lng)]
                self.path_idx = 0
                self.path_frac = 0.0

    def _do_rtb(self,
                lat: float, lng: float, alt: float,
                others: List["DroneAgent"],
                weather: WeatherData) -> None:
        # Plan A* RTB path on first entry
        if not self.path or len(self.path) <= 1:
            planner = AStarPlanner(self._fleet_nfz)
            self.path = planner.plan(self.lat, self.lng,
                                     self.home.lat, self.home.lng)
            self.path_idx = 0
            self.path_frac = 0.0

        if not self.path or self.path_idx >= len(self.path) - 1:
            # Check if close enough to home to land
            dist_home = haversine(self.lat, self.lng, self.home.lat, self.home.lng)
            if dist_home < 20.0 or self.path_idx >= len(self.path) - 1:
                self.state = FlightState.LAND
                self._send_mavlink(MAVMessage.statustext(self.drone_id, "INFO", "RTB complete → LAND"))
            return

        vx, vy, _ = self.fc.compute_velocity(lat, lng, alt,
                                              self.path[self.path_idx][0],
                                              self.path[self.path_idx][1],
                                              CRUISE_ALT)
        vx *= weather.speed_modifier
        vy *= weather.speed_modifier
        spd = math.sqrt(vx ** 2 + vy ** 2)

        if spd > 0:
            budget_m = spd * DT * SIM_SPEEDUP
            prev_lat, prev_lng = self.lat, self.lng
            while budget_m > 0 and self.path_idx < len(self.path) - 1:
                cur_lat, cur_lng = self.path[self.path_idx][0], self.path[self.path_idx][1]
                nxt_lat, nxt_lng = self.path[self.path_idx + 1][0], self.path[self.path_idx + 1][1]
                seg_m = haversine(cur_lat, cur_lng, nxt_lat, nxt_lng)
                if seg_m < 1e-3:
                    self.path_idx += 1
                    self.path_frac = 0.0
                    continue
                remaining_m = (1.0 - self.path_frac) * seg_m
                if budget_m >= remaining_m:
                    budget_m -= remaining_m
                    self.path_idx += 1
                    self.path_frac = 0.0
                    idx = min(self.path_idx, len(self.path) - 1)
                    self.lat = self.path[idx][0]
                    self.lng = self.path[idx][1]
                else:
                    self.path_frac += budget_m / seg_m
                    brg = bearing(cur_lat, cur_lng, nxt_lat, nxt_lng)
                    adv = self.path_frac * seg_m
                    self.lat, self.lng = move_along_bearing(cur_lat, cur_lng, adv, brg)
                    budget_m = 0
            self.total_distance_m += haversine(prev_lat, prev_lng, self.lat, self.lng)
            self.km_flown = self.total_distance_m / 1000

        self.v_lat, self.v_lng = vx, vy

        # Dynamic altitude during RTB
        near_hard = any(
            haversine(self.lat, self.lng, n["lat"], n["lng"]) < n["radius_m"] * 2.5
            for n in self._fleet_nfz
        )
        target_alt = NFZ_OVER_ALT if near_hard else CRUISE_ALT
        self.alt += math.copysign(min(abs(target_alt - self.alt), MAX_VSPEED * DT), target_alt - self.alt)

        # Trigger LAND once path is exhausted
        if self.path_idx >= len(self.path) - 1:
            self.state = FlightState.LAND
            self._send_mavlink(MAVMessage.statustext(self.drone_id, "INFO", "RTB complete → LAND"))

    def _do_land(self, alt: float) -> None:
        self.alt   = max(0.0, self.alt - MAX_VSPEED * DT)
        self.v_lat = 0.0
        self.v_lng = 0.0
        if self.alt <= 0.1:
            self.alt     = 0.0
            self.state   = FlightState.CHARGING
            self.cycle_count += 1
            self.soh     = max(70.0, self.soh - 0.05)  # gradual SoH degradation
            self._send_mavlink(MAVMessage.statustext(self.drone_id, "INFO", "Landed → CHARGING"))

    def _do_charging(self) -> None:
        if self.battery >= 100.0:
            self.battery = 100.0
            self.state   = FlightState.IDLE
            self._send_mavlink(MAVMessage.statustext(self.drone_id, "INFO", "Charge complete → IDLE"))
            return
        # Non-linear CC-CV charging curve
        if self.battery < 80:
            rate = 1.2 / TICK_HZ
        elif self.battery < 95:
            rate = 0.4 / TICK_HZ
        else:
            rate = 0.1 / TICK_HZ
        self.battery = min(100.0, self.battery + rate)

    def _do_fault_rth(self, lat: float, lng: float, alt: float, weather: WeatherData) -> None:
        """Return to home at reduced speed when fault is active."""
        self._do_rtb(lat, lng, alt, [], weather)

    def _do_fault_land(self, alt: float) -> None:
        """Emergency in-place landing."""
        self.alt   = max(0.0, self.alt - MAX_VSPEED * DT * 0.5)
        self.v_lat = 0.0
        self.v_lng = 0.0
        if self.alt <= 0.1:
            self.alt        = 0.0
            self.state      = FlightState.CHARGING
            self.fault_type = None

    # ── Fault detection ───────────────────────────────────────────

    def _check_faults(self) -> None:
        if self.state in (FlightState.FAULT_RTH, FlightState.FAULT_LAND,
                          FlightState.IDLE,       FlightState.CHARGING):
            return
        if self.battery < 10 and self.state not in (FlightState.RTB, FlightState.LAND):
            self.fault_type = "LOW_BATTERY"
            self.state      = FlightState.FAULT_RTH
            if self.mission:
                self.mission.status = MissionStatus.ABORTED
                self.missions_aborted += 1
                try:
                    import drone_agent as _da
                    if _da._coordinator:
                        with _da._coordinator._lock:
                            _da._coordinator.completed_missions.append(self.mission)
                except Exception:
                    pass
            self._send_mavlink(MAVMessage.statustext(
                self.drone_id, "CRITICAL", "LOW_BATTERY — emergency RTH"
            ))
        elif getattr(self, '_weather_dangerous', False) and self.state not in (FlightState.RTB, FlightState.LAND):
            self.fault_type = "DANGEROUS_WEATHER"
            self.state      = FlightState.FAULT_RTH
            if self.mission:
                self.mission.status = MissionStatus.ABORTED
                self.missions_aborted += 1
                try:
                    import drone_agent as _da
                    if _da._coordinator:
                        with _da._coordinator._lock:
                            _da._coordinator.completed_missions.append(self.mission)
                except Exception:
                    pass
            self._send_mavlink(MAVMessage.statustext(
                self.drone_id, "CRITICAL", "DANGEROUS_WEATHER — emergency RTH to nearest hub"
            ))
        elif random.random() < 0.00015 * DT:
            self.fault_type = "MOTOR_FAILURE"
            self.state      = FlightState.FAULT_LAND
            if self.mission:
                self.mission.status = MissionStatus.ABORTED
                self.missions_aborted += 1
            self._send_mavlink(MAVMessage.statustext(
                self.drone_id, "CRITICAL", "MOTOR_FAILURE — emergency land"
            ))

    # ── Telemetry helpers ─────────────────────────────────────────

    def _send_mavlink(self, msg: dict) -> None:
        self.mavlink_log.append(msg)
        if len(self.mavlink_log) > 200:
            self.mavlink_log = self.mavlink_log[-100:]

    def _snapshot(self) -> None:
        snap = {
            "ts":                    time.time(),
            "drone_id":              self.drone_id,
            "state":                 self.state.value,
            "lat":                   round(self.lat, 7),
            "lng":                   round(self.lng, 7),
            "alt":                   round(self.alt, 1),
            "v_lat":                 round(self.v_lat, 2),
            "v_lng":                 round(self.v_lng, 2),
            "battery":               round(self.battery, 1),
            "soh":                   round(self.soh, 1),
            "kalman_uncertainty_m":  round(self.kalman.uncertainty_m, 2),
            "mission_id":            self.mission.id if self.mission else None,
            "fault":                 self.fault_type,
            "path_progress_pct":     round(self.path_idx / max(len(self.path), 1) * 100),
        }
        self.telemetry_log.append(snap)
        if len(self.telemetry_log) > 500:
            self.telemetry_log = self.telemetry_log[-200:]

    # ── Serialization (for Flask /api/state) ─────────────────────

    def to_dict(self) -> dict:
        return {
            "drone_id":           self.drone_id,
            "id":                 self.drone_id,
            "color":              self.color,
            "state":              self.state.value,
            "lat":                round(self.lat, 7),
            "lng":                round(self.lng, 7),
            "alt":                round(self.alt, 1),
            "v_lat":              round(self.v_lat, 2),
            "v_lng":              round(self.v_lng, 2),
            "battery_pct":        round(self.battery, 1),
            "soh":                round(self.soh, 1),
            "cycle_count":        self.cycle_count,
            "km_flown":           round(self.km_flown, 3),
            "energy_used_wh":     round(self.energy_used_wh, 3),
            "deliveries_completed": self.deliveries_completed,
            "missions_aborted":   self.missions_aborted,
            "collisions_avoided": self.collisions_avoided,
            "total_hover_s":      round(self.total_hover_s, 1),
            "kalman_uncertainty_m": round(self.kalman.uncertainty_m, 2),
            "fault":              self.fault_type,
            "mission":            self.mission.to_dict() if self.mission else None,
            "path_idx":           self.path_idx,
            "path_len":           len(self.path),
            "recent_mavlink":     self.mavlink_log[-5:],
            "home":               self.home.to_dict(),
        }

    @property
    def velocity(self):
        """Velocity object with .magnitude() for Flask analytics endpoint."""
        class _Vel:
            def __init__(self, vx, vy):
                self.vx = vx; self.vy = vy
            def magnitude(self):
                return math.sqrt(self.vx**2 + self.vy**2)
        return _Vel(self.v_lat, self.v_lng)

    @property
    def gps(self) -> GPSCoord:
        return GPSCoord(self.lat, self.lng, self.alt)


# ══════════════════════════════════════════════════════════════════
#  7. FLEET MANAGER  (Hungarian-inspired assignment + background loop)
# ══════════════════════════════════════════════════════════════════

class FleetManager:
    """
    Coordinates multiple DroneAgent instances.

    Responsibilities:
      - Drone registry and hub management
      - Hungarian-inspired greedy task assignment
        (true Hungarian is O(n³); greedy is O(n² log n) and sufficient
         for n ≤ 10 drones in a research context)
      - Background 10 Hz simulation loop
      - Obstacle / weather registry
      - Serialization for Flask /api/state

    Also exposed as FleetCoordinator for backward compat.
    """

    def __init__(self,
                 base_gps:   GPSCoord,
                 weather:    WeatherData,
                 n_drones:   int = 0):
        self.base_gps     = base_gps
        self.weather      = weather
        self.drones:      Dict[str, DroneAgent]  = {}
        self.mission_queue: List[Mission]          = []
        self.completed_missions: List[Mission]     = []
        self.obstacles:   List[GPSCoord]           = []
        self.nfz_list:    List[dict]               = []
        self.hubs:        List[dict]               = []   # [{lat, lng, name, region}]
        self.landmarks:   List[dict]               = []   # [{lat, lng, name, label}]
        self._planner     = AStarPlanner()
        self.active_algo  = "basic"   # "basic" | "advanced"
        self.algo_stats   = {
            "basic":    {"missions":0,"total_km":0.0,"total_ms":0.0,"path_lengths":[],"timings":[]},
            "advanced": {"missions":0,"total_km":0.0,"total_ms":0.0,"path_lengths":[],"timings":[]},
        }
        self._next_mission_id = 1
        self._lock        = threading.Lock()
        self._running     = False
        self._thread:     Optional[threading.Thread] = None

        # Research logs (returned by /api/analytics)
        self.dispatch_log: List[str] = []   # hub dispatch events
        self.astar_log:    List[str] = []   # A* path planning events
        self.orca_log:     List[str] = []   # ORCA collision events
        self.overfly_log:  List[str] = []   # landmark overfly events
        self._last_overfly: Dict[str, float] = {}  # drone|landmark → timestamp

        # Auto-create drones if n_drones > 0
        NAMES  = ["Alpha","Beta","Gamma","Delta","Echo","Foxtrot","Sigma","Omega"]
        COLORS = {"Alpha":"#00e5ff","Beta":"#b44dff","Gamma":"#ff4466",
                  "Delta":"#ffb300","Echo":"#00ff88","Foxtrot":"#ff8c00",
                  "Sigma":"#3b82f6","Omega":"#f0abfc"}
        for i in range(min(n_drones, len(NAMES))):
            name = NAMES[i]
            self.add_drone(name, COLORS.get(name, "#ffffff"))

    # ── Drone management ──────────────────────────────────────────

    def add_drone(self, drone_id: str, color: str = "#00e5ff",
                  hub: Optional[GPSCoord] = None) -> DroneAgent:
        with self._lock:
            # Use provided hub, or nearest registered hub, or base_gps with small offset
            if hub is None and self.hubs:
                # Pick hub with fewest drones for balanced assignment
                hub_counts = {h["name"]: 0 for h in self.hubs}
                for d in self.drones.values():
                    if hasattr(d, "hub_name") and d.hub_name in hub_counts:
                        hub_counts[d.hub_name] += 1
                least_loaded = min(self.hubs, key=lambda h: hub_counts.get(h["name"], 0))
                hub = GPSCoord(lat=least_loaded["lat"], lng=least_loaded["lng"], alt=0.0)
                hub_name = least_loaded["name"]
            elif hub is not None:
                hub_name = getattr(hub, "name", "Hub")
            else:
                offset_lat = len(self.drones) * 0.0002
                offset_lng = (len(self.drones) // 2) * 0.0002
                hub = GPSCoord(
                    lat=self.base_gps.lat + offset_lat,
                    lng=self.base_gps.lng + offset_lng,
                    alt=0.0
                )
                hub_name = "Base"
            agent = DroneAgent(
                drone_id=drone_id,
                home=hub,
                battery=100.0,
                color=color
            )
            agent.hub_name = hub_name  # track which hub this drone belongs to
            self.drones[drone_id] = agent
            log.info("[Fleet] Drone %s → %s at (%.5f, %.5f)",
                     drone_id, hub_name, hub.lat, hub.lng)
            return agent

    # ── Mission assignment ────────────────────────────────────────

    # ── Mission assignment ────────────────────────────────────────

    def set_algo(self, algo: str) -> str:
        """Switch pathfinding algorithm. algo: 'basic' | 'advanced'"""
        with self._lock:
            algo = algo.lower().strip()
            if algo not in ("basic", "advanced"):
                return f"Unknown algorithm '{algo}'"
            self.active_algo = algo
            nfz = self.nfz_list
            self._planner = ThetaStarPlanner(nfz) if algo == "advanced" else AStarPlanner(nfz)
            label = "Theta* Advanced A* (any-angle)" if algo == "advanced" else "Basic A* (8-directional grid)"
            log.info("[Fleet] Pathfinder switched to %s", label)
            return label

    def assign_mission(self,
                       pickup_gps:    GPSCoord,
                       pickup_name:   str,
                       deliver_gps:   GPSCoord,
                       deliver_name:  str,
                       payload_kg:    float = 1.0,
                       priority:      str   = "standard",
                       preferred_hub: Optional[GPSCoord] = None,
                       alt_offset:    float = 0.0) -> Optional[str]:
        """
        Greedy nearest-hub assignment:
          1. Prefer drones that are closest to the preferred_hub (nearest hub to pickup).
          2. Fall back to globally nearest idle drone if none near hub.
        Returns assigned drone_id, or None if no drone is available.
        """
        with self._lock:
            idle = [d for d in self.drones.values()
                    if d.state == FlightState.IDLE and d.battery > 20]
            if not idle:
                return None

            # Find nearest hub to pickup
            nearest_hub_gps = preferred_hub
            nearest_hub_name = "Fleet"
            if not nearest_hub_gps and self.hubs:
                best_hub = min(self.hubs,
                    key=lambda h: haversine(h["lat"], h["lng"],
                                            pickup_gps.lat, pickup_gps.lng))
                nearest_hub_gps = GPSCoord(lat=best_hub["lat"],
                                           lng=best_hub["lng"], alt=0.0)
                nearest_hub_name = best_hub.get("name", "Hub")

            # Step 1: prefer idle drones from the nearest hub
            hub_idle = [d for d in idle
                        if nearest_hub_gps and
                        haversine(d.home.lat, d.home.lng,
                                  nearest_hub_gps.lat, nearest_hub_gps.lng) < 300]
            pool = hub_idle if hub_idle else idle

            # Step 2: from pool, pick physically closest to pickup
            best_drone = min(pool, key=lambda d:
                haversine(d.lat, d.lng, pickup_gps.lat, pickup_gps.lng))
            actual_hub_name = getattr(best_drone, "hub_name", nearest_hub_name)

            mission = Mission(
                id           = self._next_mission_id,
                pickup       = pickup_gps,
                delivery     = deliver_gps,
                pickup_name  = pickup_name,
                deliver_name = deliver_name,
                payload_kg   = payload_kg,
                priority     = priority,
            )
            self._next_mission_id += 1

            # Plan path: drone_pos → pickup → delivery
            # Combine hard NFZs + dynamic obstacles
            nfz = list(self.nfz_list) + [
                {"lat": o.lat, "lng": o.lng, "radius_m": 50}
                for o in self.obstacles
            ]
            # ── Use whichever algorithm is active ──────────────────
            planner = (ThetaStarPlanner(nfz) if self.active_algo == "advanced"
                       else AStarPlanner(nfz))
            self._planner = planner  # keep ref in sync

            t_plan = time.perf_counter()
            path_leg1 = planner.plan(
                best_drone.lat, best_drone.lng,
                pickup_gps.lat, pickup_gps.lng
            )
            path_leg2 = planner.plan(
                pickup_gps.lat, pickup_gps.lng,
                deliver_gps.lat, deliver_gps.lng
            )
            plan_ms = (time.perf_counter() - t_plan) * 1000
            full_path  = path_leg1 + path_leg2
            store_idx  = len(path_leg1)

            # Compute path length and record per-algo stats
            def path_km(pts):
                return sum(haversine(pts[i][0],pts[i][1],pts[i+1][0],pts[i+1][1])
                           for i in range(len(pts)-1)) / 1000
            leg1_km = path_km(path_leg1)
            leg2_km = path_km(path_leg2)
            total_path_km = leg1_km + leg2_km
            s = self.algo_stats[self.active_algo]
            s["missions"]   += 1
            s["total_km"]   += total_path_km
            s["total_ms"]   += plan_ms
            s["path_lengths"].append(round(total_path_km, 3))
            s["timings"].append(round(plan_ms, 2))

            # Give drone the NFZ list so _do_hover leg-2 and _do_rtb can use it
            best_drone._fleet_nfz = nfz
            # Apply altitude stagger for Fleet Dispatch collision separation
            if alt_offset != 0.0:
                best_drone._alt_offset = alt_offset
            best_drone.assign_mission(mission, full_path, store_idx)

            dist_km = total_path_km
            ts = time.strftime("%H:%M:%S")
            algo_tag = "✦Theta*" if self.active_algo == "advanced" else "⬡A*"
            self.dispatch_log.insert(0,
                f"[{ts}] {best_drone.drone_id} ({actual_hub_name}) → {pickup_name} → {deliver_name}")
            self.astar_log.insert(0,
                f"[{ts}] {algo_tag} {best_drone.drone_id}: {actual_hub_name}→{pickup_name}→{deliver_name} "
                f"| {dist_km:.3f}km | {len(full_path)} waypts | {plan_ms:.1f}ms | priority={priority}")
            if len(self.dispatch_log) > 20: self.dispatch_log = self.dispatch_log[:20]
            if len(self.astar_log)    > 20: self.astar_log    = self.astar_log[:20]

            log.info("[Fleet] Mission %d → Drone %s from %s (%d waypoints, %.2fkm)",
                     mission.id, best_drone.drone_id, actual_hub_name,
                     len(full_path), dist_km)
            return best_drone.drone_id

    # ── Queue-based assignment ────────────────────────────────────

    def queue_mission(self, mission: Mission) -> int:
        with self._lock:
            self.mission_queue.append(mission)
            self.mission_queue.sort(key=lambda m: m.priority_rank)
            return mission.id

    def _assign_queued(self) -> None:
        with self._lock:
            idle = [d for d in self.drones.values()
                    if d.state == FlightState.IDLE and d.battery > 20]
            assigned = []
            for mission in list(self.mission_queue):
                if not idle:
                    break
                # Nearest hub to pickup
                if self.hubs:
                    best_hub = min(self.hubs, key=lambda h:
                        haversine(h["lat"], h["lng"],
                                  mission.pickup.lat, mission.pickup.lng))
                    hub_idle = [d for d in idle
                                if haversine(d.home.lat, d.home.lng,
                                             best_hub["lat"], best_hub["lng"]) < 300]
                    pool = hub_idle if hub_idle else idle
                else:
                    pool = idle
                best = min(pool, key=lambda d:
                    haversine(d.lat, d.lng, mission.pickup.lat, mission.pickup.lng))
                nfz = list(self.nfz_list) + [
                    {"lat": o.lat, "lng": o.lng, "radius_m": 50}
                    for o in self.obstacles
                ]
                planner = AStarPlanner(nfz)
                leg1 = planner.plan(best.lat, best.lng,
                                    mission.pickup.lat, mission.pickup.lng)
                leg2 = planner.plan(mission.pickup.lat, mission.pickup.lng,
                                    mission.delivery.lat, mission.delivery.lng)
                best._fleet_nfz = nfz
                best.assign_mission(mission, leg1 + leg2, len(leg1))
                hub_name = getattr(best, "hub_name", "Hub")
                ts = time.strftime("%H:%M:%S")
                self.dispatch_log.insert(0,
                    f"[{ts}] {best.drone_id} ({hub_name}) → {mission.pickup_name} → {mission.deliver_name}")
                if len(self.dispatch_log) > 20: self.dispatch_log = self.dispatch_log[:20]
                assigned.append(mission)
                idle.remove(best)
            for m in assigned:
                self.mission_queue.remove(m)

    # ── Obstacles / weather ───────────────────────────────────────

    def add_obstacle(self, gps: GPSCoord) -> None:
        with self._lock:
            self.obstacles.append(gps)
            # Push updated dynamic obstacle list to all active drones
            # so mid-flight replanning uses the latest positions
            dyn = [{"lat": o.lat, "lng": o.lng, "radius_m": 50} for o in self.obstacles]
            for drone in self.drones.values():
                drone._live_dynamic_obs = dyn

    def clear_obstacles(self) -> None:
        with self._lock:
            self.obstacles.clear()
            for drone in self.drones.values():
                drone._live_dynamic_obs = []

    def update_weather(self, weather: WeatherData) -> None:
        with self._lock:
            self.weather = weather

    def set_hubs(self, hubs: List[dict]) -> None:
        """Register hub positions: [{lat, lng, name, region}]"""
        with self._lock:
            self.hubs = list(hubs)
            log.info("[Fleet] %d hubs registered: %s",
                     len(hubs), [h.get("name") for h in hubs])

    def set_landmarks(self, landmarks: List[dict]) -> None:
        """Register landmarks for overfly detection: [{lat, lng, name, label, emoji}]"""
        with self._lock:
            self.landmarks = list(landmarks)
            log.info("[Fleet] %d landmarks registered for overfly tracking",
                     len(landmarks))

    def update_nfz(self, nfz_list: List[dict]) -> None:
        """Update the hard no-fly zone list used by A* path planning.
        Each entry: {"lat": float, "lng": float, "radius_m": float}
        """
        with self._lock:
            self.nfz_list = list(nfz_list)
            # Rebuild the shared planner with the new list
            self._planner = AStarPlanner(self.nfz_list)
            log.info("[Fleet] NFZ list updated — %d zones", len(nfz_list))

    # ── Serialization ─────────────────────────────────────────────

    def get_state(self) -> dict:
        with self._lock:
            drones_list = list(self.drones.values())
            total_del   = sum(d.deliveries_completed for d in drones_list)
            total_abort = sum(d.missions_aborted for d in drones_list)
            total_mis   = total_del + total_abort
            return {
                "ts":     time.time(),
                "drones": {did: d.to_dict() for did, d in self.drones.items()},
                "queue_length":   len(self.mission_queue),
                "completed":      len(self.completed_missions),
                "weather": {
                    "wind_kmh":    self.weather.wind_speed_kmh,
                    "speed_mod":   self.weather.speed_modifier,
                    "batt_mod":    self.weather.battery_drain_modifier,
                    "is_dangerous":self.weather.is_dangerous,
                    "drift":       self.weather.wind_drift,
                },
                "obstacles":     len(self.obstacles),
                "dispatch_log":  self.dispatch_log[:15],
                "astar_log":     self.astar_log[:15],
                "orca_log":      self.orca_log[:15],
                "overfly_log":   self.overfly_log[:15],
                "active_algo":   self.active_algo,
                "algo_stats":    {
                    k: {
                        "missions":      v["missions"],
                        "avg_km":        round(v["total_km"]/v["missions"],3) if v["missions"] else 0,
                        "avg_ms":        round(v["total_ms"]/v["missions"],2) if v["missions"] else 0,
                        "path_lengths":  v["path_lengths"][-10:],
                        "timings":       v["timings"][-10:],
                    }
                    for k,v in self.algo_stats.items()
                },
                "summary": {
                    "total_missions":     total_mis,
                    "delivered":          total_del,
                    "aborted":            total_abort,
                    "success_rate_pct":   round(total_del / total_mis * 100, 1) if total_mis > 0 else 0.0,
                    "total_km":           round(sum(d.km_flown for d in drones_list), 3),
                    "total_energy_wh":    round(sum(d.energy_used_wh for d in drones_list), 3),
                    "collisions_avoided": sum(d.collisions_avoided for d in drones_list),
                    "active_drones":      sum(1 for d in drones_list if d.state not in (FlightState.IDLE, FlightState.CHARGING)),
                },
            }

    def get_mavlink_log(self) -> List[dict]:
        msgs = []
        for agent in self.drones.values():
            msgs.extend(agent.mavlink_log[-10:])
        msgs.sort(key=lambda m: m.get("ts", 0), reverse=True)
        return msgs[:50]

    # ── Background loop ───────────────────────────────────────────

    def start_background(self) -> None:
        """Start the 10 Hz simulation loop in a daemon thread."""
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        log.info("[Fleet] Background loop started — %d drones online", len(self.drones))

    # Aliases for backward compat
    start = start_background

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    @property
    def tick_rate_hz(self) -> int:
        return TICK_HZ

    def _loop(self) -> None:
        while self._running:
            t0 = time.time()
            agents = list(self.drones.values())
            weather = self.weather
            for agent in agents:
                try:
                    agent.tick(agents, weather)
                except Exception as exc:
                    log.error("[Agent %s] tick error: %s", agent.drone_id, exc, exc_info=True)

            # ── Landmark overfly detection ──
            if self.landmarks:
                ts = time.strftime("%H:%M:%S")
                for drone in agents:
                    if drone.state in (FlightState.IDLE, FlightState.CHARGING,
                                       FlightState.LAND):
                        continue
                    for lm in self.landmarks:
                        dist = haversine(drone.lat, drone.lng, lm["lat"], lm["lng"])
                        if dist < 250:
                            key = f"{drone.drone_id}|{lm['name']}"
                            last = self._last_overfly.get(key, 0)
                            if time.time() - last > 30:
                                self._last_overfly[key] = time.time()
                                entry = (f"[{ts}] {drone.drone_id} overflew "
                                         f"{lm.get('emoji','')} {lm['name']} "
                                         f"({dist:.0f}m)")
                                self.overfly_log.insert(0, entry)
                                if len(self.overfly_log) > 20:
                                    self.overfly_log = self.overfly_log[:20]

            # ── ORCA proximity logging ──
            ts = time.strftime("%H:%M:%S")
            for i in range(len(agents)):
                for j in range(i + 1, len(agents)):
                    a, b = agents[i], agents[j]
                    if (a.state == FlightState.IDLE or
                            b.state == FlightState.IDLE):
                        continue
                    dist_m = haversine(a.lat, a.lng, b.lat, b.lng)
                    if dist_m < 150:
                        key = f"{a.drone_id}|{b.drone_id}"
                        last = self._last_overfly.get(key, 0)
                        if time.time() - last > 3:
                            self._last_overfly[key] = time.time()
                            entry = (f"[{ts}] ORCA {a.drone_id}↔{b.drone_id}: "
                                     f"{dist_m:.0f}m")
                            self.orca_log.insert(0, entry)
                            if len(self.orca_log) > 20:
                                self.orca_log = self.orca_log[:20]

            self._assign_queued()
            elapsed = time.time() - t0
            time.sleep(max(0.0, DT - elapsed))


# ── Backward compat alias ──────────────────────────────────────────
FleetCoordinator = FleetManager


# ══════════════════════════════════════════════════════════════════
#  MODULE-LEVEL CONVENIENCE (used by flask_server_robotics.py)
# ══════════════════════════════════════════════════════════════════

_coordinator: Optional[FleetManager] = None


def init_fleet(home_lat: float, home_lng: float,
               n_drones: int = 4,
               nfz_list: Optional[List[dict]] = None,
               weather:  Optional[WeatherData] = None) -> FleetManager:
    global _coordinator
    base = GPSCoord(lat=home_lat, lng=home_lng, alt=0.0)
    wx   = weather or WeatherData()
    _coordinator = FleetManager(base_gps=base, weather=wx, n_drones=n_drones)
    if nfz_list:
        _coordinator._planner = AStarPlanner(nfz_list)
    _coordinator.start_background()
    return _coordinator


def get_coordinator() -> Optional[FleetManager]:
    return _coordinator


# ══════════════════════════════════════════════════════════════════
#  SELF-TEST
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")

    print("=" * 64)
    print("  SkyLogix DroneAgent v3.0 — Self-Test")
    print("=" * 64)

    # ── Unit test: A* planner ──────────────────────────────────────
    print("\n[1] A* Planner …")
    planner = AStarPlanner()
    path = planner.plan(16.5062, 80.6480, 16.5120, 80.6550)
    print(f"    {len(path)} waypoints  "
          f"start→{path[0]}  end→{path[-1]}")
    assert len(path) >= 1

    # ── Unit test: Kalman filter ───────────────────────────────────
    print("\n[2] Kalman Filter …")
    kf = KalmanFilter6DOF(16.5062, 80.6480, 0.0)
    for _ in range(5):
        kf.predict()
        kf.update_gps(16.5062 + gauss(3e-6),
                      80.6480 + gauss(3e-6),
                      gauss(0.5))
    pos = kf.position
    unc = kf.uncertainty_m
    print(f"    pos=({pos[0]:.6f},{pos[1]:.6f},{pos[2]:.2f})  "
          f"uncertainty={unc:.2f} m")
    assert unc < 20

    # ── Unit test: PID ─────────────────────────────────────────────
    print("\n[3] PID Controller …")
    pid = PIDController(kp=1.8, ki=0.05, kd=0.4, out_min=-15, out_max=15)
    outputs = [pid.compute(0, err) for err in [10, 5, 2, 1, 0.5]]
    print(f"    outputs: {[round(o,3) for o in outputs]}")
    assert all(-15 <= o <= 15 for o in outputs)

    # ── Unit test: ORCA ────────────────────────────────────────────
    print("\n[4] ORCA …")
    orca = ORCAAgent()
    adj = orca.compute_avoidance_velocity(
        (0, 0), (10, 0), (10, 0),
        [{"pos": (20, 5), "vel": (-10, 0)}]
    )
    print(f"    adjusted velocity: ({adj[0]:.2f}, {adj[1]:.2f}) m/s")

    # ── Integration test: Fleet ────────────────────────────────────
    print("\n[5] Fleet integration (3 drones, 3 missions) …")
    coord = init_fleet(16.5062, 80.6480, n_drones=3)
    time.sleep(0.3)

    m1 = coord.assign_mission(
        GPSCoord(16.5062, 80.6480), "Hub Alpha",
        GPSCoord(16.5120, 80.6550), "MG Road",
        payload_kg=1.5, priority="express")
    m2 = coord.assign_mission(
        GPSCoord(16.5062, 80.6480), "Hub Alpha",
        GPSCoord(16.5020, 80.6420), "Patamata",
        payload_kg=0.8, priority="urgent")
    m3 = coord.assign_mission(
        GPSCoord(16.5062, 80.6480), "Hub Alpha",
        GPSCoord(16.5090, 80.6390), "Governorpet",
        payload_kg=2.0, priority="standard")

    print(f"    Assigned → drones: {m1}, {m2}, {m3}")
    time.sleep(2.0)

    status = coord.get_state()
    print(f"    Fleet snapshot after 2 s:")
    for did, s in status["drones"].items():
        print(f"      {did}: state={s['state']}  "
              f"battery={s['battery_pct']}%  "
              f"mission={s['mission']['id'] if s['mission'] else None}")
    print(f"    Queue: {status['queue_length']} pending")

    # ── MAVLink log ────────────────────────────────────────────────
    print("\n[6] MAVLink log (last 6 messages):")
    for msg in coord.get_mavlink_log()[:6]:
        ts   = time.strftime("%H:%M:%S", time.localtime(msg.get("ts", 0)))
        name = msg.get("drone_id", "?")
        text = msg.get("text") or msg.get("state") or msg.get("command") or ""
        print(f"    [{ts}] SEQ={msg.get('seq',0):04d}  {name}  {text}")

    coord.stop()
    print("\n✅  All self-tests passed.\n")