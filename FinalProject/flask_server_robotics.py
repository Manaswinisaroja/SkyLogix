"""
flask_server_robotics.py — SkyLogix UAV Research Platform v4.0
===============================================================
Flask REST API + SSE bridge between the HTML frontend and the
real DroneAgent fleet running in drone_agent.py.

Three research extensions active (all additive, zero breaking changes):
  🥇 Newton-Euler / 6-DOF Dynamics  — drone_dynamics_6dof.py
     Full x,y,z + Roll,Pitch,Yaw rigid-body physics on every drone.
     Added to every DroneAgent.to_dict() as "dynamics_6dof" key.

  🥈 Full 6-DOF State Telemetry      — drone_dynamics_6dof.py
     Every orientation change (turn → roll, acceleration → pitch)
     is physically derived, not hard-coded.

  🥉 Advanced Flight Controller       — advanced_flight_controller.py
     Switchable PID / LQR / MPC / Open-Loop per drone or fleet-wide.
     POST /api/drone/<id>/control_mode  or  POST /api/fleet/control_mode
     GET  /api/controller/compare   — step-response benchmark

Endpoints:
  POST /api/init                     → Initialize the fleet
  GET  /api/state                    → Full fleet state (polling, ~500 ms)
  POST /api/mission                  → Assign a new delivery mission
  POST /api/drone/add                → Add a new drone to the fleet
  POST /api/drone/<id>/control_mode  → Switch one drone's controller
  POST /api/fleet/control_mode       → Switch all drones' controller
  GET  /api/controller/compare       → PID vs LQR vs MPC benchmark
  POST /api/weather                  → Update weather / physics
  POST /api/obstacle/add             → Add a dynamic obstacle (bird / NFZ)
  POST /api/obstacle/clear           → Clear all obstacles
  POST /api/nfz                      → Update hard no-fly zones
  GET  /api/drone/<id>               → Single drone state
  GET  /api/analytics                → Research analytics snapshot
  GET  /api/stream                   → SSE live-stream (alt to polling)
  GET  /health                       → Health check
"""

from flask import Flask, jsonify, request, Response
from flask_cors import CORS
import threading
import time
import json
import logging
from typing import Optional

# ── Import the real drone algorithms ──────────────────────────────
from drone_agent import (
    FleetManager,
    DroneAgent,
    GPSCoord,
    WeatherData,
    Mission,
    FlightState,
    MissionStatus,
)

# ── Full 6-DOF Drone Dynamics (additive extension — see drone_dynamics_6dof.py) ──
# Adds x, y, z, Roll, Pitch, Yaw on top of the existing simulation.
# Every DroneAgent.to_dict() gains a new "dynamics_6dof" key; nothing
# existing is renamed, removed, or changed.
from drone_dynamics_6dof import enable_6dof_dynamics
enable_6dof_dynamics()

# ─────────────────────────────────────────────────────────────────
#  APP SETUP
# ─────────────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app)   # Allow cross-origin requests from the HTML frontend

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("FlaskServer")

# Global fleet — initialized by POST /api/init (or auto-init on startup)
fleet: Optional[FleetManager] = None
_fleet_lock = threading.Lock()

DRONE_NAMES  = ["Alpha", "Beta", "Gamma", "Delta", "Echo", "Foxtrot", "Sigma", "Omega"]
DRONE_COLORS = {
    "Alpha":   "#00e5ff",
    "Beta":    "#b44dff",
    "Gamma":   "#ff4466",
    "Delta":   "#ffb300",
    "Echo":    "#00ff88",
    "Foxtrot": "#ff8c00",
    "Sigma":   "#3b82f6",
    "Omega":   "#f0abfc",
}


# ─────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────

def ok(data=None, message: str = "success"):
    return jsonify({"status": "ok", "message": message, "data": data or {}})


def err(message: str, code: int = 400):
    return jsonify({"status": "error", "message": message}), code


def require_fleet():
    if fleet is None:
        return err("Fleet not initialized. POST /api/init first.", 503)
    return None


# ─────────────────────────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    """Health check — also confirms drone_agent imports correctly."""
    return ok({
        "fleet_active":  fleet is not None,
        "drone_count":   len(fleet.drones) if fleet else 0,
        "server_time":   time.time(),
        "algorithms":    ["Newton-Euler 6-DOF", "PID/LQR/MPC/OpenLoop",
                          "A*/Theta*", "Kalman6DOF", "ORCA", "MAVLink",
                          "HungarianAssignment", "SwarmStateMachine"],
        "features": {
            "6dof_dynamics":        True,
            "advanced_controllers": True,
            "controller_modes":     ["PID", "LQR", "MPC", "OPEN_LOOP"],
            "nfz_avoidance":        True,
            "theta_star":           True,
        }
    }, "SkyLogix UAV Server v4.0 online")


@app.route("/api/init", methods=["POST"])
def init_fleet():
    """
    Initialize the drone fleet at a given base location.

    Body (JSON):
        {
            "base_lat":   17.385,
            "base_lng":   78.4867,
            "num_drones": 6,
            "wind_kmh":   10.0,
            "hubs": [
                {"lat": 17.395, "lng": 78.480, "name": "Hub-Alpha", "region": "North"},
                {"lat": 17.375, "lng": 78.495, "name": "Hub-Beta",  "region": "South-East"},
                {"lat": 17.375, "lng": 78.470, "name": "Hub-Gamma", "region": "South-West"}
            ],
            "landmarks": [{"lat": .., "lng": .., "name": .., "label": .., "emoji": ..}]
        }
    """
    global fleet
    data = request.get_json(force=True) or {}

    base_lat    = float(data.get("base_lat",    17.385))
    base_lng    = float(data.get("base_lng",    78.4867))
    num_drones  = min(int(data.get("num_drones", 6)), len(DRONE_NAMES))
    wind_kmh    = float(data.get("wind_kmh",    0.0))
    wind_dir    = float(data.get("wind_direction", 0.0))
    code        = int(data.get("weather_code", 0))
    hubs        = data.get("hubs", [])        # [{lat,lng,name,region}]
    landmarks   = data.get("landmarks", [])   # [{lat,lng,name,label,emoji}]

    with _fleet_lock:
        if fleet is not None:
            fleet.stop()

        base    = GPSCoord(lat=base_lat, lng=base_lng, alt=0.0)
        weather = WeatherData(
            wind_speed_kmh     = wind_kmh,
            wind_direction_deg = wind_dir,
            weather_code       = code,
            is_dangerous       = wind_kmh > 50 or code in (82, 95, 96, 99),
            is_caution         = wind_kmh > 25 or code in (61, 63, 65, 71, 75, 80, 81),
        )

        new_fleet = FleetManager(base_gps=base, weather=weather)

        # Register hubs so add_drone places drones correctly
        if hubs:
            new_fleet.set_hubs(hubs)

        # Register landmarks for overfly detection
        if landmarks:
            new_fleet.set_landmarks(landmarks)

        # Add drones — they will auto-distribute across hubs (2 per hub)
        for i in range(num_drones):
            name = DRONE_NAMES[i]
            new_fleet.add_drone(name, DRONE_COLORS[name])

        new_fleet.start_background()

        import drone_agent as _da
        _da._coordinator = new_fleet
        globals()["fleet"] = new_fleet

    log.info("Fleet initialized at (%.5f, %.5f) with %d drones across %d hubs",
             base_lat, base_lng, num_drones, len(hubs))

    return ok({
        "base":       {"lat": base_lat, "lng": base_lng},
        "drones":     DRONE_NAMES[:num_drones],
        "hubs":       len(hubs),
        "landmarks":  len(landmarks),
        "tick_hz":    new_fleet.tick_rate_hz,
        "algorithms": ["PID x2/drone", "A*", "Kalman6DOF", "ORCA",
                       "MAVLink", "HungarianAssignment"],
    }, f"Fleet initialized with {num_drones} drones across {len(hubs)} hubs")


@app.route("/api/state", methods=["GET"])
def get_state():
    """
    Full fleet state snapshot.  Called by frontend every ~500 ms.
    Returns all drone positions, states, battery, telemetry, etc.
    """
    e = require_fleet()
    if e:
        return e
    return jsonify(fleet.get_state())


@app.route("/api/drone/<drone_id>", methods=["GET"])
def get_drone(drone_id: str):
    """Get state of a single drone."""
    e = require_fleet()
    if e:
        return e
    with fleet._lock:
        drone = fleet.drones.get(drone_id)
        if not drone:
            return err(f"Drone '{drone_id}' not found", 404)
        return jsonify(drone.to_dict())


@app.route("/api/drone/add", methods=["POST"])
def add_drone():
    """Add a new drone to the fleet at the base hub."""
    e = require_fleet()
    if e:
        return e
    with fleet._lock:
        n = len(fleet.drones)
        if n >= len(DRONE_NAMES):
            return err("Maximum drones reached")
        name = DRONE_NAMES[n]
        fleet.add_drone(name, DRONE_COLORS[name])
    return ok(
        {"drone_id": name, "color": DRONE_COLORS[name]},
        f"Drone {name} added to fleet",
    )


@app.route("/api/mission", methods=["POST"])
def assign_mission():
    """
    Assign a delivery mission to the best available drone.

    Body (JSON):
        {
            "pickup_lat":    17.390,
            "pickup_lng":    78.492,
            "pickup_name":   "Restaurant A",
            "deliver_lat":   17.382,
            "deliver_lng":   78.483,
            "deliver_name":  "Home #4B",
            "payload_kg":    1.2,
            "priority":      "express"   // standard | express | urgent
        }
    """
    e = require_fleet()
    if e:
        return e

    data = request.get_json(force=True) or {}
    required = ["pickup_lat", "pickup_lng", "deliver_lat", "deliver_lng"]
    for f_name in required:
        if f_name not in data:
            return err(f"Missing field: {f_name}")

    if fleet.weather.is_dangerous:
        return err("Flights suspended — dangerous weather conditions", 503)

    pickup  = GPSCoord(lat=float(data["pickup_lat"]),
                       lng=float(data["pickup_lng"]),
                       alt=0.0)
    deliver = GPSCoord(lat=float(data["deliver_lat"]),
                       lng=float(data["deliver_lng"]),
                       alt=0.0)

    # Optional: preferred hub coords for nearest-hub dispatch
    preferred_hub = None
    if "preferred_hub_lat" in data and "preferred_hub_lng" in data:
        preferred_hub = GPSCoord(
            lat=float(data["preferred_hub_lat"]),
            lng=float(data["preferred_hub_lng"]),
            alt=0.0
        )

    drone_id = fleet.assign_mission(
        pickup_gps    = pickup,
        pickup_name   = data.get("pickup_name",  "Pickup"),
        deliver_gps   = deliver,
        deliver_name  = data.get("deliver_name", "Destination"),
        payload_kg    = float(data.get("payload_kg", 1.0)),
        priority      = data.get("priority", "standard"),
        preferred_hub = preferred_hub,
    )

    if drone_id is None:
        return err("No drones available — all busy or low battery.", 503)

    drone = fleet.drones[drone_id]
    return ok({
        "drone_id":   drone_id,
        "color":      drone.color,
        "mission_id": drone.mission.id if drone.mission else None,
        "state":      drone.state.value,
        "algorithms_active": ["PID", "A*", "Kalman6DOF", "ORCA", "MAVLink"],
    }, f"Mission assigned to Drone {drone_id}")


@app.route("/api/bulk_dispatch", methods=["POST"])
def bulk_dispatch():
    """
    Dispatch multiple deliveries from a single pickup (shopkeeper / fleet dispatch mode).

    Body (JSON):
        {
            "pickup_lat":   17.390,
            "pickup_lng":   78.492,
            "pickup_name":  "Justbake Bakery",
            "deliveries": [
                {
                    "deliver_lat":  17.382,
                    "deliver_lng":  78.483,
                    "deliver_name": "Customer House #1",
                    "item":         "2x Bread",
                    "payload_kg":   1.2,
                    "priority":     "express"
                },
                ...
            ]
        }
    Returns list of assigned drone IDs (or 'queued' if no drone available yet).
    """
    e = require_fleet()
    if e:
        return e

    if fleet.weather.is_dangerous:
        return err("Flights suspended — dangerous weather conditions", 503)

    data = request.get_json(force=True) or {}
    deliveries = data.get("deliveries", [])
    if not deliveries:
        return err("No deliveries provided")

    pickup = GPSCoord(
        lat=float(data.get("pickup_lat", 0)),
        lng=float(data.get("pickup_lng", 0)),
        alt=0.0,
    )
    pickup_name = data.get("pickup_name", "Store")

    results = []
    ALT_OFFSETS = [0, 10, -10, 15, -15, 20]  # stagger cruise altitudes to prevent ORCA conflicts
    for idx, d in enumerate(deliveries):
        try:
            deliver = GPSCoord(
                lat=float(d["deliver_lat"]),
                lng=float(d["deliver_lng"]),
                alt=0.0,
            )
            drone_id = fleet.assign_mission(
                pickup_gps   = pickup,
                pickup_name  = pickup_name,
                deliver_gps  = deliver,
                deliver_name = d.get("deliver_name", "Destination"),
                payload_kg   = float(d.get("payload_kg", 1.0)),
                priority     = d.get("priority", "standard"),
                alt_offset   = ALT_OFFSETS[idx % len(ALT_OFFSETS)],
            )
            results.append({
                "deliver_name": d.get("deliver_name", "Destination"),
                "item":         d.get("item", "Package"),
                "drone_id":     drone_id or "queued",
                "status":       "assigned" if drone_id else "queued",
            })
        except Exception as ex:
            results.append({"deliver_name": d.get("deliver_name","?"), "error": str(ex)})

    assigned = sum(1 for r in results if r.get("status") == "assigned")
    queued   = sum(1 for r in results if r.get("status") == "queued")
    log.info("[BulkDispatch] %d assigned, %d queued from %s", assigned, queued, pickup_name)
    return ok({
        "total":    len(deliveries),
        "assigned": assigned,
        "queued":   queued,
        "results":  results,
    }, f"Bulk dispatch: {assigned} assigned, {queued} queued")


@app.route("/api/set_algo", methods=["POST"])
def set_algo():
    """
    Switch pathfinding algorithm for all future mission planning.
    Body: { "algo": "basic" | "advanced" }
    "basic"    = A* 8-directional grid
    "advanced" = Theta* any-angle with line-of-sight pruning
    """
    e = require_fleet()
    if e:
        return e
    data  = request.get_json(force=True) or {}
    algo  = data.get("algo", "basic")
    label = fleet.set_algo(algo)
    return ok(f"Algorithm set to: {label}",
              {"algo": fleet.active_algo, "label": label})


@app.route("/api/weather", methods=["POST"])
def update_weather():
    """
    Update weather conditions.  Immediately affects all drone physics.

    Body (JSON):
        {
            "wind_kmh":       12.0,
            "wind_direction": 180.0,
            "temperature_c":  28.0,
            "humidity_pct":   65.0,
            "visibility_km":  8.0,
            "weather_code":   1
        }
    """
    e = require_fleet()
    if e:
        return e

    data     = request.get_json(force=True) or {}
    wind     = float(data.get("wind_kmh", 0.0))
    code     = int(data.get("weather_code", 0))
    hum      = float(data.get("humidity_pct", 50.0))
    dangerous = code in (82, 95, 96, 99) or wind > 50
    caution   = code in (61, 63, 65, 71, 75, 80, 81) or wind > 25

    weather = WeatherData(
        wind_speed_kmh     = wind,
        wind_direction_deg = float(data.get("wind_direction", 0.0)),
        temperature_c      = float(data.get("temperature_c",  25.0)),
        humidity_pct       = hum,
        visibility_km      = float(data.get("visibility_km",  10.0)),
        weather_code       = code,
        is_dangerous       = dangerous,
        is_caution         = caution,
    )
    fleet.update_weather(weather)
    log.info("Weather updated: wind=%.1f km/h, code=%d, danger=%s",
             wind, code, dangerous)

    return ok({
        "wind_kmh":     wind,
        "speed_mod":    weather.speed_modifier,
        "batt_mod":     weather.battery_drain_modifier,
        "is_dangerous": dangerous,
        "is_caution":   caution,
        "drift":        weather.wind_drift,
    }, "Weather updated — physics effects applied to all drones")


@app.route("/api/obstacle/add", methods=["POST"])
def add_obstacle():
    """
    Add a dynamic obstacle (bird, building, NFZ point).

    Body: { "lat": 17.386, "lng": 78.488, "alt": 60.0 }
    """
    e = require_fleet()
    if e:
        return e
    data = request.get_json(force=True) or {}
    obs  = GPSCoord(
        lat=float(data.get("lat", 0)),
        lng=float(data.get("lng", 0)),
        alt=float(data.get("alt", 60.0)),
    )
    fleet.add_obstacle(obs)
    log.info("Obstacle added at (%.5f, %.5f)", obs.lat, obs.lng)
    return ok({"count": len(fleet.obstacles)}, "Obstacle added — drones will avoid")


@app.route("/api/obstacle/clear", methods=["POST"])
def clear_obstacles():
    """Remove all dynamic obstacles."""
    e = require_fleet()
    if e:
        return e
    fleet.clear_obstacles()
    return ok({}, "All obstacles cleared")


# ── Feature 3: Advanced Flight Controller — control mode per drone ──

@app.route("/api/drone/<drone_id>/control_mode", methods=["POST"])
def set_drone_control_mode(drone_id: str):
    """
    Switch a single drone's flight controller to PID / LQR / MPC / OPEN_LOOP.

    Body: { "mode": "PID" | "LQR" | "MPC" | "OPEN_LOOP" }

    The AdvancedFlightController uses the same compute_velocity() signature
    as the original DroneFlightController, so the swap is transparent to
    the rest of the drone's state machine.
    """
    e = require_fleet()
    if e:
        return e

    from advanced_flight_controller import AdvancedFlightController, ControlMode

    data = request.get_json(force=True) or {}
    mode_str = data.get("mode", "PID").upper()

    try:
        mode = ControlMode[mode_str]
    except KeyError:
        return err(f"Unknown mode '{mode_str}'. Valid: PID, LQR, MPC, OPEN_LOOP")

    with fleet._lock:
        drone = fleet.drones.get(drone_id)
        if drone is None:
            return err(f"Drone '{drone_id}' not found", 404)
        # Swap flight controller in-place — same interface, drop-in compatible
        drone.fc = AdvancedFlightController(drone_id, control_mode=mode)

    log.info("[%s] Control mode switched to %s", drone_id, mode_str)
    return ok({"drone_id": drone_id, "control_mode": mode_str},
              f"Drone {drone_id} now using {mode_str} controller")


@app.route("/api/fleet/control_mode", methods=["POST"])
def set_fleet_control_mode():
    """
    Switch ALL drones to the same control mode simultaneously.

    Body: { "mode": "PID" | "LQR" | "MPC" | "OPEN_LOOP" }
    """
    e = require_fleet()
    if e:
        return e

    from advanced_flight_controller import AdvancedFlightController, ControlMode

    data = request.get_json(force=True) or {}
    mode_str = data.get("mode", "PID").upper()

    try:
        mode = ControlMode[mode_str]
    except KeyError:
        return err(f"Unknown mode '{mode_str}'. Valid: PID, LQR, MPC, OPEN_LOOP")

    with fleet._lock:
        for did, drone in fleet.drones.items():
            drone.fc = AdvancedFlightController(did, control_mode=mode)

    log.info("[Fleet] All drones switched to %s controller", mode_str)
    return ok({"control_mode": mode_str, "drones": list(fleet.drones.keys())},
              f"All {len(fleet.drones)} drones switched to {mode_str}")


@app.route("/api/controller/compare", methods=["GET"])
def compare_controllers():
    """
    Real delivery-based controller comparison.

    Groups all completed missions by the controller mode that flew them,
    then computes per-mode averages for:
        - delivery time (s)
        - energy consumed (Wh)
        - distance flown (km)
        - ORCA avoidance events
        - path waypoints (A* complexity)

    Falls back to a synthetic 1-D step-response benchmark only when
    fewer than 2 real deliveries exist (clearly labelled as synthetic).
    """
    e = require_fleet()
    if e:
        return e

    with fleet._lock:
        completed = list(fleet.completed_missions)

    # ── Real delivery data ──────────────────────────────────────
    from collections import defaultdict
    buckets = defaultdict(list)
    for m in completed:
        if m.status.value == "delivered" and m.elapsed_s and m.elapsed_s > 0:
            buckets[m.controller_mode].append(m)

    if sum(len(v) for v in buckets.values()) >= 1:
        results = {}
        for mode, missions in sorted(buckets.items()):
            times    = [m.elapsed_s  for m in missions if m.elapsed_s]
            energies = [m.energy_wh  for m in missions]
            dists    = [m.dist_km    for m in missions]
            colls    = [m.collisions_avoided for m in missions]
            wpts     = [m.path_waypoints     for m in missions]

            def avg(lst): return round(sum(lst)/len(lst), 3) if lst else None
            def mn(lst):  return round(min(lst), 3) if lst else None
            def mx(lst):  return round(max(lst), 3) if lst else None

            results[mode] = {
                "deliveries":         len(missions),
                "avg_time_s":         avg(times),
                "min_time_s":         mn(times),
                "max_time_s":         mx(times),
                "avg_energy_wh":      avg(energies),
                "avg_dist_km":        avg(dists),
                "avg_orca_events":    avg(colls),
                "avg_waypoints":      avg(wpts),
                "data_source":        "real_deliveries",
            }

        # Winner = fastest average delivery time
        ranked = [(mode, r["avg_time_s"] or 9999)
                  for mode, r in results.items() if r["avg_time_s"]]
        winner = min(ranked, key=lambda x: x[1])[0] if ranked else "N/A"

        return jsonify({
            "data_source":   "real_deliveries",
            "total_compared": sum(len(v) for v in buckets.values()),
            "controllers":   results,
            "winner":        winner,
            "note": "Based on actual drone deliveries. Switch controller mode and run more deliveries to build comparison data.",
        })

    # ── Fallback: synthetic step-response (clearly labelled) ───
    from advanced_flight_controller import (
        PIDController, OpenLoopController, LQRAxisController, MPCController,
        MAX_SPEED, DT, _step_response_closed_loop, _step_response_open_loop, _metrics,
    )

    step_m = float(request.args.get("step_m", 50.0))
    ticks  = int(request.args.get("ticks", 150))

    synth = {}

    pid = PIDController(kp=1.8, ki=0.05, kd=0.40, out_min=-MAX_SPEED, out_max=MAX_SPEED)
    t = _step_response_closed_loop(pid, step_m, ticks)
    synth["PID"] = {**_metrics(t, step_m), "data_source": "synthetic_benchmark", "deliveries": 0}

    ol = OpenLoopController(max_speed=MAX_SPEED)
    t = _step_response_open_loop(ol, step_m, ticks)
    synth["OPEN_LOOP"] = {**_metrics(t, step_m), "data_source": "synthetic_benchmark", "deliveries": 0}

    lqr = LQRAxisController(out_min=-MAX_SPEED, out_max=MAX_SPEED)
    t = _step_response_closed_loop(lqr, step_m, ticks)
    synth["LQR"] = {**_metrics(t, step_m), "data_source": "synthetic_benchmark", "deliveries": 0}

    mpc = MPCController(out_min=-MAX_SPEED, out_max=MAX_SPEED)
    t = _step_response_closed_loop(mpc, step_m, ticks)
    synth["MPC"] = {**_metrics(t, step_m), "data_source": "synthetic_benchmark", "deliveries": 0}

    winner = min(
        (k for k in synth if synth[k].get("settling_time_s") is not None),
        key=lambda k: synth[k].get("settling_time_s") or 9999,
        default="LQR"
    )
    return jsonify({
        "data_source":    "synthetic_benchmark",
        "total_compared": 0,
        "controllers":    synth,
        "winner":         winner,
        "note": "⚠ No real deliveries yet. This is a synthetic 1-D step-response test. Switch controller mode, run deliveries, then re-check to see real comparison data.",
    })


@app.route("/api/nfz", methods=["POST"])
def update_nfz():
    """
    Push hard no-fly zones from the frontend so A* routes around them.
    Body: { "zones": [{"lat": .., "lng": .., "radius_m": ..}, ...] }
    Drones also climb to 160 m when within 2.5× the zone radius.
    """
    e = require_fleet()
    if e:
        return e
    data = request.get_json(force=True) or {}
    zones = data.get("zones", [])
    validated = []
    for z in zones:
        try:
            validated.append({
                "lat":      float(z["lat"]),
                "lng":      float(z["lng"]),
                "radius_m": float(z.get("radius_m", 100)),
            })
        except (KeyError, TypeError, ValueError):
            continue
    fleet.update_nfz(validated)
    return ok({"count": len(validated)}, f"NFZ list updated — {len(validated)} zones")


@app.route("/api/analytics", methods=["GET"])
def get_analytics():
    """
    Research-grade analytics snapshot.
    Includes per-drone stats, energy model, MAVLink log, mission history.
    """
    e = require_fleet()
    if e:
        return e

    with fleet._lock:
        drones = list(fleet.drones.values())
        total_del   = sum(d.deliveries_completed for d in drones)
        total_abort = sum(d.missions_aborted     for d in drones)
        total_mis   = total_del + total_abort
        success_rate = (total_del / total_mis * 100) if total_mis > 0 else 0.0

        per_drone = []
        for d in drones:
            avg_e = (d.energy_used_wh / d.deliveries_completed
                     if d.deliveries_completed > 0 else 0.0)
            per_drone.append({
                "id":                  d.id,
                "color":               d.color,
                "state":               d.state.value,
                "battery_pct":         round(d.battery, 1),
                "km_flown":            round(d.km_flown, 3),
                "deliveries":          d.deliveries_completed,
                "missions_aborted":    d.missions_aborted,
                "energy_wh":           round(d.energy_used_wh, 3),
                "avg_energy_per_del":  round(avg_e, 3),
                "collisions_avoided":  d.collisions_avoided,
                "total_hover_s":       round(d.total_hover_s, 1),
                "current_speed_kmh":   round(d.velocity.magnitude() * 3.6, 1),
                "altitude_m":          round(d.gps.alt, 1),
                "kalman_uncertainty_m": round(d.kalman.uncertainty_m, 2),
                "soh":                 round(d.soh, 1),
                "cycle_count":         d.cycle_count,
            })

        return jsonify({
            "summary": {
                "total_missions":     total_mis,
                "delivered":          total_del,
                "aborted":            total_abort,
                "success_rate_pct":   round(success_rate, 1),
                "total_km":           round(sum(d.km_flown for d in drones), 3),
                "total_energy_wh":    round(sum(d.energy_used_wh for d in drones), 3),
                "collisions_avoided": sum(d.collisions_avoided for d in drones),
                "active_drones":      sum(
                    1 for d in drones
                    if d.state not in (FlightState.IDLE, FlightState.CHARGING)
                ),
            },
            "energy_model": {
                "formula":           "E = K1·dist_km + K2·payload_kg + K3·wind_kmh + K4·hover_s",
                "K1_dist":           0.12,
                "K2_payload":        0.08,
                "K3_wind":           0.04,
                "K4_hover":          0.15,
                "weather_modifier":  round(fleet.weather.battery_drain_modifier, 2),
                "speed_modifier":    round(fleet.weather.speed_modifier, 2),
            },
            "algorithms": {
                "path_planner":        ("Theta* Advanced A* (any-angle, line-of-sight)"
                                        if fleet.active_algo == "advanced"
                                        else "Basic A* 8-directional grid (CELL_SIZE=0.0004°≈44m)"),
                "active_algo":         fleet.active_algo,
                "state_estimation":    "Kalman Filter 6-DOF (pure Python, no numpy)",
                "collision_avoidance": "ORCA (van den Berg et al.)",
                "flight_control":      "AdvancedFlightController — PID / LQR / MPC / Open-Loop (switchable per drone)",
                "dynamics_6dof":       "Full 6-DOF rigid-body: x,y,z + Roll,Pitch,Yaw via Newton-Euler",
                "task_assignment":     "Hungarian-inspired greedy O(n² log n)",
                "protocol":            "MAVLink-style JSON messages",
                "control_modes":       ["PID (classical closed-loop)", "LQR (modern optimal)",
                                        "MPC (receding-horizon)", "Open-Loop (classical open)"],
            },
            "per_drone": per_drone,
            "weather": {
                "wind_kmh":     fleet.weather.wind_speed_kmh,
                "is_dangerous": fleet.weather.is_dangerous,
                "is_caution":   fleet.weather.is_caution,
                "speed_mod":    fleet.weather.speed_modifier,
                "batt_mod":     fleet.weather.battery_drain_modifier,
                "drift":        fleet.weather.wind_drift,
            },
            "mavlink_log": fleet.get_mavlink_log()[:20],
        })


@app.route("/api/mavlink", methods=["GET"])
def get_mavlink():
    """Latest MAVLink messages across all drones."""
    e = require_fleet()
    if e:
        return e
    return jsonify({"messages": fleet.get_mavlink_log()})


@app.route("/api/stream", methods=["GET"])
def stream_state():
    """
    Server-Sent Events stream (2 Hz).  Alternative to polling.

    Usage in frontend JS:
        const es = new EventSource('http://localhost:5000/api/stream');
        es.onmessage = e => { const state = JSON.parse(e.data); updateUI(state); };
    """
    e = require_fleet()
    if e:
        return e

    def generate():
        while True:
            try:
                state = fleet.get_state()
                yield f"data: {json.dumps(state)}\n\n"
                time.sleep(0.5)
            except GeneratorExit:
                break
            except Exception as ex:
                log.error("SSE stream error: %s", ex)
                yield f"data: {json.dumps({'error': str(ex)})}\n\n"
                time.sleep(1.0)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ─────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────

import math
import os
import numpy as np
from flask import send_file
import drone_agent as _da
import trajectory_generation as tg
from mission_replanner import V6Coordinator

# =============================================================================
#  V6 EXTENSION: queue, online replanning, per-drone history, live dashboard API
# =============================================================================
V6_COLORS = ["#3b82f6", "#22c55e", "#ef4444", "#f97316", "#a855f7",
             "#06b6d4", "#eab308", "#ec4899", "#ffffff"]          # Drone 1..9
coord = V6Coordinator(lambda: fleet)


def _drone(did):
    f = fleet
    return f.drones.get(did) if f else None


# ---------- automatic replans on V5 events ----------
_TRIGGERS = {"/api/obstacle/add": "obstacle", "/api/nfz": "no_fly_zone", "/api/weather": "weather"}


@app.after_request
def _auto_replan(resp):
    reason = _TRIGGERS.get(request.path)
    if reason and request.method == "POST" and resp.status_code < 300 and fleet:
        coord.replan_fleet(reason)
    return resp


# ---------- missions ----------
@app.route("/api/v6/mission", methods=["POST"])
def v6_mission():
    d = request.get_json(force=True) or {}
    for k in ("pickup_lat", "pickup_lng", "deliver_lat", "deliver_lng"):
        if k not in d:
            return err(f"Missing field: {k}")
    m = coord.add_mission(GPSCoord(float(d["pickup_lat"]), float(d["pickup_lng"]), 0.0),
                          GPSCoord(float(d["deliver_lat"]), float(d["deliver_lng"]), 0.0),
                          d.get("pickup_name", "Pickup"), d.get("deliver_name", "Destination"),
                          float(d.get("payload_kg", 1.0)), d.get("priority", "standard"), d.get("drone_id"))
    return ok(m.to_dict(), "Mission queued") if m else err("No drone available", 503)


@app.route("/api/v6/queue", methods=["GET"])
@app.route("/api/v6/queue/<drone_id>", methods=["GET"])
def v6_queue(drone_id=None):
    return ok(coord.queue_view(drone_id))


@app.route("/api/v6/drone/<drone_id>/fail", methods=["POST"])
def v6_fail(drone_id):
    if not _drone(drone_id):
        return err("Unknown drone", 404)
    return ok({"reassigned": coord.fail_drone(drone_id)}, "Drone failed, missions reassigned")


# ---------- replanning ----------
@app.route("/api/v6/replan", methods=["POST"])
def v6_replan():
    d = request.get_json(force=True) or {}
    reason = d.get("reason", "new_delivery")
    done = coord.replan_fleet(reason) if not d.get("drone_id") else \
        ([d["drone_id"]] if coord.request_replan(d["drone_id"], reason) else [])
    return ok({"replanned": done, "battery_rtb": coord.check_battery()})


@app.route("/api/v6/drone/<drone_id>/paths", methods=["GET"])
def v6_paths(drone_id):
    """old (red) and new (green) path after the latest replan, plus current path."""
    dr = _drone(drone_id)
    if not dr:
        return err("Unknown drone", 404)
    p = coord.paths.get(drone_id, {})
    return ok(dict(old=p.get("old"), new=p.get("new"), reason=p.get("reason"),
                   current=[list(x) for x in dr.path[dr.path_idx:]]))


# ---------- per-drone analytics ----------
@app.route("/api/v6/drone/<drone_id>/history", methods=["GET"])
def v6_history(drone_id):
    if not _drone(drone_id):
        return err("Unknown drone", 404)
    keys = request.args.get("keys")
    h = coord.history(drone_id, float(request.args.get("since", 0)), keys.split(",") if keys else None)
    return ok(dict(drone_id=drone_id, state=coord.v6_state(_drone(drone_id)), series=h))


@app.route("/api/v6/fleet/history", methods=["GET"])
def v6_fleet_history():
    keys = ["t", "x_m", "y_m", "z_m"]
    return ok({did: coord.history(did, float(request.args.get("since", 0)), keys) for did in fleet.drones})


@app.route("/api/v6/analytics", methods=["GET"])
def v6_analytics():
    return ok(coord.analytics())


# ---------- trajectories ----------
def _pts(d):
    return np.array(d["start"], float), np.array(d["goal"], float), (np.array(d["via"], float) if d.get("via") else None)


@app.route("/api/v6/trajectory/compare", methods=["POST"])
def v6_compare():
    d = request.get_json(force=True) or {}
    if "start" not in d or "goal" not in d:
        return err("Need start and goal as [x, y, z] metres")
    s, g, v = _pts(d)
    return ok(tg.compare(s, g, v, float(d.get("speed", 12.0))))


@app.route("/api/v6/trajectory/generate", methods=["POST"])
def v6_generate():
    d = request.get_json(force=True) or {}
    if d.get("method") not in tg.METHODS:
        return err(f"method must be one of {tg.METHODS}")
    s, g, v = _pts(d)
    tr = tg.generate(d["method"], s, g, v, float(d.get("speed", 12.0)))
    out = tg.to_json(tr, float(d.get("dt", 0.1)))
    out["metrics"] = tr.metrics()
    return ok(out)


@app.route("/api/v6/trajectory/bspline/modify", methods=["POST"])
def v6_bspline_modify():
    d = request.get_json(force=True) or {}
    tr = tg.bspline(d["control_points"], float(d.get("speed", 12.0)))
    tr = tg.modify_control_point(tr, int(d["index"]), d["new_point"], float(d.get("speed", 12.0)))
    return ok(tg.to_json(tr))


@app.route("/api/v6/health", methods=["GET"])
def v6_health():
    return ok({"version": "V6", "drones": len(fleet.drones) if fleet else 0,
               "methods": list(tg.METHODS), "counters": coord.counters})


# ---------- live dashboard data (common ENU frame: x = north, y = east, z = up, metres) ----------
R_E, DEG, CRUISE = 6_371_000.0, math.pi / 180.0, 80.0


def _ref():
    return next(iter(fleet.drones.values())).home


def _enu(lat, lng, ref):
    return (lat - ref.lat) * R_E * DEG, (lng - ref.lng) * R_E * math.cos(lat * DEG) * DEG


def _gps(x, y, ref):
    lat = ref.lat + x / (R_E * DEG)
    return GPSCoord(lat, ref.lng + y / (R_E * DEG * math.cos(lat * DEG)), 0.0)


def _path(pts, ref, z=CRUISE):
    return [[*_enu(p[0], p[1], ref), z] for p in pts or []]


def _mission(m, ref):
    if not m:
        return None
    return dict(m.to_dict(), pickup_xy=_enu(m.pickup.lat, m.pickup.lng, ref),
                deliver_xy=_enu(m.delivery.lat, m.delivery.lng, ref))


def _page(name):
    return send_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), name))


@app.route("/")
def v5_index():
    return _page("SkyLogix_v5.html")      # main demo site


@app.route("/v6")
def v6_index():
    return _page("skylogix_v6.html")      # research analytics


@app.route("/api/v6/overview", methods=["GET"])
def v6_overview():
    f, ref = fleet, _ref()
    st, ds = f.get_state(), []
    for did, d in f.drones.items():
        x, y = _enu(d.lat, d.lng, ref)
        ds.append(dict(id=did, color=d.color, state=coord.v6_state(d), raw_state=d.state.value, x=x, y=y, z=d.alt,
                       battery=d.battery, fault=d.fault_type, mission=_mission(d.mission, ref),
                       queue_len=len(coord.queues.get(did, [])), replans=getattr(d, "replan_count", 0),
                       deliveries=d.deliveries_completed, km=d.km_flown, home=_enu(d.home.lat, d.home.lng, ref)))
    return ok(dict(drones=ds, weather=st["weather"], summary=st["summary"], counters=coord.counters, method=coord.method,
                   obstacles=[dict(x=_enu(o.lat, o.lng, ref)[0], y=_enu(o.lat, o.lng, ref)[1], r=50) for o in f.obstacles],
                   log=[f"[replan] {e['drone']}: {e['reason']}" for e in coord.replans[-20:]][::-1] + st["dispatch_log"][:15]))


@app.route("/api/v6/drone/<drone_id>/detail", methods=["GET"])
def v6_detail(drone_id):
    d = _drone(drone_id)
    if not d:
        return err("Unknown drone", 404)
    ref = _ref()
    ox, oy = _enu(d.home.lat, d.home.lng, ref)
    h = {k: list(v) for k, v in coord.history(drone_id, float(request.args.get("since", 0))).items()}
    if h:
        for kx, ky in (("x_m", "y_m"), ("des_x_m", "des_y_m")):
            if kx in h:
                h[kx], h[ky] = [v + ox for v in h[kx]], [v + oy for v in h[ky]]
    p = coord.paths.get(drone_id, {})
    return ok(dict(series=h, state=coord.v6_state(d), battery=d.battery, mission=_mission(d.mission, ref),
                   queue=coord.queue_view(drone_id)[drone_id],
                   history=[dict(id=k, **v) for k, v in coord.delivered.items() if v["drone"] == drone_id],
                   current=_path(d.path[d.path_idx:], ref), old=_path(p.get("old"), ref), new=_path(p.get("new"), ref),
                   reason=p.get("reason"), current_metrics=coord.current_metrics(drone_id),
                   lifetime=coord.lifetime(drone_id), delivery_history=coord.delivery_history(drone_id),
                   events=coord.events.get(drone_id, [])[-80:], flown_method=getattr(d, "traj_method", None),
                   limits=dict(horizontal_accel=_da.A_MAX, vertical_accel=_da.AZ_MAX, cruise_alt=_da.CRUISE_ALT)))


@app.route("/api/v6/fleet/trajectories", methods=["GET"])
def v6_fleet_traj():
    ref, out = _ref(), {}
    for did, d in fleet.drones.items():
        ox, oy = _enu(d.home.lat, d.home.lng, ref)
        h = coord.hist.get(did, {})
        n = min(600, len(h.get("t", ())))
        p = coord.paths.get(did, {})
        out[did] = dict(color=d.color, state=coord.v6_state(d),
                        past=[[h["x_m"][-n + i] + ox, h["y_m"][-n + i] + oy, h["z_m"][-n + i]] for i in range(n)],
                        future=_path(d.path[d.path_idx:], ref), old=_path(p.get("old"), ref), new=_path(p.get("new"), ref))
    return ok(out)


@app.route("/api/v6/mission/enu", methods=["POST"])
def v6_mission_enu():
    d, ref = request.get_json(force=True) or {}, _ref()
    m = coord.add_mission(_gps(float(d["pickup_x"]), float(d["pickup_y"]), ref), _gps(float(d["deliver_x"]), float(d["deliver_y"]), ref),
                          d.get("pickup_name", "Pickup"), d.get("deliver_name", "Destination"), 1.0, d.get("priority", "standard"),
                          d.get("drone_id"), d.get("method"))
    return ok(m.to_dict(), "Mission queued") if m else err("No drone available", 503)


@app.route("/api/v6/obstacle/enu", methods=["POST"])
def v6_obstacle_enu():
    d = request.get_json(force=True) or {}
    fleet.add_obstacle(_gps(float(d["x"]), float(d["y"]), _ref()))
    return ok({"replanned": coord.replan_fleet("obstacle")}, "Obstacle added")



@app.route("/api/v6/drone/<drone_id>/current_metrics", methods=["GET"])
def v6_current_metrics(drone_id):
    return ok(coord.current_metrics(drone_id)) if _drone(drone_id) else err("Unknown drone", 404)


@app.route("/api/v6/drone/<drone_id>/lifetime_metrics", methods=["GET"])
def v6_lifetime_metrics(drone_id):
    return ok(coord.lifetime(drone_id)) if _drone(drone_id) else err("Unknown drone", 404)


@app.route("/api/v6/drone/<drone_id>/delivery_history", methods=["GET"])
def v6_delivery_history(drone_id):
    return jsonify({"status": "ok", "message": "success", "data": coord.delivery_history(drone_id)}) if _drone(drone_id) else err("Unknown drone", 404)


@app.route("/api/v6/fleet/analytics", methods=["GET"])
def v6_fleet_analytics():
    return ok(coord.fleet_analytics())


@app.route("/api/v6/research/analytics", methods=["GET"])
def v6_research_analytics():
    return ok(coord.research_analytics())


@app.route("/api/v6/trajectory/method", methods=["GET", "POST"])
def v6_method():
    if request.method == "POST":
        m = (request.get_json(force=True) or {}).get("method")
        if m not in tg.METHODS:
            return err(f"method must be one of {tg.METHODS}")
        coord.set_method(m)
    return ok({"method": coord.method, "methods": list(tg.METHODS)})


@app.route("/api/v6/drone/<drone_id>/diagnostics", methods=["GET"])
def v6_diag(drone_id):
    return ok(coord.diagnostics(drone_id)) if _drone(drone_id) else err("Unknown drone", 404)


@app.route("/api/v6/config", methods=["GET", "POST"])
def v6_config():
    b = (request.get_json(force=True) or {}) if request.method == "POST" else {}
    return ok(_da.configure(b.get("cruise"), b.get("max_vspeed"), b.get("accel"), b.get("vaccel")))


@app.route("/api/v6/debug/battery", methods=["POST"])
def v6_debug_battery():
    if not os.environ.get("SKY_DEBUG"):
        return err("debug endpoints disabled (set SKY_DEBUG=1)", 403)
    b = request.get_json(force=True) or {}
    _drone(b["drone_id"]).battery = float(b["battery"])
    return ok({"battery": _drone(b["drone_id"]).battery})



# ---------- six-drone controller benchmark (Open-Loop / PID / MPC / LQR) ----------
import advanced_flight_controller as _cb
_CTL_LAST = {"data": None}


@app.route("/api/v6/controllers/run", methods=["POST", "GET"])
def v6_controllers_run():
    """Runs a fresh benchmark (identical plant, references, wind for every controller)."""
    b = (request.get_json(silent=True) or {}) if request.method == "POST" else request.args
    try:
        _CTL_LAST["data"] = _cb.run_benchmark(float(b.get("duration", 40)), int(b.get("seed", 7)), float(b.get("wind", 1.0)))
    except (ValueError, TypeError) as ex:
        return err(str(ex))
    return ok(_CTL_LAST["data"])


@app.route("/api/v6/controllers/latest", methods=["GET"])
def v6_controllers_latest():
    return ok(_CTL_LAST["data"])


@app.before_request
def _v6_start():
    if fleet is not None and not coord._run:
        coord.start()


if __name__ == "__main__":
    base_gps = GPSCoord(lat=17.385, lng=78.4867, alt=0.0)
    fleet = FleetManager(base_gps=base_gps, weather=WeatherData(wind_speed_kmh=8.0))
    for i in range(9):
        fleet.add_drone(f"Drone-{i + 1}", V6_COLORS[i])
    fleet.start_background()
    coord.start()
    log.info("SkyLogix V5+V6 started: 9 drones, dashboard at http://localhost:5000/")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
