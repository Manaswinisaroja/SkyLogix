"""
flask_server_robotics.py — SkyLogix UAV Research Platform v3.0
===============================================================
Flask REST API + SSE bridge between the HTML frontend and the
real DroneAgent fleet running in drone_agent.py.

Endpoints:
  POST /api/init              → Initialize the fleet
  GET  /api/state             → Full fleet state (polling, ~500 ms)
  POST /api/mission           → Assign a new delivery mission
  POST /api/drone/add         → Add a new drone to the fleet
  POST /api/weather           → Update weather / physics
  POST /api/obstacle/add      → Add a dynamic obstacle (bird / NFZ)
  POST /api/obstacle/clear    → Clear all obstacles
  GET  /api/drone/<id>        → Single drone state
  GET  /api/analytics         → Research analytics snapshot
  GET  /api/stream            → SSE live-stream (alt to polling)
  GET  /health                → Health check

Architecture:
  Browser (HTML/JS)
      ↕  REST JSON  (fetch every 500 ms, or EventSource for SSE)
  Flask Server  (this file)
      ↕  Python objects
  FleetManager  →  DroneAgent × N
      ↕  PID + Kalman + ORCA + A* + MAVLink  (10 Hz loop)
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
        "algorithms":    ["PID", "A*", "Kalman6DOF", "ORCA", "MAVLink",
                          "HungarianAssignment", "SwarmStateMachine"],
    }, "SkyLogix UAV Server v3.0 online")


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
                "path_planner":       ("Theta* Advanced A* (any-angle, line-of-sight)"
                                       if fleet.active_algo == "advanced"
                                       else "Basic A* 8-directional grid (CELL_SIZE=0.0004°≈44m)"),
                "active_algo":        fleet.active_algo,
                "state_estimation":   "Kalman Filter 6-DOF (pure Python, no numpy)",
                "collision_avoidance":"ORCA (van den Berg et al.)",
                "flight_control":     "Cascaded PID (position → velocity)",
                "task_assignment":    "Hungarian-inspired greedy O(n² log n)",
                "protocol":          "MAVLink-style JSON messages",
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

if __name__ == "__main__":
    print("=" * 64)
    print("  SkyLogix UAV Research Platform — Flask Server v3.0")
    print("  Algorithms: PID · A* · Kalman6DOF · ORCA · MAVLink")
    print("=" * 64)
    print()
    print("  Endpoints:")
    print("    POST /api/init           Initialize fleet")
    print("    GET  /api/state          Full fleet snapshot (poll 500 ms)")
    print("    POST /api/mission        Assign delivery mission")
    print("    POST /api/drone/add      Add drone to fleet")
    print("    POST /api/weather        Update weather physics")
    print("    POST /api/obstacle/add   Add dynamic obstacle")
    print("    GET  /api/analytics      Research metrics + MAVLink log")
    print("    GET  /api/stream         SSE live stream")
    print("    GET  /health             Health check")
    print()
    print("  Quick start:")
    print('    curl -X POST http://localhost:5000/api/init \\')
    print('      -H "Content-Type: application/json" \\')
    print("      -d '{\"base_lat\":17.385,\"base_lng\":78.4867,\"num_drones\":3}'")
    print()
    print("  Starting server on http://0.0.0.0:5000 ...")
    print()

    # Auto-initialize a demo fleet (Vijayawada) on startup
    base    = GPSCoord(lat=17.385, lng=78.4867, alt=0.0)
    weather = WeatherData(wind_speed_kmh=8.0)
    fleet   = FleetManager(base_gps=base, weather=weather)
    for i, name in enumerate(["Alpha", "Beta", "Gamma"]):
        fleet.add_drone(name, DRONE_COLORS[name])
    fleet.start_background()
    log.info("Demo fleet auto-initialized with 3 drones")

    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)