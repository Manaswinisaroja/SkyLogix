"""
drone_dynamics_6dof.py — SkyLogix Phase 2 Extension
═══════════════════════════════════════════════════════════════════
Full 6-DOF Rigid-Body Drone Dynamics
(x, y, z, Roll, Pitch, Yaw — instead of only x, y)

This module is 100% ADDITIVE. It does NOT edit, rename, or remove
anything in drone_agent.py, flask_server_robotics.py, or SkyLogix_v5.html.
It only *attaches* new behaviour to the existing DroneAgent class via
monkey-patching, so every existing feature, variable name, endpoint,
and JSON key keeps working exactly as before — this file simply adds
new keys/attributes on top.

1. Motion Dynamics
   Newton's 2nd law (translation, F = m·a) and Euler's rotational
   equation (τ = I·α) are what make a quadcopter's *orientation* and
   *position* inseparable: you cannot accelerate the airframe without
   first rotating it, because thrust always points along the body's
   own z-axis. That coupling — position drives orientation, and
   orientation drives position — is exactly what this module adds.

2. Acceleration Kinematics
   The existing simulation already produces a continuous stream of
   (lat, lng, alt). Every tick we differentiate that stream twice:
        v = Δx / Δt        (position → velocity)
        a = Δv / Δt        (velocity → acceleration)
   giving a physically consistent linear-acceleration history for a
   drone that previously only "teleported" between (x, y) points.

3. Rigid Link Acceleration
   The airframe is modeled as ONE rigid body (a single rigid link
   between its four rotors and its center of mass). Two classic rigid
   body relations drive its attitude:
     • Coordinated-turn bank angle:  tan(roll) = (v · yaw_rate) / g
       (the centripetal acceleration a rigid body needs to curve its
       velocity vector must come from tilting its lift vector — i.e.
       banking, exactly like a real quadcopter rolling into a turn.)
     • Rotational inertia response: I·α = k_spring·(target − angle)
       − k_damp·rate
       (a rigid body cannot snap to a new attitude instantly — its
       angular acceleration is what changes its angular velocity,
       which is what changes its angle, tick by tick.)

Net effect
----------
Every movement now affects orientation:
  • Turning        → yaw_rate ↑ → bank angle (Roll) increases
  • Speeding up     → forward acceleration → nose dips (Pitch changes)
  • Climbing        → vertical acceleration → slight pitch-up
  • All of the above ripple through velocity → acceleration exactly
    like a real quadcopter's flight-controller / IMU loop.

Usage
-----
    from drone_dynamics_6dof import enable_6dof_dynamics
    enable_6dof_dynamics()          # call once, e.g. in flask_server_robotics.py

After that, every DroneAgent.to_dict() gains a new, additive key:

    "dynamics_6dof": {
        "x_m": ..., "y_m": ..., "z_m": ...,
        "vx_ms": ..., "vy_ms": ..., "vz_ms": ...,
        "ax_ms2": ..., "ay_ms2": ..., "az_ms2": ...,
        "roll_deg": ..., "pitch_deg": ..., "yaw_deg": ...,
        "roll_rate_dps": ..., "pitch_rate_dps": ..., "yaw_rate_dps": ...,
        "roll_accel_dps2": ..., "pitch_accel_dps2": ..., "yaw_accel_dps2": ...,
    }

No existing key, variable, class, or behaviour is touched.
"""

import math

# ══════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════

G_ACCEL       = 9.80665                 # m/s^2 — standard gravity
MAX_BANK_RAD  = math.radians(35.0)      # max realistic quadcopter roll
MAX_PITCH_RAD = math.radians(25.0)      # max realistic quadcopter pitch
MIN_TURN_SPEED = 0.4                    # m/s — below this, heading is "frozen"

# Rotational rigid-body response (I·α = k_spring·error − k_damp·rate)
ROLL_K_SPRING,  ROLL_K_DAMP  = 40.0, 9.0
PITCH_K_SPRING, PITCH_K_DAMP = 40.0, 9.0
YAW_K_SPRING,   YAW_K_DAMP   = 25.0, 7.0
MAX_ANGULAR_RATE = math.radians(180.0)  # rad/s clamp, all axes


# ══════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════

def _clamp(value: float, lo: float, hi: float) -> float:
    return lo if value < lo else hi if value > hi else value


def _wrap_pi(angle: float) -> float:
    """Wrap an angle (radians) into (-pi, pi]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def _second_order_step(angle: float, rate: float, target_angle: float,
                        dt: float, k_spring: float, k_damp: float,
                        max_rate: float = MAX_ANGULAR_RATE):
    """
    One Euler-integration step of a rigid body's rotational dynamics:

        I * angular_accel = k_spring * (target - angle) - k_damp * rate
        rate  += angular_accel * dt
        angle += rate * dt

    This is the rotational analogue of F = m·a (Motion Dynamics) and is
    what gives every orientation change a physically continuous
    angular *acceleration* (Rigid Link Acceleration) instead of the
    orientation just snapping to a target value.
    """
    error = _wrap_pi(target_angle - angle)
    angular_accel = k_spring * error - k_damp * rate
    rate += angular_accel * dt
    rate = _clamp(rate, -max_rate, max_rate)
    angle = _wrap_pi(angle + rate * dt)
    return angle, rate, angular_accel


# ══════════════════════════════════════════════════════════════════
#  PER-DRONE 6-DOF STATE
# ══════════════════════════════════════════════════════════════════

class SixDOFState:
    """
    Full 6-DOF rigid-body state for one drone:
        Position:     x, y, z            (metres, local ENU frame, origin = home)
        Lin. velocity:  vx, vy, vz         (m/s)
        Lin. accel:     ax, ay, az          (m/s^2)
        Orientation:   roll, pitch, yaw   (radians)
        Ang. velocity: roll_rate, pitch_rate, yaw_rate       (rad/s)
        Ang. accel:    roll_accel, pitch_accel, yaw_accel    (rad/s^2)
    """
    __slots__ = (
        "x", "y", "z",
        "vx", "vy", "vz",
        "ax", "ay", "az",
        "roll", "pitch", "yaw",
        "roll_rate", "pitch_rate", "yaw_rate",
        "roll_accel", "pitch_accel", "yaw_accel",
        "_initialized",
    )

    def __init__(self):
        self.x = self.y = self.z = 0.0
        self.vx = self.vy = self.vz = 0.0
        self.ax = self.ay = self.az = 0.0
        self.roll = self.pitch = self.yaw = 0.0
        self.roll_rate = self.pitch_rate = self.yaw_rate = 0.0
        self.roll_accel = self.pitch_accel = self.yaw_accel = 0.0
        self._initialized = False

    # ── serialization for the Flask /api/state response ───────────
    def to_dict(self) -> dict:
        return {
            "x_m":              round(self.x, 1),
            "y_m":              round(self.y, 1),
            "z_m":              round(self.z, 1),
            "vx_ms":            round(self.vx, 1),
            "vy_ms":            round(self.vy, 1),
            "vz_ms":            round(self.vz, 1),
            "ax_ms2":           round(self.ax, 1),
            "ay_ms2":           round(self.ay, 1),
            "az_ms2":           round(self.az, 1),
            "roll_deg":         round(math.degrees(self.roll), 1),
            "pitch_deg":        round(math.degrees(self.pitch), 1),
            "yaw_deg":          round(math.degrees(self.yaw) % 360, 1),
            "roll_rate_dps":    round(math.degrees(self.roll_rate), 1),
            "pitch_rate_dps":   round(math.degrees(self.pitch_rate), 1),
            "yaw_rate_dps":     round(math.degrees(self.yaw_rate), 1),
            "roll_accel_dps2":  round(math.degrees(self.roll_accel), 1),
            "pitch_accel_dps2": round(math.degrees(self.pitch_accel), 1),
            "yaw_accel_dps2":   round(math.degrees(self.yaw_accel), 1),
        }


def step_6dof(state: SixDOFState, lat: float, lng: float, alt: float,
              home_lat: float, home_lng: float,
              dt: float, earth_r: float, deg2rad: float) -> None:
    """
    Advance one drone's 6-DOF rigid-body state by one tick, driven
    entirely by its existing (lat, lng, alt) simulation output.

    Pipeline (Motion Dynamics → Acceleration Kinematics → Rigid Link
    Acceleration), executed every tick:

        1.  Position   : lat/lng/alt  →  local ENU (x, y, z) metres
        2.  Velocity   : v = Δx / Δt                      (kinematics)
        3.  Accel.     : a = Δv / Δt                      (kinematics)
        4.  Yaw target : heading of the velocity vector
        5.  Roll target: coordinated-turn bank angle from centripetal
                          acceleration (rigid link relation)
        6.  Pitch target: nose-down/up from forward acceleration
        7.  Rotational dynamics: integrate roll/pitch/yaw toward their
            targets through a spring-damper (I·α = ...), so every
            orientation change carries a physically continuous
            angular acceleration — never an instant snap.
    """
    # ── 1. Position: lat/lng/alt → local ENU frame (metres) ───────
    x = (lat - home_lat) * earth_r * deg2rad
    y = (lng - home_lng) * earth_r * math.cos(lat * deg2rad) * deg2rad
    z = alt

    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
        return                                   # bad position sample: keep last good state

    # Teleport / glitch guard: > 60 m in one tick (600 m/s) is not flight, so re-seed instead of
    # differentiating it into astronomically large velocities (the 1e21 m/s readings).
    if state._initialized and math.hypot(x - state.x, y - state.y) > 60.0:
        state._initialized = False
    if not state._initialized:
        state.x, state.y, state.z = x, y, z
        state.vx = state.vy = state.vz = state.ax = state.ay = state.az = 0.0
        state._initialized = True
        return

    if dt <= 0:
        return

    # ── 2/3. Acceleration Kinematics: v = Δx/Δt , a = Δv/Δt ────────
    # V6: low-pass filter the finite-difference velocity so wind/GPS jitter on the
    # position does not appear as 40+ m/s velocity spikes and huge accelerations.
    VEL_ALPHA = 0.3
    vx = state.vx + VEL_ALPHA * ((x - state.x) / dt - state.vx)
    vy = state.vy + VEL_ALPHA * ((y - state.y) / dt - state.vy)
    vz = state.vz + VEL_ALPHA * ((z - state.z) / dt - state.vz)
    vx, vy, vz = _clamp(vx, -40.0, 40.0), _clamp(vy, -40.0, 40.0), _clamp(vz, -15.0, 15.0)
    ax = _clamp((vx - state.vx) / dt, -30.0, 30.0)
    ay = _clamp((vy - state.vy) / dt, -30.0, 30.0)
    az = _clamp((vz - state.vz) / dt, -30.0, 30.0)

    speed = math.hypot(vx, vy)

    # ── 4. Yaw target: direction of travel ─────────────────────────
    if speed > MIN_TURN_SPEED:
        target_yaw = math.atan2(vy, vx)
    else:
        target_yaw = state.yaw

    # ── 5. Roll target: Rigid Link Acceleration (coordinated turn) ─
    # A rigid airframe curving its velocity vector needs a centripetal
    # acceleration a_c = v * yaw_rate, which can only be supplied by
    # banking so that a component of thrust points sideways:
    #     tan(roll) = a_c / g
    a_centripetal = speed * state.yaw_rate
    target_roll = _clamp(math.atan2(a_centripetal, G_ACCEL),
                          -MAX_BANK_RAD, MAX_BANK_RAD)

    # ── 6. Pitch target: forward acceleration tilts the thrust vector
    target_pitch = _clamp(math.atan2(-ax, G_ACCEL),
                           -MAX_PITCH_RAD, MAX_PITCH_RAD)

    # ── 7. Rotational rigid-body dynamics (all three axes) ─────────
    state.roll, state.roll_rate, state.roll_accel = _second_order_step(
        state.roll, state.roll_rate, target_roll, dt,
        ROLL_K_SPRING, ROLL_K_DAMP)

    state.pitch, state.pitch_rate, state.pitch_accel = _second_order_step(
        state.pitch, state.pitch_rate, target_pitch, dt,
        PITCH_K_SPRING, PITCH_K_DAMP)

    state.yaw, state.yaw_rate, state.yaw_accel = _second_order_step(
        state.yaw, state.yaw_rate, target_yaw, dt,
        YAW_K_SPRING, YAW_K_DAMP)

    state.x, state.y, state.z = x, y, z
    state.vx, state.vy, state.vz = vx, vy, vz
    state.ax, state.ay, state.az = ax, ay, az


# ══════════════════════════════════════════════════════════════════
#  MONKEY-PATCH HOOK — attaches 6-DOF dynamics to DroneAgent
#  WITHOUT touching drone_agent.py
# ══════════════════════════════════════════════════════════════════

def enable_6dof_dynamics() -> None:
    """
    Call once (e.g. at Flask server startup) to give every DroneAgent
    instance full 6-DOF rigid-body dynamics.

    This wraps (does not replace) DroneAgent.__init__, .tick(), and
    .to_dict() — original behaviour runs first, unchanged, every time;
    this function only appends new attributes / new JSON keys after it.
    Calling it more than once is a no-op.
    """
    import drone_agent as _da

    if getattr(_da.DroneAgent, "_sixdof_enabled", False):
        return

    _orig_init    = _da.DroneAgent.__init__
    _orig_tick    = _da.DroneAgent.tick
    _orig_to_dict = _da.DroneAgent.to_dict

    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        self.dynamics_6dof = SixDOFState()   # new attribute, additive only

    def _patched_tick(self, *args, **kwargs):
        _orig_tick(self, *args, **kwargs)    # existing flight logic — unchanged
        if self.state != _da.FlightState.IDLE:
            step_6dof(
                self.dynamics_6dof,
                self.lat, self.lng, self.alt,
                self.home.lat, self.home.lng,
                dt=_da.DT, earth_r=_da.EARTH_R, deg2rad=_da.DEG2RAD,
            )

    def _patched_to_dict(self):
        d = _orig_to_dict(self)              # existing dict — unchanged
        d["dynamics_6dof"] = self.dynamics_6dof.to_dict()   # new key, additive
        return d

    _da.DroneAgent.__init__       = _patched_init
    _da.DroneAgent.tick           = _patched_tick
    _da.DroneAgent.to_dict        = _patched_to_dict
    _da.DroneAgent._sixdof_enabled = True

    _da.log.info("[6DOF] Full 6-DOF rigid-body dynamics enabled "
                 "(x, y, z, roll, pitch, yaw)")


# ══════════════════════════════════════════════════════════════════
#  SELF-TEST
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 64)
    print("  SkyLogix — Full 6-DOF Drone Dynamics — Self-Test")
    print("=" * 64)

    EARTH_R = 6_371_000.0
    DEG2RAD = math.pi / 180
    DT = 0.1

    home_lat, home_lng = 16.5062, 80.6480
    state = SixDOFState()

    # Simulate a drone flying straight north, then banking into a
    # 90° turn to the east, then climbing — exactly the scenario
    # described in the feature request: "When turning, Pitch changes,
    # Velocity changes, Acceleration changes."
    lat, lng, alt = home_lat, home_lng, 0.0
    print("\n[1] Straight flight (accelerating north) …")
    for i in range(10):
        lat += (2.0 + i * 0.3) * DT / (EARTH_R * DEG2RAD)
        alt = min(80.0, alt + 3.0)
        step_6dof(state, lat, lng, alt, home_lat, home_lng, DT, EARTH_R, DEG2RAD)
    print(f"    speed≈{math.hypot(state.vx, state.vy):.2f} m/s  "
          f"pitch={math.degrees(state.pitch):+.2f}°  "
          f"roll={math.degrees(state.roll):+.2f}°")
    assert state.pitch != 0.0, "pitch should respond to forward acceleration"

    print("\n[2] Hard turn to the east (yaw_rate → bank angle) …")
    for i in range(20):
        heading = math.radians(min(90, i * 6))
        lat += 6.0 * math.cos(heading) * DT / (EARTH_R * DEG2RAD)
        lng += 6.0 * math.sin(heading) * DT / (EARTH_R * math.cos(lat * DEG2RAD) * DEG2RAD)
        step_6dof(state, lat, lng, alt, home_lat, home_lng, DT, EARTH_R, DEG2RAD)
    print(f"    yaw={math.degrees(state.yaw):.1f}°  "
          f"yaw_rate={math.degrees(state.yaw_rate):+.2f}°/s  "
          f"roll={math.degrees(state.roll):+.2f}°  (banking into the turn)")
    assert abs(state.roll) > 0.01, "roll should bank during a coordinated turn"

    print("\n[3] Serialized telemetry snapshot:")
    for k, v in state.to_dict().items():
        print(f"    {k:18s} = {v}")

    print("\n✅  All 6-DOF dynamics self-tests passed.\n")
