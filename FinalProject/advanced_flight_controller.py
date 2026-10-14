"""
advanced_flight_controller.py — SkyLogix Advanced Flight Controller Extension
═══════════════════════════════════════════════════════════════════════════
Standalone extension module. Does NOT modify drone_agent.py or
flask_server_robotics.py — it is a separate, self-contained file that can
be imported alongside them.

Adds four interchangeable control paradigms behind the SAME public
interface used by DroneFlightController in drone_agent.py
(compute_velocity() / reset()), so AdvancedFlightController can be dropped
in anywhere DroneFlightController is used today, without touching that
file.

Related syllabus topics → implementation:

  • Classical Control   → PIDController         (closed loop, error feedback)
  • Classical Control   → OpenLoopController     (open loop, no feedback)
  • Modern Control       → LQRAxisController      (optimal state feedback, DARE)
  • Modern Control       → MPCController          (finite-horizon, receding horizon)
  • Closed Loop Control  → PID / LQR / MPC  (all re-measure error every tick)
  • Open Loop Control    → OpenLoopController  (command computed once, replayed
                                                 blindly, never corrected)

All four controllers share one call signature — compute(setpoint,
measurement) -> velocity command (m/s) — except OpenLoopController, whose
compute() deliberately takes no measurement at all (see class docstring),
which is itself the whole point of the classical open-loop vs closed-loop
comparison this module is built to demonstrate.

Pure Python only (no numpy/scipy), matching drone_agent.py's own
"no numpy req" design — the LQR/MPC Riccati-equation math below reuses the
same style of hand-written 2×2 matrix helpers as KalmanFilter6DOF.
"""

import math
import time
from enum import Enum
from typing import List, Optional, Tuple

# ══════════════════════════════════════════════════════════════════
#  CONSTANTS  (mirrors drone_agent.py — kept identical intentionally
#              so gains/limits behave the same way across both files)
# ══════════════════════════════════════════════════════════════════

EARTH_R      = 6_371_000.0   # metres
DEG2RAD      = math.pi / 180
RAD2DEG      = 180 / math.pi
MAX_SPEED    = 15.0           # m/s cruise
MAX_VSPEED   = 3.0            # m/s vertical
TICK_HZ      = 10             # simulation frequency
DT           = 1.0 / TICK_HZ  # seconds per tick


class ControlMode(Enum):
    PID       = "PID"
    OPEN_LOOP = "OPEN_LOOP"
    LQR       = "LQR"
    MPC       = "MPC"


# ══════════════════════════════════════════════════════════════════
#  1. CLASSICAL CONTROL — PID  (closed loop)
# ══════════════════════════════════════════════════════════════════

class PIDController:
    """
    Standard discrete PID with anti-windup clamp.

        u(t) = Kp·e + Ki·∫e dt + Kd·(de/dt)

    Closed-loop: the error e = setpoint − measurement is re-measured and
    fed back into the controller on every single call to compute().
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

    def compute(self, setpoint: float, measurement: float, dt: Optional[float] = None) -> float:
        # dt=None -> wall-clock dt (live flight, unchanged behaviour).
        # dt given -> fixed simulation step (used by the benchmark so results are reproducible).
        now = time.time()
        if dt is None:
            dt = max(now - self._prev_time, 1e-6)
        self._prev_time = now

        error = setpoint - measurement

        # Integrator with anti-windup clamp
        self._integral += error * dt
        self._integral  = max(-self.integral_limit,
                               min(self.integral_limit, self._integral))

        derivative       = (error - self._prev_error) / dt
        self._prev_error = error

        u = self.kp * error + self.ki * self._integral + self.kd * derivative
        return max(self.out_min, min(self.out_max, u))


# ══════════════════════════════════════════════════════════════════
#  2. CLASSICAL CONTROL — OPEN LOOP  (no feedback)
# ══════════════════════════════════════════════════════════════════

class OpenLoopController:
    """
    Pure open-loop controller — the classical-control counterpart to PID.

    plan() computes the command ONCE from the error observed at planning
    time, using a simple kinematic model ("go at max speed for
    distance / max_speed seconds"). compute() then just replays that
    pre-computed velocity on a timer — it never looks at the measurement
    again, so it can never detect or correct for wind drift, mis-modelled
    dynamics, or anything else that pushes the drone off course.

    This is deliberately the weak baseline: comparing it against
    PID / LQR / MPC below is what demonstrates *why* closed-loop feedback
    control is used in real flight controllers.
    """

    def __init__(self, max_speed: float = MAX_SPEED):
        self.max_speed = max_speed
        self._planned_velocity = 0.0
        self._planned_duration = 0.0
        self._elapsed = 0.0
        self._has_plan = False

    def plan(self, setpoint: float, measurement: float) -> None:
        """Compute the command once. Called only when a NEW target is issued."""
        error = setpoint - measurement
        direction = 1.0 if error > 0 else (-1.0 if error < 0 else 0.0)
        self._planned_velocity = direction * self.max_speed
        self._planned_duration = abs(error) / self.max_speed if self.max_speed > 0 else 0.0
        self._elapsed = 0.0
        self._has_plan = True

    def compute(self, dt: float = DT) -> float:
        """Blindly replay the pre-planned command — measurement is never consulted."""
        if not self._has_plan:
            return 0.0
        self._elapsed += dt
        if self._elapsed >= self._planned_duration:
            return 0.0   # "arrived" per the plan, whether that's actually true or not
        return self._planned_velocity

    def reset(self) -> None:
        self._planned_velocity = 0.0
        self._planned_duration = 0.0
        self._elapsed = 0.0
        self._has_plan = False


# ══════════════════════════════════════════════════════════════════
#  3. MODERN CONTROL — LQR  (Linear Quadratic Regulator, closed loop)
# ══════════════════════════════════════════════════════════════════

class LQRAxisController:
    """
    Modern Control — discrete-time, infinite-horizon LQR with integral action.

    FIXES APPLIED:
    1. Actual velocity from measurement delta — not integrated estimate.
       vel_actual = (measurement_k − measurement_{k-1}) / dt
       This uses real GPS/Kalman position changes, matching real flight controllers.

    2. Integral augmentation — augmented state x = [pos_error, velocity, integral]ᵀ
       Eliminates steady-state error under constant disturbances (wind bias).
       Without this, LQR settles with residual error; PID doesn't.

    3. Retuned: r=0.15 (was 0.6) — LQR now comparable aggression to PID.
       The old r=0.6 penalised acceleration so heavily that LQR looked sluggish
       even when it was mathematically optimal for a different cost function.

    Model: 1-D double integrator, augmented state:
        x = [position_error, velocity, integral_of_error]ᵀ
        x_(k+1) = A x_k + B u_k
        A = [[1, -dt, 0], [0, 1, 0], [dt, 0, 1]]   B = [[-0.5dt², dt, 0]]ᵀ

    Cost: J = Σ (xᵀ Q x + u R u)   with Q = diag(q_pos, q_vel, q_int)

    DARE solved via value iteration (pure Python, no numpy/scipy).
    """

    def __init__(self, q_pos: float = 8.0, q_vel: float = 3.0,
                 q_int: float = 0.5, r: float = 0.15,
                 dt: float = DT, out_min: float = -MAX_SPEED, out_max: float = MAX_SPEED,
                 max_accel: float = 8.0):
        self.dt = dt
        self.out_min, self.out_max = out_min, out_max
        self.max_accel = max_accel

        # Augmented 3-state system: [pos_error, velocity, integral_error]
        self._A = [[1.0, -dt, 0.0],
                   [0.0,  1.0, 0.0],
                   [dt,   0.0, 1.0]]
        self._B = [[-0.5 * dt * dt],
                   [dt],
                   [0.0]]
        self._Q = [[q_pos, 0.0,   0.0  ],
                   [0.0,   q_vel, 0.0  ],
                   [0.0,   0.0,   q_int]]
        self._R = [[r]]
        self._K = self._solve_dare_gain()

        # State tracking — actual velocity from measurement delta (Fix 1)
        self._prev_measurement = None   # last GPS/Kalman position measurement
        self._vel_actual       = 0.0   # derived from measurement delta
        self._integral         = 0.0   # integral of position error (Fix 2)
        self._vel_output       = 0.0   # last commanded velocity (for output)

    @staticmethod
    def _transpose(A):
        rows, cols = len(A), len(A[0])
        return [[A[r][c] for r in range(rows)] for c in range(cols)]

    @staticmethod
    def _mm(A, B):
        ra, ca = len(A), len(A[0])
        rb, cb = len(B), len(B[0])
        C = [[0.0]*cb for _ in range(ra)]
        for i in range(ra):
            for j in range(cb):
                C[i][j] = sum(A[i][k]*B[k][j] for k in range(ca))
        return C

    @staticmethod
    def _mv(A, v):
        return [sum(A[i][j]*v[j] for j in range(len(v))) for i in range(len(A))]

    @staticmethod
    def _madd(A, B):
        return [[A[i][j]+B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

    @staticmethod
    def _msub(A, B):
        return [[A[i][j]-B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

    def _solve_dare_gain(self, iters: int = 600, tol: float = 1e-10) -> List[List[float]]:
        """Solve 3×3 DARE via value iteration → K shape 1×3."""
        A, B, Q, R = self._A, self._B, self._Q, self._R
        P = [row[:] for row in Q]
        At, Bt = self._transpose(A), self._transpose(B)
        n = len(A)

        for _ in range(iters):
            AtP   = self._mm(At, P)
            AtPA  = self._mm(AtP, A)
            AtPB  = self._mm(AtP, B)         # n×1
            BtPA  = self._mm(Bt, self._mm(P, A))   # 1×n
            BtPB  = self._mm(Bt, self._mm(P, B))   # 1×1
            denom = R[0][0] + BtPB[0][0]
            if abs(denom) < 1e-12:
                break
            correction = [[AtPB[i][0]*BtPA[0][j]/denom for j in range(n)] for i in range(n)]
            P_new = self._madd(Q, self._msub(AtPA, correction))
            diff  = max(abs(P_new[i][j]-P[i][j]) for i in range(n) for j in range(n))
            P = P_new
            if diff < tol:
                break

        BtPA  = self._mm(Bt, self._mm(P, A))
        BtPB  = self._mm(Bt, self._mm(P, B))
        denom = R[0][0] + BtPB[0][0]
        return [[BtPA[0][j]/denom for j in range(n)]]   # K shape 1×3

    def compute(self, setpoint: float, measurement: float) -> float:
        """
        Compute optimal control using ACTUAL velocity from measurement delta.

        Fix 1: vel_actual = (measurement - prev_measurement) / dt
               Uses real GPS position change — not integrated estimate.

        Fix 2: Integral term eliminates steady-state error under disturbances.
        """
        pos_error = setpoint - measurement

        # Fix 1: derive actual velocity from consecutive GPS measurements
        if self._prev_measurement is None:
            self._vel_actual = 0.0
        else:
            # measurement is -dlat_m (distance to target) — delta is velocity toward target
            dm = measurement - self._prev_measurement
            self._vel_actual = dm / self.dt   # m/s, sign: negative = moving toward target
        self._prev_measurement = measurement

        # Fix 2: integrate error
        self._integral += pos_error * self.dt
        self._integral  = max(-50.0, min(50.0, self._integral))

        # Augmented state: [pos_error, actual_velocity, integral]
        x = [pos_error, self._vel_actual, self._integral]
        u = -self._mv(self._K, x)[0]
        u = max(-self.max_accel, min(self.max_accel, u))

        # Integrate velocity from acceleration command
        self._vel_output += u * self.dt
        self._vel_output  = max(self.out_min, min(self.out_max, self._vel_output))
        return self._vel_output

    def reset(self) -> None:
        self._prev_measurement = None
        self._vel_actual       = 0.0
        self._integral         = 0.0
        self._vel_output       = 0.0


# ══════════════════════════════════════════════════════════════════
#  4. MODERN CONTROL — MPC  (Model Predictive Control, closed loop)
# ══════════════════════════════════════════════════════════════════

class MPCController:
    """
    Modern Control — true receding-horizon Model Predictive Control.

    FIXES APPLIED:
    1. True receding horizon: _plan_horizon() runs EVERY tick with the
       current state. The gain K0 changes each call. This is the defining
       property of MPC — without it you just have a static-gain controller.

    2. Actual velocity from measurement delta — same fix as LQR above.
       vel_actual = (measurement_k - measurement_{k-1}) / dt

    At every control tick this controller:
      1. Measures actual state (pos_error, vel_actual) from GPS/Kalman
      2. Runs backward Riccati recursion over `horizon` future steps
      3. Extracts only the first-step gain K0 (receding horizon principle)
      4. Applies u0* = -K0 * x_current
      5. Next tick: repeats from step 1 with updated measurements

    Unlike LQR (one fixed steady-state gain), this recomputes over a
    FINITE lookahead every tick — tuning `horizon` and `q_terminal`
    independently trades aggressiveness vs smoothness.

    Closed-loop: pos_error and vel_actual are re-measured every call.

    MODEL OPTIONS (model=...)
    "double" (legacy)  x=[e, v], u = accel. Assumes the vehicle follows the command instantly.
    "lag"              x=[e, v, v_cmd], u = rate of change of the velocity command. The inner velocity loop of the
                       airframe is a = g*(v_cmd - v)/tau, so v follows v_cmd with a first-order lag of time constant
                       tau/g and UNIT DC gain (a thrust loss only slows the response, it does not scale the final
                       speed):   v+ = a v + (1-a) v_cmd+,  a = exp(-dt*g/tau).
                       Solved as a real QP with constraints INSIDE the prediction:
                         |u| <= max_accel  (command rate), |v_cmd| <= speed limit, |(v_cmd - v)/tau_eff| <= plant_accel
                       (ADMM, KKT matrix factorised once, warm-started; exits early when the free optimum is feasible).
                       Commands still in flight (delay_ticks) are rolled through the model before planning.
    Sign convention (both models): e = target - position, v = velocity toward the target, v_cmd = commanded velocity.
    The velocity is differenced from the reference-relative error, so it is ALREADY relative to the moving
    reference; do not subtract ff_vel again.
    """

    def __init__(self, horizon: int = 12, q_pos: float = 10.0, q_vel: float = 1.0,
                 q_terminal: float = 25.0, r: float = 0.5, dt: float = DT,
                 out_min: float = -MAX_SPEED, out_max: float = MAX_SPEED,
                 max_accel: float = 9.0, delay_ticks: int = 0, constrained: bool = False, k_int: float = 0.3, int_limit: float = 4.0,
                 vel_alpha: float = 1.0, model: str = "double", tau: float = 0.35, gain: float = 0.94,
                 plant_accel: Optional[float] = None, admm_iters: int = 25):
        self.model = model                       # "double": legacy 2-state [e, v];  "lag": 3-state [e, v, v_cmd]
        self.tau_eff = tau / max(gain, 1e-3)     # inner loop is a = g*(v_cmd - v)/tau  ->  DC gain 1, time constant tau/g
        self.plant_accel = max_accel if plant_accel is None else plant_accel
        self.admm_iters = int(admm_iters)
        self._c_hist, self._adm = [], None
        self.horizon    = horizon
        self.delay_ticks, self._u_hist, self.constrained, self._qp = int(delay_ticks), [], constrained, None
        self.k_int, self.int_limit, self.vel_alpha = k_int, int_limit, vel_alpha   # disturbance estimator + velocity filter
        self._dist = 0.0
        self.dt         = dt
        self.out_min, self.out_max = out_min, out_max
        self.max_accel  = max_accel

        self._A  = [[1.0, -dt], [0.0, 1.0]]
        self._B  = [[-0.5 * dt * dt], [dt]]
        self._Q  = [[q_pos, 0.0], [0.0, q_vel]]
        self._Qf = [[q_terminal, 0.0], [0.0, q_vel]]
        self._R  = [[r]]
        if model == "lag":
            a_ = math.exp(-dt / self.tau_eff)    # exact first-order discretisation of the velocity loop
            # state [e, v, c]; c_{k+1} = c_k + dt*u_k (command rate), v_{k+1} = a v_k + (1-a) c_{k+1}
            self._A  = [[1.0, -dt, 0.0], [0.0, a_, 1.0 - a_], [0.0, 0.0, 1.0]]
            self._B  = [[0.0], [(1.0 - a_) * dt], [dt]]
            self._Q  = [[q_pos, 0.0, 0.0], [0.0, q_vel, 0.0], [0.0, 0.0, 0.0]]
            self._Qf = [[q_terminal, 0.0, 0.0], [0.0, q_vel, 0.0], [0.0, 0.0, 0.0]]
            self.constrained = True              # the lag model is only solved as a QP (constraints inside the prediction)
            self._lag_setup()                    # build now, not on the first control tick

        # Fix 2: actual velocity tracking
        self._prev_measurement = None
        self._vel_actual       = 0.0
        self._vel_output       = 0.0
        self._vel_rel          = 0.0

    @staticmethod
    def _transpose(A):
        rows, cols = len(A), len(A[0])
        return [[A[r][c] for r in range(rows)] for c in range(cols)]

    @staticmethod
    def _mm(A, B):
        ra, ca = len(A), len(A[0])
        rb, cb = len(B), len(B[0])
        C = [[0.0]*cb for _ in range(ra)]
        for i in range(ra):
            for j in range(cb):
                C[i][j] = sum(A[i][k]*B[k][j] for k in range(ca))
        return C

    @staticmethod
    def _mv(A, v):
        return [sum(A[i][j]*v[j] for j in range(len(v))) for i in range(len(A))]

    @staticmethod
    def _madd(A, B):
        return [[A[i][j]+B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

    @staticmethod
    def _msub(A, B):
        return [[A[i][j]-B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

    def _qp_setup(self):
        """Condensed prediction model X = Phi x0 + Gamma U over the horizon (time-invariant, built once)."""
        H, dt = self.horizon, self.dt
        A = np.array(self._A); B = np.array(self._B)
        Q = np.array(self._Q); Qf = np.array(self._Qf)
        Phi = np.zeros((2 * H, 2)); Gam = np.zeros((2 * H, H)); Qb = np.zeros((2 * H, 2 * H))
        Ap = [np.eye(2)]
        for _ in range(H): Ap.append(Ap[-1] @ A)
        for k in range(H):
            Phi[2*k:2*k+2] = Ap[k + 1]
            for j in range(k + 1):
                Gam[2*k:2*k+2, j:j+1] = Ap[k - j] @ B
            Qb[2*k:2*k+2, 2*k:2*k+2] = Qf if k == H - 1 else Q
        Hq = Gam.T @ Qb @ Gam + self._R[0][0] * np.eye(H)
        Hinv, M = np.linalg.inv(Hq), Gam.T @ Qb @ Phi
        self._qp = (Hq, Hinv, M, float(np.linalg.eigvalsh(Hq).max()), -Hinv @ M)   # last: explicit unconstrained gain (H x 2)

    def _solve_qp(self, x0) -> float:
        """Box-constrained QP |u_k| <= max_accel over the horizon (FISTA on the condensed problem).
        Unconstrained optimum is returned directly when it already satisfies the limits."""
        if getattr(self, "_qp", None) is None:
            self._qp_setup()
        Hq, Hinv, M, L, Kx = self._qp
        xv = np.asarray(x0, float)
        U = Kx @ xv                                   # unconstrained optimum via precomputed explicit gain
        a = self.max_accel
        if np.abs(U).max() <= a:
            return float(U[0])
        g = M @ xv
        U = np.clip(U, -a, a); Y = U.copy(); t = 1.0
        for _ in range(60):
            Un = np.clip(Y - (Hq @ Y + g) / L, -a, a)
            tn = 0.5 * (1 + math.sqrt(1 + 4 * t * t))
            Y = Un + (t - 1) / tn * (Un - U); U, t = Un, tn
        return float(U[0])

    # ── 3-state lag model + constrained QP (ADMM) ──────────────────────────────────────────────
    _LAG_CACHE = {}

    def _lag_setup(self):
        key = (self.horizon, self.dt, self.tau_eff, self.max_accel, self.plant_accel, tuple(map(tuple, self._Q)),
               tuple(map(tuple, self._Qf)), self._R[0][0])
        hit = MPCController._LAG_CACHE.get(key)
        if hit is not None:                      # shared read-only matrices; warm-start vectors are per-instance
            self._adm = dict(hit, x=np.zeros(self.horizon), z=np.zeros(len(hit["z"])), y=np.zeros(len(hit["y"])))
            return
        self._lag_build()
        MPCController._LAG_CACHE[key] = dict(self._adm)

    def _lag_build(self):
        """Condensed model X = Phi x0 + Gam U plus linear constraints  lo <= F U + f(x0) <= hi :
             (1) command rate        |u_k| <= max_accel
             (2) command level       |v_cmd,k| <= out limit  (shifted by the reference velocity)
             (3) physical accel      |(v_cmd,k+1 - v_k)/tau_eff| <= plant_accel   (tilt limit, what the airframe really feels)
        Everything is time-invariant, so the KKT matrix is inverted once."""
        H, n = self.horizon, 3
        A, B = np.array(self._A), np.array(self._B)
        Q, Qf = np.array(self._Q), np.array(self._Qf)
        Phi = np.zeros((n * H, n)); Gam = np.zeros((n * H, H)); Qb = np.zeros((n * H, n * H))
        Ap = [np.eye(n)]
        for _ in range(H): Ap.append(Ap[-1] @ A)
        for k in range(H):
            Phi[n*k:n*k+n] = Ap[k + 1]
            for j in range(k + 1):
                Gam[n*k:n*k+n, j:j+1] = Ap[k - j] @ B
            Qb[n*k:n*k+n, n*k:n*k+n] = Qf if k == H - 1 else Q
        P = Gam.T @ Qb @ Gam + self._R[0][0] * np.eye(H)
        M = Gam.T @ Qb @ Phi
        Gv, Gc, Pv, Pc = Gam[1::n], Gam[2::n], Phi[1::n], Phi[2::n]
        Gv_prev = np.vstack([np.zeros((1, H)), Gv[:-1]])          # v_k (k = 0..H-1)
        Pv_prev = np.vstack([np.array([[0.0, 1.0, 0.0]]), Pv[:-1]])
        Fa, Pa = (Gc - Gv_prev) / self.tau_eff, (Pc - Pv_prev) / self.tau_eff
        F = np.vstack([np.eye(H), Gc, Fa])
        rho, sig = 5.0, 1e-6
        Kinv = np.linalg.inv(P + sig * np.eye(H) + rho * F.T @ F)
        self._adm = dict(P=P, M=M, F=F, Pc=Pc, Pa=Pa, rho=rho, sig=sig, Kinv=Kinv, H=H,
                         x=np.zeros(H), z=np.zeros(F.shape[0]), y=np.zeros(F.shape[0]),
                         Kx=-np.linalg.solve(P, M))

    def _solve_lag(self, x0, ff_vel: float = 0.0) -> float:
        if self._adm is None:
            self._lag_setup()
        S = self._adm; H = S["H"]; x0 = np.asarray(x0, float)
        q = S["M"] @ x0
        ua, vmax, vmin = self.max_accel, self.out_max - ff_vel, self.out_min - ff_vel
        lo = np.concatenate([np.full(H, -ua), np.full(H, vmin) - S["Pc"] @ x0, -np.full(H, self.plant_accel) - S["Pa"] @ x0])
        hi = np.concatenate([np.full(H,  ua), np.full(H, vmax) - S["Pc"] @ x0,  np.full(H, self.plant_accel) - S["Pa"] @ x0])
        U0 = S["Kx"] @ x0
        # free optimum already feasible?  (cheap exit, identical to the unconstrained MPC solution)
        FU = S["F"] @ U0
        if np.all(FU >= lo) and np.all(FU <= hi):
            return float(U0[0])
        F, rho, sig, Kinv = S["F"], S["rho"], S["sig"], S["Kinv"]
        x = np.clip(np.concatenate([S["x"][1:], S["x"][-1:]]), -ua, ua)          # warm start: shifted previous solution
        z = np.clip(F @ x, lo, hi); y = S["y"] * 0.0
        for _ in range(self.admm_iters):
            xt = Kinv @ (sig * x - q + F.T @ (rho * z - y))
            zt = F @ xt
            zn = np.clip(zt + y / rho, lo, hi)
            y = y + rho * (zt - zn)
            x, z = xt, zn
        x = np.clip(x, -ua, ua)
        S["x"] = x
        return float(x[0])

    def _compute_lag(self, pos_error: float, ff_vel: float) -> float:
        a_ = math.exp(-self.dt / self.tau_eff)
        v = self._vel_actual     # already relative to the moving reference: it is differenced from the reference-relative error
        e = pos_error
        d = self.delay_ticks
        if d > 0:   # roll through commands already in flight (they act on the plant during the next d ticks)
            for c in self._c_hist[-d:]:
                e -= v * self.dt
                v = a_ * v + (1.0 - a_) * c
        c_last = self._vel_rel
        u = self._solve_lag([e, v, c_last], ff_vel)
        u = max(-self.max_accel, min(self.max_accel, u))
        return u

    def _plan_horizon(self) -> List[List[float]]:
        """
        Backward Riccati recursion over `horizon` steps.
        Called EVERY tick — this is what makes it genuinely receding-horizon.
        Returns K0 (1×2 gain for current step only).
        """
        A, B, Q, R, Qf = self._A, self._B, self._Q, self._R, self._Qf
        At, Bt = self._transpose(A), self._transpose(B)
        P  = [row[:] for row in Qf]   # start from terminal cost
        K0 = [[0.0, 0.0]]

        for _ in range(self.horizon):
            AtP   = self._mm(At, P)
            AtPA  = self._mm(AtP, A)
            AtPB  = self._mm(AtP, B)
            BtPA  = self._mm(Bt, self._mm(P, A))
            BtPB  = self._mm(Bt, self._mm(P, B))
            denom = R[0][0] + BtPB[0][0]
            if abs(denom) < 1e-12:
                K0 = [[0.0, 0.0]]
                correction = [[0.0, 0.0], [0.0, 0.0]]
            else:
                K0 = [[BtPA[0][j]/denom for j in range(2)]]
                correction = [[AtPB[i][0]*BtPA[0][j]/denom for j in range(2)] for i in range(2)]
            P = self._madd(Q, self._msub(AtPA, correction))

        return K0   # gain for the FIRST (current) step

    def compute(self, setpoint: float, measurement: float, ff_vel: float = 0.0) -> float:
        """
        Receding-horizon MPC with (a) reference-velocity preview and (b) a disturbance estimator.

        ff_vel : planned reference velocity at this instant (m/s). MPC plans over the error relative to the
                 moving reference (v_rel = v - ff_vel), which is the standard way MPC uses a known trajectory.
        The integral-style disturbance estimate removes the steady wind-induced offset (same job as LQR's
        integral state). Velocity is low-pass filtered because it is differenced from noisy positions.
        """
        pos_error = setpoint - measurement

        if self._prev_measurement is None:
            raw_v = 0.0
        else:
            raw_v = (measurement - self._prev_measurement) / self.dt
        self._vel_actual += self.vel_alpha * (raw_v - self._vel_actual)     # filtered velocity
        self._prev_measurement = measurement

        if self.model == "lag":
            u = self._compute_lag(pos_error, ff_vel)
            self._c_hist.append(self._vel_rel + u * self.dt); self._c_hist = self._c_hist[-max(1, self.delay_ticks):]
            self._dist += self.k_int * (-pos_error) * self.dt
            self._dist  = max(-self.int_limit, min(self.int_limit, self._dist))
            self._vel_rel  = max(self.out_min, min(self.out_max, self._vel_rel + u * self.dt))
            self._vel_output = max(self.out_min, min(self.out_max, ff_vel + self._vel_rel - self._dist))
            return self._vel_output

        K0 = None if self.constrained else self._plan_horizon()              # replanned every tick

        # Latency compensation (predictive control): roll the state forward through the known delay using the
        # commands already in flight, so the plan acts on where the vehicle WILL be when the command lands.
        d, vr, e = self.delay_ticks, self._vel_actual, pos_error
        if d > 0:
            for uu in self._u_hist[-d:]:
                e  -= vr * self.dt
                vr += uu * self.dt
        x  = [e, vr]      # differenced error is already relative to the moving reference
        if self.constrained:
            u = self._solve_qp(x)                      # constraint-aware plan: knows the acceleration limit ahead of time
        else:
            u = -self._mv(K0, x)[0]
        u  = max(-self.max_accel, min(self.max_accel, u))
        self._u_hist.append(u); self._u_hist = self._u_hist[-max(1, self.delay_ticks):]

        # disturbance estimate (anti-windup: only integrate while the command is not saturated)
        self._dist += self.k_int * (-pos_error) * self.dt
        self._dist  = max(-self.int_limit, min(self.int_limit, self._dist))

        self._vel_rel  += u * self.dt
        self._vel_rel   = max(self.out_min, min(self.out_max, self._vel_rel))
        self._vel_output = max(self.out_min, min(self.out_max, ff_vel + self._vel_rel - self._dist))
        return self._vel_output

    def reset(self) -> None:
        self._prev_measurement = None
        self._vel_actual       = 0.0
        self._vel_output       = 0.0
        self._vel_rel          = 0.0
        self._dist             = 0.0
        self._c_hist, self._u_hist = [], []
        self._adm              = None


# ══════════════════════════════════════════════════════════════════
#  5. ADVANCED FLIGHT CONTROLLER  (pluggable classical vs modern control)
# ══════════════════════════════════════════════════════════════════

# Tuned defaults (grid search on training seeds 101-102; evaluated on other seeds). Both controllers are told the same
# 6.5 m/s^2 horizontal acceleration limit (35 deg tilt); only the MPC plans around it ahead of time.
BASE_LQR_KW = dict(q_pos=10.0, q_vel=3.0, q_int=0.5, r=2.0, max_accel=6.5)
# MPC: 3-state lag model [error, velocity, commanded velocity] solved as a constrained QP (see MPCController docstring).
BASE_MPC_KW = dict(horizon=20, q_pos=3.0, q_vel=1.0, q_terminal=7.5, r=1.0, max_accel=6.5, delay_ticks=1,
                   model="lag", tau=0.35, gain=0.94)


class AdvancedFlightController:
    """
    Advanced Flight Controller — pluggable classical vs modern control.

    Same cascaded position → velocity architecture and public interface as
    DroneFlightController in drone_agent.py (compute_velocity() / reset()),
    but the active control law can be swapped at runtime between:

        ControlMode.PID        — classical, closed-loop  (PIDController)
        ControlMode.OPEN_LOOP  — classical, open-loop     (OpenLoopController)
        ControlMode.LQR        — modern,    closed-loop   (LQRAxisController)
        ControlMode.MPC        — modern,    closed-loop   (MPCController)

    Drop-in compatible: can be used anywhere DroneFlightController is used
    today (same drone_id, cur_lat/cur_lng/cur_alt, tgt_lat/tgt_lng/tgt_alt,
    vx/vy/vz naming), without editing drone_agent.py or
    flask_server_robotics.py.
    """

    def __init__(self, drone_id: str, control_mode: ControlMode = ControlMode.PID):
        self.drone_id = drone_id
        self.control_mode = control_mode
        self._last_target: Optional[Tuple[float, float, float]] = None
        self._build_axis_controllers()

    def _build_axis_controllers(self) -> None:
        mode = self.control_mode
        if mode == ControlMode.PID:
            self.ctrl_lat = PIDController(kp=1.8, ki=0.05, kd=0.40,
                                           out_min=-MAX_SPEED, out_max=MAX_SPEED)
            self.ctrl_lng = PIDController(kp=1.8, ki=0.05, kd=0.40,
                                           out_min=-MAX_SPEED, out_max=MAX_SPEED)
            self.ctrl_alt = PIDController(kp=2.5, ki=0.10, kd=0.80,
                                           out_min=-MAX_VSPEED, out_max=MAX_VSPEED)
        elif mode == ControlMode.OPEN_LOOP:
            self.ctrl_lat = OpenLoopController(max_speed=MAX_SPEED)
            self.ctrl_lng = OpenLoopController(max_speed=MAX_SPEED)
            self.ctrl_alt = OpenLoopController(max_speed=MAX_VSPEED)
        elif mode == ControlMode.LQR:
            self.ctrl_lat = LQRAxisController(out_min=-MAX_SPEED, out_max=MAX_SPEED, **BASE_LQR_KW)
            self.ctrl_lng = LQRAxisController(out_min=-MAX_SPEED, out_max=MAX_SPEED, **BASE_LQR_KW)
            self.ctrl_alt = LQRAxisController(out_min=-MAX_VSPEED, out_max=MAX_VSPEED, **{**BASE_LQR_KW, "max_accel": 4.0})
        elif mode == ControlMode.MPC:
            self.ctrl_lat = MPCController(out_min=-MAX_SPEED, out_max=MAX_SPEED, **BASE_MPC_KW)
            self.ctrl_lng = MPCController(out_min=-MAX_SPEED, out_max=MAX_SPEED, **BASE_MPC_KW)
            self.ctrl_alt = MPCController(out_min=-MAX_VSPEED, out_max=MAX_VSPEED, **{**BASE_MPC_KW, "max_accel": 4.0})
        else:
            raise ValueError(f"Unknown control_mode: {mode}")

    def set_mode(self, control_mode: ControlMode) -> None:
        """Switch control paradigm at runtime and rebuild fresh axis controllers."""
        self.control_mode = control_mode
        self._last_target = None
        self._build_axis_controllers()

    def compute_velocity(self,
                          cur_lat: float, cur_lng: float, cur_alt: float,
                          tgt_lat: float, tgt_lng: float, tgt_alt: float
                          ) -> Tuple[float, float, float]:
        """
        Convert position error to velocity commands.

        Returns:
            (vx, vy, vz) in m/s — North, East, Up
        """
        dlat_m = (tgt_lat - cur_lat) * EARTH_R * DEG2RAD
        dlng_m = ((tgt_lng - cur_lng) * EARTH_R
                  * math.cos(cur_lat * DEG2RAD) * DEG2RAD)
        dalt_m = tgt_alt - cur_alt

        if self.control_mode == ControlMode.OPEN_LOOP:
            # Open loop only re-plans when the TARGET itself changes — it
            # never reacts to the drone's actual measured progress in between.
            target_now = (round(tgt_lat, 7), round(tgt_lng, 7), round(tgt_alt, 1))
            if target_now != self._last_target:
                self._last_target = target_now
                self.ctrl_lat.plan(0.0, -dlat_m)
                self.ctrl_lng.plan(0.0, -dlng_m)
                self.ctrl_alt.plan(0.0, -dalt_m)
            vx = self.ctrl_lat.compute(DT)
            vy = self.ctrl_lng.compute(DT)
            vz = self.ctrl_alt.compute(DT)
        else:
            vx = self.ctrl_lat.compute(0.0, -dlat_m)
            vy = self.ctrl_lng.compute(0.0, -dlng_m)
            vz = self.ctrl_alt.compute(0.0, -dalt_m)

        # Scale horizontal speed to not exceed MAX_SPEED
        speed_2d = math.sqrt(vx ** 2 + vy ** 2)
        if speed_2d > MAX_SPEED:
            scale = MAX_SPEED / speed_2d
            vx *= scale
            vy *= scale

        return vx, vy, vz

    def reset(self) -> None:
        self.ctrl_lat.reset()
        self.ctrl_lng.reset()
        self.ctrl_alt.reset()
        self._last_target = None




# ══════════════════════════════════════════════════════════════════
#  6. SIX-DRONE CONTROLLER BENCHMARK  (Open-Loop / PID / MPC / LQR)
#  Same plant, references, wind and initial conditions for every
#  controller; all metrics computed from the simulated time series.
#  Used by /api/v6/controllers/run in flask_server_robotics.py.
#  Open-loop here replays the reference velocity blindly (feed-forward).
# ══════════════════════════════════════════════════════════════════

import numpy as np

                                   # 0.1 s, same tick as the main sim
BENCH_CONTROLLERS = [("OPEN_LOOP", "Open-Loop"), ("PID", "PID"), ("MPC", "MPC"), ("LQR", "LQR")]
BENCH_LIM_V = np.array([MAX_SPEED, MAX_SPEED, MAX_VSPEED])   # m/s per axis
BENCH_LIM_A = np.array([8.0, 8.0, 4.0])             # plant acceleration limits (m/s^2)
BENCH_TAU, BENCH_GAIN = 0.35, 0.90                        # plant lag + unmodelled 10% thrust loss
BENCH_BAND = 0.75                                   # settling band (m): error must stay inside this after settling time

# six drones: (start, goal) in local ENU metres
BENCH_MISSIONS = [([0, 0, 0], [150, 90, 25]), ([20, -30, 0], [-120, 80, 22]),
            ([-40, 10, 0], [60, -140, 28]), ([0, 50, 0], [170, 20, 20]),
            ([30, 30, 0], [-90, -110, 30]), ([-10, -60, 0], [100, 130, 24])]


def _bench_wind(seed, n):
    """Deterministic per-drone wind (m/s, ENU): steady bias + slow gusts + Ornstein-Uhlenbeck turbulence."""
    r = np.random.default_rng(seed)
    t = np.arange(n) * DT
    ax_ = np.array([1, 1, 0.3])
    bias = r.uniform(-1.5, 1.5, 3) * ax_
    ph = r.uniform(0, 6.28, 3)
    gust = 0.9 * np.sin(0.45 * t[:, None] + ph) * ax_
    turb = np.zeros((n, 3))
    for k in range(1, n):                              # OU turbulence: tau ~3 s, sigma ~0.7 m/s
        turb[k] = turb[k - 1] * (1 - DT / 3.0) + r.normal(0, 0.7 * math.sqrt(2 * DT / 3.0), 3) * ax_
    return bias + gust + turb


BENCH_PEAK_SPEED = 12.0   # m/s peak speed of the rest-to-rest reference
# Realism terms (identical rules for every controller; only the random draw differs per drone)
BENCH_DRAG = 0.10          # wind drag coefficient (1/s): a_wind = k (w - v)
BENCH_DELAY = 1            # control + actuation latency in ticks (0.2 s)
BENCH_ATT_TAU = 0.06      # attitude/motor lag, two cascaded first-order stages (s)
BENCH_JERK = 80.0          # m/s^3 jerk limit of the airframe
BENCH_GPS_SIGMA = np.array([0.35, 0.35, 0.5])   # position noise (m), GPS-class
BENCH_GPS_EVERY = 2        # GPS update every 2 ticks (5 Hz), sample-and-hold
BENCH_VEL_SIGMA = 0.12     # velocity estimate noise (m/s)
BENCH_TILT_MAX = math.radians(35.0)
BENCH_CORRIDOR = 2.5       # safety corridor around the reference (m)


def _bench_reference(i, n):
    """Quintic (minimum-jerk) rest-to-rest reference: s(tau)=10tau^3-15tau^4+6tau^5.
    Duration T is chosen so the peak speed (1.875*D/T) equals BENCH_PEAK_SPEED."""
    s, g = BENCH_MISSIONS[i]
    s_, g_ = np.array(s, float), np.array(g, float)
    T = 1.875 * np.linalg.norm(g_ - s_) / BENCH_PEAK_SPEED
    tau = np.clip(np.arange(n) * DT / T, 0.0, 1.0)
    sh = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    p = s_ + sh[:, None] * (g_ - s_)
    v = np.vstack([np.diff(p, axis=0) / DT, np.zeros((1, 3))])
    return p, v, np.array(s, float), T


def _bench_ref_duration(i):
    s, g = BENCH_MISSIONS[i]
    return 1.875 * np.linalg.norm(np.array(g, float) - np.array(s, float)) / BENCH_PEAK_SPEED


def _bench_simulate(mode, i, n, wind, seed=7, delay=None, kw=None):
    """Closed-loop run on a non-ideal plant: noisy sampled GPS/velocity, 0.2 s latency, attitude lag,
    jerk + acceleration limits, wind drag and per-drone thrust/mass mismatch (same draw for all controllers)."""
    ref, ref_v, start, T_ref = _bench_reference(i, n)
    rng = np.random.default_rng(10_000 + 31 * seed + i)       # sensor noise: same stream for every controller
    gain_i = rng.uniform(0.90, 0.98)                           # unmodelled thrust loss for this airframe
    pos, vel = start.copy(), np.zeros(3)
    acc1 = np.zeros(3); acc2 = np.zeros(3); acc_prev = np.zeros(3)
    delay = BENCH_DELAY if delay is None else delay
    ctl = None if mode == "OPEN_LOOP" else AdvancedFlightController(f"Drone-{i+1}", ControlMode[mode])
    axes = None if ctl is None else (ctl.ctrl_lat, ctl.ctrl_lng, ctl.ctrl_alt)
    if kw and mode in ("LQR", "MPC"):                           # tuned parameter overrides (same sensors/plant)
        C = LQRAxisController if mode == "LQR" else MPCController
        kz = dict(kw); kz["max_accel"] = min(kz.get("max_accel", 4.0), 4.0)
        axes = (C(out_min=-MAX_SPEED, out_max=MAX_SPEED, **kw), C(out_min=-MAX_SPEED, out_max=MAX_SPEED, **kw),
                C(out_min=-MAX_VSPEED, out_max=MAX_VSPEED, **kz))
    err, effort, ctime, viol = np.zeros(n), np.zeros(n), np.zeros(n), 0
    P = np.zeros((n, 3)); vprev = np.zeros(3)
    pos_meas, bias, pos_est = start.copy(), np.zeros(3), start.copy()
    vm = np.zeros(3)                                           # open-loop internal (ideal) model velocity
    queue = [np.zeros(3)] * delay
    in_viol, last_viol_k, airborne = False, -10**9, False
    for k in range(n):
        e_true = ref[k] - pos
        err[k] = np.linalg.norm(e_true); P[k] = pos
        # --- sensing: GPS bias (slow OU) + white noise, sampled at 5 Hz and held; noisy velocity ---
        bias = bias * (1 - DT / 20.0) + rng.normal(0, 0.12 * math.sqrt(2 * DT / 20.0), 3) * BENCH_GPS_SIGMA / 0.35
        if k % BENCH_GPS_EVERY == 0:
            pos_meas = pos + bias + rng.normal(0, 1, 3) * BENCH_GPS_SIGMA
        vel_meas = vel + rng.normal(0, BENCH_VEL_SIGMA, 3)
        # --- estimator (complementary / alpha filter): propagate with measured velocity, correct with GPS ---
        pos_est = pos_est + vel_meas * DT
        if k % BENCH_GPS_EVERY == 0:
            pos_est = pos_est + 0.2 * (pos_meas - pos_est)
        e = ref[k] - pos_est
        t0 = time.perf_counter()
        if mode == "OPEN_LOOP":
            vc = ref_v[k].copy()                               # pre-planned replay, no feedback at all
            acc_cmd = (vc - vm) / BENCH_TAU + BENCH_DRAG * vm   # feed-forward incl. nominal drag; wind and thrust loss unknown
            vm = vm + np.clip((vc - vm) / BENCH_TAU, -BENCH_LIM_A, BENCH_LIM_A) * DT
        else:
            vc = np.zeros(3)
            for a, c in enumerate(axes):
                if hasattr(c, "_prev_time"):                   # PID: fixed simulation dt (deterministic)
                    vc[a] = c.compute(0.0, -e[a], dt=DT)
                else:
                    vc[a] = c.compute(0.0, -e[a], ff_vel=ref_v[k][a]) if isinstance(c, MPCController) else c.compute(0.0, -e[a])
            sp = math.hypot(vc[0], vc[1])
            if sp > MAX_SPEED:
                vc[:2] *= MAX_SPEED / sp
            acc_cmd = (vc - vel_meas) / BENCH_TAU              # inner velocity loop on the noisy estimate
        ctime[k] = (time.perf_counter() - t0) * 1e3
        queue.append(acc_cmd); acc_cmd = queue.pop(0)          # latency
        a_lim = np.clip(acc_cmd, -BENCH_LIM_A, BENCH_LIM_A)
        a_lim = acc_prev + np.clip(a_lim - acc_prev, -BENCH_JERK * DT, BENCH_JERK * DT)   # jerk limit
        acc_prev = a_lim
        al = 1.0 - math.exp(-DT / BENCH_ATT_TAU)                 # exact discretisation (tau < DT, Euler would over-step)
        acc1 += (a_lim - acc1) * al                              # motor / attitude lag (2 stages)
        acc2 += (acc1 - acc2) * al
        a_true = gain_i * acc2 + BENCH_DRAG * (wind[k] - vel)  # thrust loss + wind drag
        vel = vel + a_true * DT
        pos = pos + vel * DT
        if pos[2] < 0.0:                                       # ground contact
            pos[2], vel[2] = 0.0, max(vel[2], 0.0)
        airborne = airborne or pos[2] > 1.0                    # ground contact only counts after takeoff
        # constraint violations on the TRUE state: tilt, speed, corridor, ground
        tilt = math.atan2(math.hypot(a_true[0], a_true[1]), G_ACCEL_BENCH)
        if (tilt > BENCH_TILT_MAX or np.hypot(vel[0], vel[1]) > 1.1 * MAX_SPEED or abs(vel[2]) > 1.2 * MAX_VSPEED
                or err[k] > BENCH_CORRIDOR or (airborne and pos[2] <= 0.0)):
            if not in_viol and k - last_viol_k > int(5.0 / DT):    # count distinct EPISODES (new breach after >=5 s clear), not ticks
                viol += 1
            in_viol, last_viol_k = True, k
        else:
            in_viol = False
        effort[k] = np.linalg.norm(vc - vprev) / DT; vprev = vc
    # settling: measured after the reference has finished (final approach), error must stay inside the band
    k_ref = min(int(T_ref / DT), n - 1)
    out_of = np.where(err[k_ref:] > BENCH_BAND)[0]
    settle = 0.0 if len(out_of) == 0 else (None if out_of[-1] == n - 1 - k_ref else (out_of[-1] + 1) * DT)
    return dict(drone=f"Drone {i+1}", err=err, ref=ref, pos=P,
                rmse=float(np.sqrt((err**2).mean())), mae=float(err.mean()),
                max_err=float(err.max()), settling_s=settle,
                effort=float(effort.mean()), comp_ms=float(ctime.mean()),
                violations=int(viol))


G_ACCEL_BENCH = 9.81


def run_benchmark(duration=40.0, seed=7, wind_scale=1.0):
    duration = min(max(float(duration), 20.0), 120.0)      # sane bounds for an interactive request
    duration = max(duration, max(_bench_ref_duration(i) for i in range(6)) + 15.0)   # leave time to settle after the reference ends
    wind_scale = min(max(float(wind_scale), 0.0), 5.0)
    seed = int(seed)
    n = int(round(duration / DT)) + 1
    winds = [_bench_wind(seed + i, n) * wind_scale for i in range(6)]
    res = {}
    for mode, label in BENCH_CONTROLLERS:
        res[mode] = dict(label=label, drones=[_bench_simulate(mode, i, n, winds[i], seed) for i in range(6)])
    t = (np.arange(n) * DT).round(2).tolist()
    summ = _bench_summary(res, duration)
    rank = _bench_rank(summ)
    validation = {m: dict(traces=len(r["drones"]), ok=len(r["drones"]) == 6 and all(len(d["err"]) == n for d in r["drones"]))
                  for m, r in res.items()}
    conditions = dict(dt=DT, steps=n, seed=seed, wind_scale=wind_scale, settle_band_m=BENCH_BAND,
                      plant=dict(tau_s=BENCH_TAU, gain=BENCH_GAIN, max_speed=MAX_SPEED, max_vspeed=MAX_VSPEED,
                                 max_accel=BENCH_LIM_A.tolist()),
                      missions=[dict(drone=f"Drone {i+1}", start=m[0], goal=m[1]) for i, m in enumerate(BENCH_MISSIONS)])
    return dict(t=t, validation=validation, conditions=conditions, duration=duration, dt=DT, seed=seed, wind_scale=wind_scale,
                graphs={m: dict(label=r["label"],
                                traces=[dict(name=d["drone"], err=d["err"].round(4).tolist()) for d in r["drones"]])
                        for m, r in res.items()},
                detail=_bench_detail(res), summary=summ, **rank)


def _bench_detail(res):
    return [dict(controller=r["label"], drone=d["drone"], rmse=round(d["rmse"], 3), mae=round(d["mae"], 3),
                 max_err=round(d["max_err"], 3),
                 settling_s=None if d["settling_s"] is None else round(d["settling_s"], 1),
                 effort=round(d["effort"], 2), comp_ms=round(d["comp_ms"], 4), violations=d["violations"])
            for r in res.values() for d in r["drones"]]


def _bench_summary(res, dur):
    rows = []
    for m, r in res.items():
        D = r["drones"]; st = [d["settling_s"] for d in D if d["settling_s"] is not None]
        rows.append(dict(key=m, controller=r["label"],
                         mean_rmse=float(np.mean([d["rmse"] for d in D])),
                         mean_mae=float(np.mean([d["mae"] for d in D])),
                         worst_rmse=float(max(d["rmse"] for d in D)),
                         mean_settling=float(np.mean(st)) if st else None,
                         settled=f"{len(st)}/6",
                         mean_settling_eff=float(np.mean([d["settling_s"] if d["settling_s"] is not None else dur
                                                          for d in D])),
                         mean_comp_ms=float(np.mean([d["comp_ms"] for d in D])),
                         mean_effort=float(np.mean([d["effort"] for d in D])),
                         violations=int(sum(d["violations"] for d in D))))
    return rows


BENCH_WEIGHTS = dict(mean_rmse=0.45, mean_settling_eff=0.20, mean_effort=0.10, mean_comp_ms=0.10, violations=0.15)


# Optional, OFF by default: min-max on raw RMSE is dominated by open-loop (24 m), so good controllers barely differ.
# True = normalise RMSE on a log scale (ratios matter, not metres). It changes the ranking criterion, so label it if used.
BENCH_LOG_RMSE_SCORE = False


def _bench_rank(rows):
    """Composite score = weighted min-max normalised cost (lower is better). Unsettled drones count as the full duration (mean_settling_eff)."""
    cost = {r["key"]: {} for r in rows}
    for k in BENCH_WEIGHTS:
        v = {r["key"]: (r[k] if r[k] is not None else 1e3) for r in rows}
        if BENCH_LOG_RMSE_SCORE and k == "mean_rmse":
            v = {kk: math.log(max(x, 1e-3)) for kk, x in v.items()}
        if k == "mean_comp_ms":     # judged against the real-time budget (one tick = DT), not min-max: 0.09 ms vs 0.01 ms of a 100 ms tick is not a 10-point difference
            for key in v:
                cost[key][k] = min(1.0, v[key] / (DT * 1000.0))
            continue
        lo, hi = min(v.values()), max(v.values())
        for key in v:
            cost[key][k] = 0.0 if hi == lo else (v[key] - lo) / (hi - lo)
    for r in rows:
        r["score"] = round(100 * (1 - sum(BENCH_WEIGHTS[k] * cost[r["key"]][k] for k in BENCH_WEIGHTS)), 1)
    rows.sort(key=lambda r: -r["score"])
    best = rows[0]
    rmse_best = min(rows, key=lambda r: r["mean_rmse"])
    imp = []
    for r in rows[1:]:
        p = (r["mean_rmse"] - best["mean_rmse"]) / r["mean_rmse"] * 100 if r["mean_rmse"] else 0
        imp.append(dict(vs=r["controller"], pct=round(p, 1), claim=p > 0))
    # MPC-specific block (spec formula): Improvement(%) = (Other - MPC) / Other * 100.
    # A claim is only made when the calculated value is positive.
    mpc = next((r for r in rows if r["key"] == "MPC"), None)
    mpc_vs = []
    if mpc:
        for r in rows:
            if r["key"] == "MPC":
                continue
            p = (r["mean_rmse"] - mpc["mean_rmse"]) / r["mean_rmse"] * 100 if r["mean_rmse"] else 0.0
            mpc_vs.append(dict(vs=r["controller"], pct=round(p, 1), improved=p > 0,
                               text=(f"MPC reduced mean RMSE by {p:.1f}% vs {r['controller']}" if p > 0 else
                                     f"No improvement claim: {r['controller']} has lower mean RMSE than MPC")))
    lowest = best["key"] == rmse_best["key"]
    txt = (f"{best['controller']} achieved the lowest mean tracking RMSE ({best['mean_rmse']:.3f} m) and the best overall score."
           if lowest else
           f"{best['controller']} ranked first overall (score {best['score']}), although {rmse_best['controller']} had the lowest mean RMSE "
           f"({rmse_best['mean_rmse']:.3f} m vs {best['mean_rmse']:.3f} m).")
    return dict(best=best["controller"], best_text=txt, mpc_vs=mpc_vs, rmse_best=rmse_best["controller"], improvements=imp,
                criteria=dict(weights=BENCH_WEIGHTS, note="Score 0-100 = 100 x (1 - weighted min-max normalised cost); lower RMSE, settling time, effort, compute time and violations are better. Compute time is scored as a fraction of the 100 ms control tick (real-time budget); the other criteria are min-max normalised. A drone that never settles counts as the full simulation duration."))


def export_benchmark_png(path="controller_graphs.png", result=None):
    """Save the 2x2 six-drone graphs as PNG (needs matplotlib) and validate 6 traces each."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r = result or run_benchmark()
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    for a, k in zip(ax.flat, ["PID", "MPC", "OPEN_LOOP", "LQR"]):
        g = r["graphs"][k]
        assert len(g["traces"]) == 6, f"{k}: expected 6 traces"
        for tr in g["traces"]:
            a.plot(r["t"], tr["err"], label=tr["name"], lw=1.4)
        a.set_title(f"{g['label']} - 6 drones"); a.set_ylabel("Tracking error (m)")
        a.set_xlabel("Time (s)"); a.grid(alpha=.3); a.legend(ncol=3, fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130)
    return r


# ══════════════════════════════════════════════════════════════════
#  SELF-TEST / COMPARISON BENCHMARK
# ══════════════════════════════════════════════════════════════════

def _step_response_closed_loop(controller, step_m: float, ticks: int,
                                dt: float = DT, disturbance_ms: float = 0.0) -> List[dict]:
    """
    Simulate a 1-D step response. Returns list of dicts with position, velocity,
    and control effort per tick — richer than the old float-only trace.
    """
    position   = 0.0
    velocity   = 0.0
    trace      = []
    is_pid     = hasattr(controller, "_prev_time")
    is_lqr     = hasattr(controller, "_integral")    # LQR has integral state
    is_mpc     = hasattr(controller, "_plan_horizon") # MPC replans every tick

    for i in range(ticks):
        if is_pid:
            controller._prev_time = time.time() - dt
        measurement = -(step_m - position)
        v = controller.compute(0.0, measurement)
        accel = (v - velocity) / dt if i > 0 else 0.0
        velocity = v
        position += (v + disturbance_ms) * dt
        trace.append({"pos": position, "vel": velocity,
                      "effort": abs(accel), "u": v})
    return trace


def _step_response_open_loop(controller, step_m: float, ticks: int,
                              dt: float = DT, disturbance_ms: float = 0.0) -> List[dict]:
    position = 0.0
    velocity = 0.0
    controller.plan(0.0, -(step_m - position))
    trace = []
    for i in range(ticks):
        v = controller.compute(dt)
        accel = (v - velocity) / dt if i > 0 else 0.0
        velocity = v
        position += (v + disturbance_ms) * dt
        trace.append({"pos": position, "vel": velocity,
                      "effort": abs(accel), "u": v})
    return trace


def _metrics(trace: List[dict], target: float) -> dict:
    """
    Compute multi-dimensional performance metrics.
    Fix 6: adds energy (∫u²dt) and control effort alongside time metrics.
    """
    positions = [t["pos"] for t in trace]
    efforts   = [t["effort"] for t in trace]

    peak = max(positions, key=abs) if positions else 0.0
    overshoot_pct = max(0.0, (peak - target) / target * 100) if target else 0.0

    rise_time = settle_time = None
    last_settle = None
    for i, p in enumerate(positions):
        if rise_time is None and p >= 0.9 * target:
            rise_time = round(i * DT, 2)
        in_band = abs(p - target) <= 0.02 * abs(target)
        if in_band:
            if last_settle is None:
                last_settle = i
        else:
            last_settle = None
    if last_settle is not None:
        settle_time = round(last_settle * DT, 2)

    # Control effort: integral of u² over time (∫u²dt) — LQR minimises this
    energy_lqr_cost = round(sum(t["u"]**2 for t in trace) * DT, 2)
    # Mean absolute acceleration (smoothness — lower = gentler flight)
    mean_effort = round(sum(efforts)/len(efforts), 3) if efforts else 0.0

    return {
        "rise_time_s":         rise_time,
        "overshoot_pct":       round(overshoot_pct, 1),
        "settling_time_s":     settle_time,
        "steady_state_err_m":  round(target - positions[-1], 3) if positions else None,
        "final_position_m":    round(positions[-1], 2) if positions else None,
        # Fix 6: new metrics
        "control_energy_J":    energy_lqr_cost,     # ∫u²dt — LQR minimises this
        "mean_accel_ms2":      mean_effort,          # smoothness
        "avg_velocity_ms":     round(sum(t["vel"] for t in trace)/len(trace), 2) if trace else 0,
    }


def print_benchmark_report(r):
    """Console tables for demo / viva (all numbers come from the simulation run)."""
    print("\n" + "=" * 100)
    print(f"SIX-DRONE CONTROLLER BENCHMARK  duration={r['duration']}s dt={r['dt']}s seed={r['seed']} wind x{r['wind_scale']}")
    print("=" * 100)
    for m, v in r["validation"].items():
        print(f"  graph {m:<10} traces={v['traces']}  {'OK' if v['ok'] else 'FAIL'}")
    h = f"{'Controller':<11}{'MeanRMSE':>9}{'MeanMAE':>9}{'WorstRMSE':>10}{'MeanSettle':>11}{'Settled':>8}{'Comp(ms)':>9}{'Effort':>8}{'Viol':>6}{'Score':>7}"
    print("\n" + h + "\n" + "-" * len(h))
    for x in r["summary"]:
        st = "n/a" if x["mean_settling"] is None else f"{x['mean_settling']:.1f}"
        print(f"{x['controller']:<11}{x['mean_rmse']:>9.3f}{x['mean_mae']:>9.3f}{x['worst_rmse']:>10.3f}{st:>11}{x['settled']:>8}"
              f"{x['mean_comp_ms']:>9.4f}{x['mean_effort']:>8.2f}{x['violations']:>6}{x['score']:>7}")
    print(f"\nBest-performing controller: {r['best']}\n  {r['best_text']}")
    for i in r["improvements"]:
        print(f"  {r['best']} vs {i['vs']}: {i['pct']:+.1f}% RMSE improvement" + ("" if i["claim"] else "  (no claim)"))
    print("  MPC-specific:", "; ".join(i["text"] for i in r["mpc_vs"]))
    print("  Criteria:", r["criteria"]["weights"], "\n  ", r["criteria"]["note"])


if __name__ == "__main__" and len(__import__("sys").argv) > 1 and __import__("sys").argv[1] == "--benchmark":
    _res = export_benchmark_png("controller_graphs.png")
    print_benchmark_report(_res)
    print("\nSaved controller_graphs.png (2x2, six drones per graph)")
    raise SystemExit(0)

if __name__ == "__main__":
    print("=" * 70)
    print("  Advanced Flight Controller — PID vs LQR vs MPC (+ Open Loop)")
    print("=" * 70)

    STEP  = 50.0   # metres — step-response target distance
    TICKS = 150    # 15 s of simulated flight at 10 Hz

    print(f"\nScenario: 1-D step response, target = {STEP} m, {TICKS} ticks @ {TICK_HZ} Hz\n")

    results = {}

    pid = PIDController(kp=1.8, ki=0.05, kd=0.40, out_min=-MAX_SPEED, out_max=MAX_SPEED)
    results["PID (closed loop)"] = _metrics(_step_response_closed_loop(pid, STEP, TICKS), STEP)

    ol = OpenLoopController(max_speed=MAX_SPEED)
    results["Open Loop (no feedback)"] = _metrics(_step_response_open_loop(ol, STEP, TICKS), STEP)

    lqr = LQRAxisController(out_min=-MAX_SPEED, out_max=MAX_SPEED)
    results["LQR (closed loop)"] = _metrics(_step_response_closed_loop(lqr, STEP, TICKS), STEP)
    mpc = MPCController(out_min=-MAX_SPEED, out_max=MAX_SPEED)
    results["MPC (closed loop)"] = _metrics(_step_response_closed_loop(mpc, STEP, TICKS), STEP)

    header = f"{'Controller':<26}{'Rise(s)':>10}{'Overshoot%':>12}{'Settle(s)':>11}{'SS Err(m)':>11}{'Energy(J)':>11}{'Smoothness':>12}"
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        print(f"{name:<26}{str(m['rise_time_s']):>10}{m['overshoot_pct']:>12}"
              f"{str(m['settling_time_s']):>11}{m['steady_state_err_m']:>11}"
              f"{m['control_energy_J']:>11}{m['mean_accel_ms2']:>12}")

    # ── Disturbance test: shows WHY closed loop beats open loop ──────
    print("\nDisturbance test: constant -0.5 m/s crosswind drift for whole flight")
    print(f"{'Controller':<26}{'Final pos (m, target=50)':>26}{'SS Error':>10}")
    print("-" * 62)

    pid2 = PIDController(kp=1.8, ki=0.05, kd=0.40, out_min=-MAX_SPEED, out_max=MAX_SPEED)
    t = _step_response_closed_loop(pid2, STEP, TICKS, disturbance_ms=-0.5)
    print(f"{'PID (closed loop)':<26}{round(t[-1]['pos'], 2):>26}{round(STEP-t[-1]['pos'],3):>10}")

    ol2 = OpenLoopController(max_speed=MAX_SPEED)
    t = _step_response_open_loop(ol2, STEP, TICKS, disturbance_ms=-0.5)
    print(f"{'Open Loop (no feedback)':<26}{round(t[-1]['pos'], 2):>26}{round(STEP-t[-1]['pos'],3):>10}  ← cannot correct drift")

    lqr2 = LQRAxisController(out_min=-MAX_SPEED, out_max=MAX_SPEED)
    t = _step_response_closed_loop(lqr2, STEP, TICKS, disturbance_ms=-0.5)
    print(f"{'LQR+integral (closed)':<26}{round(t[-1]['pos'], 2):>26}{round(STEP-t[-1]['pos'],3):>10}  ← integral kills SS error")

    mpc2 = MPCController(out_min=-MAX_SPEED, out_max=MAX_SPEED)
    t = _step_response_closed_loop(mpc2, STEP, TICKS, disturbance_ms=-0.5)
    print(f"{'MPC receding (closed)':<26}{round(t[-1]['pos'], 2):>26}{round(STEP-t[-1]['pos'],3):>10}  ← replans around disturbance")

    # ── AdvancedFlightController integration test (real GPS coords) ──
    print("\nAdvancedFlightController — 3D compute_velocity() per mode:")
    origin = (16.5062, 80.6480, 0.0)
    target = (16.5120, 80.6550, 80.0)
    for mode in ControlMode:
        afc = AdvancedFlightController(drone_id="TEST-1", control_mode=mode)
        vx, vy, vz = afc.compute_velocity(*origin, *target)
        print(f"  {mode.value:<10} → vx={vx:6.2f}  vy={vy:6.2f}  vz={vz:6.2f}  m/s")

    print("\n✅  Advanced Flight Controller self-tests passed.\n")


# ══════════════════════════════════════════════════════════════════
#  DELAYED-FLIGHT STUDY  (multi-seed, tuned LQR vs tuned constrained MPC)
# ══════════════════════════════════════════════════════════════════
DELAYED_TICKS = 3          # 0.3 s control+actuation latency (same plant, sensors, wind, references as baseline)
# Parameters chosen by grid search on TRAINING seeds (101, 102); reported results use different seeds.
DELAYED_LQR_KW = dict(q_pos=0.05, q_vel=0.05, q_int=0.5, r=6.0, max_accel=6.5)
# Lag-aware MPC re-tuned on training seeds 101-102 (tune_lag.py); reported on seeds 1-20.
DELAYED_MPC_KW = dict(horizon=20, q_pos=3.0, q_vel=1.0, q_terminal=7.5, r=1.0, max_accel=6.5,
                      delay_ticks=DELAYED_TICKS, model="lag", tau=0.35, gain=0.94)
# Previous 2-state model, kept for A/B comparison (python sens.py).
LEGACY_MPC_KW = dict(horizon=30, q_pos=1.0, q_vel=1.0, q_terminal=2.5, r=8.0, max_accel=6.5,
                     delay_ticks=DELAYED_TICKS, constrained=True, model="double")
# Both controllers are told the same 6.5 m/s^2 horizontal limit (35 deg tilt); only MPC plans around it.


def run_delayed_study(seeds=range(1, 21), duration=50.0, modes=("PID", "LQR", "MPC", "OPEN_LOOP"),
                      mpc_kw=None, lqr_kw=None, delay=DELAYED_TICKS):
    """Mean +/- std over seeds of per-seed fleet means (6 drones). Seeds change wind turbulence, thrust loss
    and sensor noise; missions, plant and sensors are identical for every controller."""
    n = int(max(duration, max(_bench_ref_duration(i) for i in range(6)) + 15.0) / DT)
    kws = {"LQR": lqr_kw or DELAYED_LQR_KW, "MPC": mpc_kw or DELAYED_MPC_KW}
    out = {}
    for mode in modes:
        per_seed = []
        for sd in seeds:
            w = _bench_wind(sd, n)
            rs = [_bench_simulate(mode, i, n, w, sd, delay, kws.get(mode)) for i in range(6)]
            st = [r["settling_s"] for r in rs]
            per_seed.append(dict(rmse=np.mean([r["rmse"] for r in rs]),
                                 settled=sum(x is not None for x in st) / 6.0,
                                 settle=np.mean([x for x in st if x is not None]) if any(x is not None for x in st) else np.nan,
                                 viol=np.sum([r["violations"] for r in rs]),
                                 comp=np.mean([r["comp_ms"] for r in rs])))
        f = lambda k: np.array([p[k] for p in per_seed], float)
        out[mode] = dict(rmse=f("rmse").mean(), rmse_std=f("rmse").std(), settled=f("settled").mean(),
                         settle=float(np.nanmean(f("settle"))) if np.isfinite(f("settle")).any() else None,
                         viol=f("viol").mean(), comp_ms=f("comp").mean(), seeds=len(per_seed),
                         wins_vs=None, per_seed_rmse=f("rmse").tolist())
    return out


def print_delayed_study(res):
    print(f"{'Controller':<11}{'RMSE (m)':>14}{'Settled':>9}{'Settle(s)':>10}{'Viol/seed':>10}{'Comp(ms)':>10}")
    for m, r in res.items():
        se = "n/a" if r["settle"] is None else f"{r['settle']:.1f}"
        print(f"{m:<11}{r['rmse']:>8.3f}±{r['rmse_std']:<5.3f}{r['settled']*100:>8.0f}%{se:>10}{r['viol']:>10.1f}{r['comp_ms']:>10.3f}")
    if "LQR" in res and "MPC" in res:
        a, b = np.array(res["LQR"]["per_seed_rmse"]), np.array(res["MPC"]["per_seed_rmse"])
        print(f"MPC better than LQR on {int((b < a).sum())}/{len(a)} seeds; mean ratio MPC/LQR = {b.mean()/a.mean():.2f}")
