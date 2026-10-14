"""
trajectory_generation.py  -  SkyLogix V6
Linear, Quintic, Minimum-Jerk, Minimum-Snap, Bezier and B-Spline trajectories
in local ENU metres (x, y, z), plus the comparison metrics.

Every generator returns a Trajectory; Trajectory.sample(dt) returns
t, pos, vel, acc, jerk, snap as (N,3) arrays.
Minimum jerk / snap are true optimal piecewise polynomials through the
waypoints (degree 5 / 7, derivatives continuous, rest at both ends).
"""
import time
from math import factorial
import numpy as np

G = 9.81
METHODS = ("linear", "quintic", "bezier", "bspline", "min_jerk", "min_snap")


def _law5(u):  # rest-to-rest 5th order timing law
    return u**3 * (10 - 15 * u + 6 * u**2)


class Trajectory:
    def __init__(self, name, T, fn, ctrl=None, knots=None, compute_ms=0.0):
        self.name, self.T, self.fn = name, float(T), fn
        self.ctrl = None if ctrl is None else np.asarray(ctrl, float)
        self.knots, self.compute_ms = knots, compute_ms

    def position(self, t):
        return self.fn(np.clip(np.atleast_1d(t), 0.0, self.T))

    def sample(self, dt=0.05):
        t = np.arange(0.0, self.T + dt, dt)
        p = self.position(t)
        d = [p]
        for _ in range(4):
            d.append(np.gradient(d[-1], dt, axis=0))
        return dict(t=t, pos=d[0], vel=d[1], acc=d[2], jerk=d[3], snap=d[4])

    def metrics(self, dt=0.05):
        s = self.sample(dt)
        sp = lambda a: np.linalg.norm(a, axis=1)
        length = float(sp(np.diff(s["pos"], axis=0)).sum())
        v, a, j = sp(s["vel"]), sp(s["acc"]), sp(s["jerk"])
        energy = float((180.0 * (1 + 0.004 * a**2 + 0.002 * v**2) * dt / 3600.0).sum())
        rms_j = float(np.sqrt((j**2).mean()))
        return dict(method=self.name, path_length_m=round(length, 2),
                    flight_time_s=round(self.T, 2), energy_wh=round(energy, 3),
                    max_velocity=round(float(v.max()), 2),
                    max_acceleration=round(float(a.max()), 2),
                    max_jerk=round(float(j.max()), 2),
                    max_snap=round(float(sp(s["snap"]).max()), 1),
                    smoothness_score=round(100.0 / (1.0 + rms_j / 5.0), 1),
                    computation_ms=round(self.compute_ms, 3))


def _timed(build):
    def wrap(*a, **k):
        t0 = time.perf_counter()
        tr = build(*a, **k)
        tr.compute_ms = (time.perf_counter() - t0) * 1e3
        return tr
    return wrap


def _wp(w):
    return np.asarray(w, float).reshape(-1, 3)


def _duration(length, speed, factor):
    return max(4.0, length / speed * factor)


@_timed
def linear(waypoints, speed=12.0):
    W = _wp(waypoints)
    seg = np.linalg.norm(np.diff(W, axis=0), axis=1)
    cum = np.r_[0, np.cumsum(seg)]
    T = max(4.0, cum[-1] / speed)

    def fn(t):
        d = t / T * cum[-1]
        return np.stack([np.interp(d, cum, W[:, k]) for k in range(3)], 1)
    return Trajectory("linear", T, fn)


@_timed
def quintic(waypoints, speed=12.0):
    """x(t)=a0+a1 t+...+a5 t^5 per axis, rest-to-rest between first and last waypoint."""
    W = _wp(waypoints)
    p0, p1 = W[0], W[-1]
    T = _duration(np.linalg.norm(p1 - p0), speed, 1.6)
    A = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 2, 0, 0, 0],
                  [1, T, T**2, T**3, T**4, T**5], [0, 1, 2*T, 3*T**2, 4*T**3, 5*T**4],
                  [0, 0, 2, 6*T, 12*T**2, 20*T**3]], float)
    coef = np.stack([np.linalg.solve(A, [p0[k], 0, 0, p1[k], 0, 0]) for k in range(3)])  # (3,6)

    def fn(t):
        return np.stack([np.polyval(coef[k][::-1], t) for k in range(3)], 1)
    return Trajectory("quintic", T, fn)


def _piecewise(W, T_seg, k):
    """Minimum-derivative-k polynomial through W. k=3 jerk (deg 5), k=4 snap (deg 7)."""
    n, N = len(W) - 1, 2 * k - 1
    # Solve in normalised time u = tau / T_seg in [0,1] (T^7 reaches ~1e17 on long legs and made the
    # linear system ill-conditioned: altitude shot to 18 km). Derivatives are rescaled by T^-m.
    row = lambda t, m, T=1.0: np.array([factorial(j) // factorial(j - m) * t**(j - m) / T**m if j >= m else 0.0
                                        for j in range(N + 1)])
    A = np.zeros((n * (N + 1), n * (N + 1)))
    B = np.zeros((n * (N + 1), 3))
    r, c = 0, lambda i: slice(i * (N + 1), (i + 1) * (N + 1))
    for i in range(n):                       # waypoint positions
        A[r, c(i)] = row(0, 0); B[r] = W[i]; r += 1
        A[r, c(i)] = row(1.0, 0); B[r] = W[i + 1]; r += 1
    for m in range(1, k):                    # rest at start and end
        A[r, c(0)] = row(0, m, T_seg[0]); r += 1
        A[r, c(n - 1)] = row(1.0, m, T_seg[-1]); r += 1
    for i in range(n - 1):                   # continuity at joints (physical derivatives)
        for m in range(1, 2 * k - 1):
            A[r, c(i)] = row(1.0, m, T_seg[i]); A[r, c(i + 1)] = -row(0, m, T_seg[i + 1]); r += 1
    coef = np.linalg.solve(A, B).reshape(n, N + 1, 3)
    edges = np.r_[0, np.cumsum(T_seg)]

    def fn(t):
        out = np.zeros((len(t), 3))
        idx = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, n - 1)
        for i in range(n):
            m = idx == i
            if m.any():
                u = (t[m] - edges[i]) / T_seg[i]
                out[m] = np.stack([np.polyval(coef[i, ::-1, a], u) for a in range(3)], 1)
        return out
    return fn, edges[-1]


def _densify(W, max_len=60.0):
    """Split long legs into <= max_len pieces. Free-velocity min-jerk/snap through a 40 m climb leg followed by
    a 1.9 km cruise leg overshoots wildly (altitude hit 18 km); similar leg lengths keep it well behaved."""
    out = [W[0]]
    for a, b in zip(W[:-1], W[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / max_len)))
        out += [a + (b - a) * (i / n) for i in range(1, n + 1)]
    return np.array(out)


def _min_der(name, k, waypoints, speed):
    W = _densify(_wp(waypoints))
    seg = np.linalg.norm(np.diff(W, axis=0), axis=1)
    T_seg = np.maximum(seg / speed * 1.6, 0.5)
    fn, T = _piecewise(W, T_seg, k)
    return Trajectory(name, T, fn)


@_timed
def min_jerk(waypoints, speed=12.0):
    return _min_der("min_jerk", 3, waypoints, speed)


@_timed
def min_snap(waypoints, speed=12.0):
    return _min_der("min_snap", 4, waypoints, speed)


@_timed
def bezier(control_points, speed=12.0):
    """Bezier of any order (quadratic = 3 pts, cubic = 4 pts, higher = more). Control points returned in .ctrl."""
    P = _wp(control_points)
    n = len(P) - 1
    from math import comb

    def curve(u):
        return sum(comb(n, i) * (1 - u)[:, None]**(n - i) * u[:, None]**i * P[i] for i in range(n + 1))
    L = np.linalg.norm(np.diff(curve(np.linspace(0, 1, 200)), axis=0), axis=1).sum()
    T = _duration(L, speed, 1.6)
    return Trajectory("bezier", T, lambda t: curve(_law5(t / T)), ctrl=P)


def _basis(i, k, u, K):
    if k == 0:
        return ((K[i] <= u) & (u < K[i + 1])) | ((u == 1.0) & (K[i + 1] == 1.0) & (K[i] < 1.0))
    a = b = 0.0
    if K[i + k] != K[i]:
        a = (u - K[i]) / (K[i + k] - K[i]) * _basis(i, k - 1, u, K)
    if K[i + k + 1] != K[i + 1]:
        b = (K[i + k + 1] - u) / (K[i + k + 1] - K[i + 1]) * _basis(i + 1, k - 1, u, K)
    return a + b


def clamped_knots(n_ctrl, degree=3):
    inner = np.linspace(0, 1, n_ctrl - degree + 1)[1:-1]
    return np.r_[[0.0] * (degree + 1), inner, [1.0] * (degree + 1)]


@_timed
def bspline(control_points, speed=12.0, degree=3, knots=None):
    """Clamped B-spline. Moving one control point only changes the curve locally (see modify_control_point)."""
    P = _wp(control_points)
    K = clamped_knots(len(P), degree) if knots is None else np.asarray(knots, float)

    def curve(u):
        return sum(_basis(i, degree, u, K)[:, None] * P[i] for i in range(len(P)))
    L = np.linalg.norm(np.diff(curve(np.linspace(0, 1, 200)), axis=0), axis=1).sum()
    T = _duration(L, speed, 1.6)
    return Trajectory("bspline", T, lambda t: curve(_law5(t / T)), ctrl=P, knots=K)


def modify_control_point(traj, index, new_xyz, speed=12.0):
    """Local path modification of a B-spline: only the span influenced by that control point changes."""
    P = traj.ctrl.copy()
    P[index] = new_xyz
    return bspline(P, speed, knots=traj.knots)


def control_polygon(start, goal, via=None, cruise_alt=40.0):
    """Climb - cruise - descend control points used by Bezier / B-spline / min-jerk / min-snap."""
    s, g = np.asarray(start, float), np.asarray(goal, float)
    mid = np.asarray(via, float) if via is not None else (s + g) / 2
    up = lambda p: np.array([p[0], p[1], max(p[2], cruise_alt)])
    return [s, up(s), up(mid), up(g), g]


def generate(method, start, goal, via=None, speed=12.0, cruise_alt=40.0):
    start, goal = np.asarray(start, float), np.asarray(goal, float)
    if method in ("linear", "quintic"):
        wp = [start, goal] if via is None else [start, via, goal]
        return {"linear": linear, "quintic": quintic}[method](wp, speed)
    cp = control_polygon(start, goal, via, cruise_alt)
    return {"bezier": bezier, "bspline": bspline, "min_jerk": min_jerk, "min_snap": min_snap}[method](cp, speed)


def compare(start, goal, via=None, speed=12.0):
    return [generate(m, start, goal, via, speed).metrics() for m in METHODS]


def to_json(traj, dt=0.1):
    s = traj.sample(dt)
    out = {k: v.round(3).tolist() for k, v in s.items()}
    out.update(method=traj.name, T=traj.T,
               control_points=None if traj.ctrl is None else traj.ctrl.tolist(),
               knots=None if traj.knots is None else list(map(float, traj.knots)))
    return out


if __name__ == "__main__":
    for row in compare([0, 0, 0], [150, 90, 0]):
        print(row)
