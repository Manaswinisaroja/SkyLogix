"""
mission_replanner.py  -  SkyLogix V6
Plugs into the existing V5 FleetManager without editing it:
  * V6 drone states (IDLE ... LANDED) mapped from FlightState
  * per-drone history recorder (x,y,z, v, a, roll/pitch/yaw, rates) for graphs
  * runtime mission queue with priority insertion and automatic next mission
  * online replanning (old path / new path) for obstacle, weather, NFZ,
    new delivery, failure and low battery
  * research counters for the analytics page
"""
import json
import math
import os
import threading
import time
from datetime import datetime
from collections import deque

import drone_agent as da
from drone_agent import AStarPlanner, FlightState, GPSCoord, Mission, haversine

MAX_HIST = 3000
STORE = os.environ.get("SKY_STORE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "skylogix_analytics.json"))
# Energy model (ESTIMATED, not measured): P = 180 W * (1 + 0.004*|a|^2 + 0.002*|v|^2), integrated with the actual
# elapsed time between samples and reported in Wh. Payload, weather and charging are not part of this estimate.
V6_STATE = {
    FlightState.IDLE: "IDLE",
    FlightState.TAKEOFF: "TAKEOFF", FlightState.CLIMB: "TAKEOFF",
    FlightState.CRUISE_PICKUP: "EN_ROUTE", FlightState.CRUISE_DELIVER: "EN_ROUTE",
    FlightState.HOVER_PICKUP: "ASSIGNED", FlightState.HOVER_DELIVER: "DELIVERING",
    FlightState.RTB: "RETURNING", FlightState.FAULT_RTH: "RETURNING",
    FlightState.LAND: "LANDED", FlightState.CHARGING: "LANDED", FlightState.FAULT_LAND: "LANDED",
}
EVENT_NAMES = {FlightState.TAKEOFF: "Takeoff", FlightState.HOVER_PICKUP: "Pickup", FlightState.HOVER_DELIVER: "Delivery",
               FlightState.RTB: "Return to base", FlightState.LAND: "Landing"}
R_E, DEG = 6_371_000.0, math.pi / 180.0
REPLAN_REASONS = ("new_delivery", "obstacle", "no_fly_zone", "weather", "drone_failure", "battery")


class V6Coordinator:
    def __init__(self, fleet_getter, hz=10):
        self._get = fleet_getter           # callable returning the current FleetManager
        self.hz = hz
        self.hist = {}                     # drone_id -> {key: deque}
        self.queues = {}                   # drone_id -> list[Mission] (pending after current)
        self.replans = []                  # event dicts, newest last
        self.paths = {}                    # drone_id -> {"old": [...], "new": [...], "reason": str}
        self.delivered = {}                # mission_id -> record
        self.counters = dict(replans=0, obstacle_events=0, deliveries=0, replan_requests=0, replans_skipped=0)
        self._seen_delivered, self._replan_seen = set(), {}
        self._lock, self._run = threading.RLock(), False
        self.t0 = time.time()
        self.method = "min_snap"           # trajectory algorithm applied to new missions
        self.cm, self.life, self.hist_deliv = {}, {}, {}   # level 1 / level 3 / level 2 analytics
        self.tilt, self.failed_missions, self._dc = {}, 0, {}
        self.events, self._last_state, self._prev, self._aborted = {}, {}, {}, set()
        self._load()

    # ---------------- lifecycle ----------------
    def start(self):
        if self._run:
            return
        self._run = True
        threading.Thread(target=self._loop, daemon=True, name="v6-coordinator").start()

    def stop(self):
        self._run = False

    def _loop(self):
        while self._run:
            t = time.time()
            try:
                with self._lock:
                    self._record()
                    self._advance_queues()
            except Exception as exc:                  # never kill the thread
                da.log.error("V6 coordinator error: %s", exc, exc_info=True)
            time.sleep(max(0.0, 1.0 / self.hz - (time.time() - t)))

    # ---------------- history ----------------
    def _record(self):
        fleet = self._get()
        if not fleet:
            return
        now = round(time.time() - self.t0, 2)
        for did, d in list(fleet.drones.items()):
            dyn = getattr(d, "dynamics_6dof", None)
            if dyn is None:
                continue
            h = self.hist.setdefault(did, {})
            dx, dy, dz = self._desired(d)
            row = dict(t=now, **dyn.to_dict(), battery=round(d.battery, 1), state=self.v6_state(d),
                       des_x_m=round(dx, 2), des_y_m=round(dy, 2), des_z_m=round(dz, 2))   # target position (local ENU)
            if d.state != self._last_state.get(did):
                self._last_state[did] = d.state
                if d.state in EVENT_NAMES:
                    self.events.setdefault(did, []).append(dict(t=now, event=EVENT_NAMES[d.state],
                                                                mission_id=d.mission.id if d.mission else None))
                    del self.events[did][:-200]
            for k, v in row.items():
                h.setdefault(k, deque(maxlen=MAX_HIST)).append(v)
            self._metrics(did, d, row)
            am = d.mission
            if am and am.status.value == "aborted" and am.id not in self._aborted:    # V5 fault/battery abort
                self._aborted.add(am.id)
                self.failed_missions += 1
                self._life(did)["failed_missions"] += 1
                if self.cm.get(did) and self.cm[did]["mission_id"] == am.id:
                    self.cm[did]["active"], self.cm[did]["status"] = False, "ABORTED"
                self.replans.append(dict(t=now, drone=did, reason="mission_aborted", wall=time.time()))
                self._save()
                self.add_mission(am.pickup, am.delivery, am.pickup_name, am.deliver_name, am.payload_kg, am.priority,
                                 method=getattr(am, "traj_method", None))        # never lose a mission
            dc = getattr(d, "deliveries_completed", 0)      # V5 clears d.mission on delivery, so count instead
            if dc > self._dc.get(did, 0):
                self._dc[did] = dc
                cm = self.cm.get(did)
                self.counters["deliveries"] += 1
                if cm and cm["active"]:
                    self.delivered[cm["mission_id"]] = dict(drone=did, elapsed_s=round(cm["flight_time"], 1),
                                                            dist_km=cm["distance_travelled"] / 1000,
                                                            energy_wh=cm["energy_consumed"], at=now)
                    self._finalize(did, cm["mission_id"])
            rc = getattr(d, "replan_count", 0)         # V5 internal obstacle replans
            if rc > self._replan_seen.get(did, 0):
                self._replan_seen[did] = rc
                self.counters["replans"] += 1
                self.counters["obstacle_events"] += 1
                self.replans.append(dict(t=now, drone=did, reason="obstacle", auto=True))
                self._bump(did)

    def v6_state(self, d):
        if time.time() - self._replan_seen_time(d.drone_id) < 3.0:
            return "REPLANNING"
        return V6_STATE.get(d.state, d.state.value)

    def _replan_seen_time(self, did):
        for e in reversed(self.replans):
            if e["drone"] == did:
                return e["wall"] if "wall" in e else 0.0
        return 0.0

    def history(self, did, since=0.0, keys=None):
        h = self.hist.get(did, {})
        n = len(h.get("t", ()))
        idx = [i for i, t in enumerate(h.get("t", ())) if t > since]
        keys = keys or list(h.keys())
        return {k: [h[k][i] for i in idx] for k in keys if k in h} if n else {}

    # ---------------- mission queue ----------------
    def add_mission(self, pickup, deliver, pickup_name="Pickup", deliver_name="Destination",
                    payload_kg=1.0, priority="standard", drone_id=None, method=None):
        """Insert at runtime. urgent/express go first; an urgent mission preempts the current leg."""
        fleet = self._get()
        m = Mission(id=int(time.time() * 1000) % 10**9, pickup=pickup, delivery=deliver,
                    pickup_name=pickup_name, deliver_name=deliver_name,
                    payload_kg=payload_kg, priority=priority)
        m.traj_method = method
        with self._lock:
            ok = [d for d in fleet.drones.values() if d.state != FlightState.FAULT_LAND and d.battery > 20]
            if not ok:
                return None
            d = fleet.drones[drone_id] if drone_id else min(ok, key=self._load)
            q = self.queues.setdefault(d.drone_id, [])
            q.append(m)
            q.sort(key=lambda x: x.priority_rank)         # stable: FIFO inside a priority class
            if priority == "urgent" and d.mission and d.state in (FlightState.CRUISE_PICKUP, FlightState.CRUISE_DELIVER):
                self._requeue_current(d)
                self.request_replan(d.drone_id, "new_delivery")
        return m

    def _load(self, d):
        return len(self.queues.get(d.drone_id, [])) + (1 if d.mission and d.state != FlightState.IDLE else 0)

    def _requeue_current(self, d):
        if d.mission:
            d.mission.status = da.MissionStatus.PENDING
            self.queues.setdefault(d.drone_id, []).insert(1, d.mission)

    def _advance_queues(self):
        """A drone never stays idle while it has work: IDLE/RTB/LAND drones fetch the next mission."""
        fleet = self._get()
        for did, d in list(fleet.drones.items()):
            q = self.queues.get(did)
            if not q or d.battery < 20 or d.state not in (FlightState.IDLE, FlightState.RTB, FlightState.LAND, FlightState.CHARGING):
                continue
            if d.state == FlightState.CHARGING and d.battery < 60:
                continue
            m = q.pop(0)
            start = GPSCoord(d.lat, d.lng, 0.0)           # current destination becomes next start
            planner = AStarPlanner(list(getattr(d, "_fleet_nfz", [])))
            path = planner.plan(start.lat, start.lng, m.pickup.lat, m.pickup.lng)
            tail = planner.plan(m.pickup.lat, m.pickup.lng, m.delivery.lat, m.delivery.lng)
            d.assign_mission(m, path + tail[1:], 0)
            m.dist_km = sum(haversine(*a, *b) for a, b in zip(path + tail[1:], (path + tail[1:])[1:])) / 1000

    # ---------------- online replanning ----------------
    def request_replan(self, did, reason):
        """Old path (red) -> A* from current position -> new path (green); mission continues."""
        fleet = self._get()
        d = fleet.drones[did]
        if not d.path or d.state not in (FlightState.CRUISE_PICKUP, FlightState.CRUISE_DELIVER, FlightState.RTB):
            return False
        self.counters["replan_requests"] += 1
        old = [list(p) for p in d.path[d.path_idx:]]
        obs = list(getattr(d, "_fleet_nfz", [])) + list(getattr(d, "_live_dynamic_obs", []))
        dest = d.path[-1]
        sig = (round(dest[0], 5), round(dest[1], 5), len(obs), round(self._get().weather.wind_speed_kmh))
        if getattr(d, "_rp_sig", None) == sig and time.time() - getattr(d, "_rp_t", 0.0) < 15.0:
            self.counters["replans_skipped"] += 1          # identical condition: do not regenerate the same path
            return False
        d._rp_sig, d._rp_t = sig, time.time()
        new = AStarPlanner(obs).plan(d.lat, d.lng, dest[0], dest[1])
        if len(new) < 2:
            return False
        d.path, d.path_idx, d.path_frac = new, 0, 0.0
        d._path_obs_key = da.obstacle_key(obs)         # tell V5 this path already accounts for the obstacles
        d.replan_count = getattr(d, "replan_count", 0) + 1
        self._replan_seen[did] = d.replan_count            # avoid double counting
        self.paths[did] = dict(old=old, new=[list(p) for p in new], reason=reason, t=time.time() - self.t0)
        self.replans.append(dict(t=round(time.time() - self.t0, 2), drone=did, reason=reason, wall=time.time()))
        self.counters["replans"] += 1
        self._bump(did)
        if reason in ("obstacle", "no_fly_zone"):
            self.counters["obstacle_events"] += 1
        return True

    def replan_fleet(self, reason):
        fleet = self._get()
        return [did for did in fleet.drones if self.request_replan(did, reason)]

    def fail_drone(self, did):
        """Drone failure: force land, hand all its work to the rest of the fleet, replan neighbours."""
        fleet = self._get()
        d = fleet.drones[did]
        work = ([d.mission] if d.mission and d.mission.status.value == "inflight" else []) + self.queues.pop(did, [])
        d.state = FlightState.FAULT_LAND
        nf = sum(1 for m in work if m is d.mission)
        self.failed_missions += nf
        self._life(did)["failed_missions"] += nf
        self._save()
        if self.cm.get(did):
            self.cm[did]["active"] = False
        for m in work:
            self.add_mission(m.pickup, m.delivery, m.pickup_name, m.deliver_name, m.payload_kg, m.priority)
        self.replans.append(dict(t=round(time.time() - self.t0, 2), drone=did, reason="drone_failure", wall=time.time()))
        self.counters["replans"] += 1
        return len(work)

    def check_battery(self, threshold=25.0):
        fleet, hit = self._get(), []
        for did, d in fleet.drones.items():
            if d.battery < threshold and d.state in (FlightState.CRUISE_PICKUP, FlightState.CRUISE_DELIVER):
                for m in self.queues.pop(did, []):
                    self.add_mission(m.pickup, m.delivery, m.pickup_name, m.deliver_name, m.payload_kg, m.priority)
                self.replans.append(dict(t=round(time.time() - self.t0, 2), drone=did, reason="battery", wall=time.time()))
                self.counters["replans"] += 1
                self._bump(did)
                hit.append(did)
        return hit

    # ---------------- analytics: level 1 current / level 2 history / level 3 lifetime ----------------
    def _life(self, did):
        return self.life.setdefault(did, dict(deliveries_completed=0, total_distance=0.0, total_flight_time=0.0,
                                              total_energy=0.0, total_replans=0, average_delivery_time=None, failed_missions=0,
                                              max_velocity_ever=0.0, max_acceleration_ever=0.0))

    def _bump(self, did):
        cm = self.cm.get(did)
        if cm and cm["active"]:
            cm["replans"] += 1

    def _desired(self, d):
        """Active target in the drone's local frame: pickup -> delivery -> home, at the commanded altitude."""
        m, st = d.mission, d.state
        if m and st in (FlightState.TAKEOFF, FlightState.CLIMB, FlightState.CRUISE_PICKUP, FlightState.HOVER_PICKUP):
            p = m.pickup
        elif m and st in (FlightState.CRUISE_DELIVER, FlightState.HOVER_DELIVER):
            p = m.delivery
        else:
            p = d.home
        h = d.home
        return ((p.lat - h.lat) * R_E * DEG, (p.lng - h.lng) * R_E * math.cos(p.lat * DEG) * DEG,
                getattr(d, "target_alt", d.alt))

    def _metrics(self, did, d, row):
        m, cm = d.mission, self.cm.get(did)
        if m and m.status.value not in ("delivered", "failed") and (cm is None or cm["mission_id"] != m.id):
            cm = self.cm[did] = dict(mission_id=m.id, start=m.pickup_name, dest=m.deliver_name, distance_travelled=0.0,
                                     distance_remaining=0.0, flight_time=0.0, avg_velocity=0.0, max_velocity=0.0,
                                     avg_acceleration=0.0, max_acceleration=0.0, energy_consumed=0.0,
                                     trajectory_type=getattr(m, "traj_method", None) or self.method, replans=0, eta=0.0,
                                     current_velocity=0.0, speed_h=0.0, speed_v=0.0, min_speed=None, avg_speed_moving=0.0,
                                     max_abs_ax=0.0, max_abs_ay=0.0, max_abs_az=0.0, violations=0, status="ASSIGNED",
                                     start_time=datetime.now().isoformat(timespec="seconds"), _sm=0.0, _nm=0,
                                     current_acceleration=0.0, active=True, _sv=0.0, _sa=0.0, _n=0,
                                     wind=self._get().weather.wind_speed_kmh)       # reset for every new delivery
            d.traj_method = cm["trajectory_type"]
        tl = self.tilt.setdefault(did, [0.0, 0])
        tl[0] += row["roll_deg"] ** 2 + row["pitch_deg"] ** 2
        tl[1] += 1
        ts, pos = time.time(), (row["x_m"], row["y_m"], row["z_m"])
        prev, self._prev[did] = self._prev.get(did), (ts, pos)
        dt = min(max(ts - prev[0], 0.02), 0.5) if prev else 1.0 / self.hz      # actual elapsed time
        step = math.dist(pos, prev[1]) if prev else 0.0                         # distance from position samples
        lf = self._life(did)
        v = math.sqrt(row["vx_ms"] ** 2 + row["vy_ms"] ** 2 + row["vz_ms"] ** 2)  # one speed definition everywhere
        lf["max_velocity_ever"] = max(lf["max_velocity_ever"], v)
        if not cm or not cm["active"]:
            return
        a = math.sqrt(row["ax_ms2"] ** 2 + row["ay_ms2"] ** 2 + row["az_ms2"] ** 2)
        cm["flight_time"] += dt
        cm["speed_h"], cm["speed_v"] = math.hypot(row["vx_ms"], row["vy_ms"]), abs(row["vz_ms"])
        cm["min_speed"] = v if cm["min_speed"] is None else min(cm["min_speed"], v)
        if v > 0.5:
            cm["_sm"] += v; cm["_nm"] += 1
            cm["avg_speed_moving"] = cm["_sm"] / cm["_nm"]
        for ax_ in "xyz":
            cm["max_abs_a" + ax_] = max(cm["max_abs_a" + ax_], abs(row[f"a{ax_}_ms2"]))
        if math.hypot(row["ax_ms2"], row["ay_ms2"]) > da.A_MAX * 1.1 or abs(row["az_ms2"]) > da.AZ_MAX * 1.1:
            cm["violations"] += 1
        cm["status"] = self.v6_state(d)
        cm["distance_travelled"] += step
        cm["_sv"] += v; cm["_sa"] += a; cm["_n"] += 1
        cm.update(current_velocity=v, current_acceleration=a, avg_velocity=cm["_sv"] / cm["_n"],
                  avg_acceleration=cm["_sa"] / cm["_n"], max_velocity=max(cm["max_velocity"], v),
                  max_acceleration=max(cm["max_acceleration"], a))
        cm["energy_consumed"] += 180.0 * (1 + 0.004 * a * a + 0.002 * v * v) * dt / 3600.0
        rem = d._path_remaining(len(d.path) - 1) if d.path else 0.0
        cm["distance_remaining"], cm["eta"] = rem, (rem / max(v, 3.0) if rem > 1 else 0.0)
        cm["distance_to_target"] = math.dist(pos, (row["des_x_m"], row["des_y_m"], row["des_z_m"]))
        cm["trajectory_type"] = getattr(d, "traj_method", None) or self.method     # method actually flown
        lf["max_acceleration_ever"] = max(lf["max_acceleration_ever"], a)

    def _finalize(self, did, mission_id):
        cm = self.cm.get(did)
        if not cm or cm["mission_id"] != mission_id or not cm["active"]:
            return
        if any(r["delivery_id"] == str(mission_id) for r in self.hist_deliv.get(did, [])):
            return                                          # never count one delivery twice
        cm["active"] = False
        r = lambda x, n=2: round(x, n)
        self.hist_deliv.setdefault(did, []).append(dict(
            delivery_id=str(mission_id), drone=did, start_location=cm["start"], destination=cm["dest"],
            trajectory_type=cm["trajectory_type"], distance=r(cm["distance_travelled"], 1),
            flight_time=r(cm["flight_time"], 1), avg_velocity=r(cm["avg_velocity"]), max_velocity=r(cm["max_velocity"]),
            avg_acceleration=r(cm["avg_acceleration"]), max_acceleration=r(cm["max_acceleration"]),
            energy_consumed=r(cm["energy_consumed"], 3), replans=cm["replans"], wind_kmh=cm["wind"],
            min_speed=r(cm["min_speed"] or 0), avg_speed_moving=r(cm["avg_speed_moving"]),
            max_abs_ax=r(cm["max_abs_ax"]), max_abs_ay=r(cm["max_abs_ay"]), max_abs_az=r(cm["max_abs_az"]),
            violations=cm["violations"], start_time=cm["start_time"],
            completion_time=datetime.now().isoformat(timespec="seconds")))
        lf = self._life(did)
        lf["deliveries_completed"] += 1
        lf["total_distance"] += cm["distance_travelled"]
        lf["total_flight_time"] += cm["flight_time"]
        lf["total_energy"] += cm["energy_consumed"]
        lf["total_replans"] += cm["replans"]
        lf["average_delivery_time"] = lf["total_flight_time"] / lf["deliveries_completed"]
        self._save()

    def _save(self):
        try:
            tmp = STORE + ".tmp"
            with open(tmp, "w") as f:
                json.dump(dict(history=self.hist_deliv, life=self.life, failed=self.failed_missions), f)
            os.replace(tmp, STORE)                           # atomic: a crash cannot leave a half-written store
        except OSError as exc:
            da.log.error("analytics store write failed: %s", exc)

    def _load(self):
        try:
            with open(STORE) as f:
                d = json.load(f)
            self.hist_deliv, self.life, self.failed_missions = d.get("history", {}), d.get("life", {}), d.get("failed", 0)
        except (OSError, ValueError):
            pass

    def diagnostics(self, did):
        d = self._get().drones[did]
        return dict(diag=getattr(d, "diag", {}), config=da.configure(), replan_requests=self.counters["replan_requests"],
                    replans=self.counters["replans"], skipped=self.counters["replans_skipped"],
                    battery=d.battery, state=self.v6_state(d))

    def current_metrics(self, did):
        cm = self.cm.get(did)
        return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in cm.items() if not k.startswith("_")} if cm else None

    def lifetime(self, did):
        lf = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in self._life(did).items()}
        n = lf["deliveries_completed"] + lf["failed_missions"]
        lf["success_rate_pct"] = round(100.0 * lf["deliveries_completed"] / n, 1) if n else None
        return lf

    def delivery_history(self, did):
        return list(self.hist_deliv.get(did, []))

    def _all(self):
        return [r for h in self.hist_deliv.values() for r in h]

    def set_method(self, method):
        self.method = method
        for did, d in self._get().drones.items():
            d.traj_method = method
            if self.cm.get(did) and self.cm[did]["active"]:
                self.cm[did]["trajectory_type"] = method

    def fleet_analytics(self):
        fleet, allr = self._get(), self._all()
        ids = list(fleet.drones)
        live = lambda did, k: (self.cm[did][k] if self.cm.get(did) and self.cm[did]["active"] else 0.0)
        per = dict(names=ids,
                   deliveries=[self._life(i)["deliveries_completed"] for i in ids],
                   replans=[self._life(i)["total_replans"] + live(i, "replans") for i in ids],
                   distance=[round(self._life(i)["total_distance"] + live(i, "distance_travelled"), 1) for i in ids],
                   energy=[round(self._life(i)["total_energy"] + live(i, "energy_consumed"), 3) for i in ids],
                   battery=[round(fleet.drones[i].battery, 1) for i in ids])
        mean = lambda xs: round(sum(xs) / len(xs), 2) if xs else 0
        bt = []
        for m in dict.fromkeys(r["trajectory_type"] for r in allr):
            g = [r for r in allr if r["trajectory_type"] == m]
            bt.append(dict(method=m, n=len(g), avg_time=mean([r["flight_time"] for r in g]),
                           avg_energy=mean([r["energy_consumed"] for r in g]), avg_distance=mean([r["distance"] for r in g]),
                           avg_replans=mean([r["replans"] for r in g])))
        return dict(fleet_deliveries=len(allr), fleet_distance_m=round(sum(per["distance"]), 1),
                    fleet_energy_wh=round(sum(per["energy"]), 2), fleet_replans=int(sum(per["replans"])),
                    avg_delivery_time_s=mean([r["flight_time"] for r in allr]) if allr else None,
                    # fleet average speed = total distance / total flight time of completed deliveries
                    avg_velocity=round(sum(r["distance"] for r in allr) / sum(r["flight_time"] for r in allr), 2) if allr else None,
                    active_missions=sum(1 for c in self.cm.values() if c["active"]), completed_missions=len(allr),
                    per_drone=per, by_trajectory=bt)

    def research_analytics(self):
        fleet, allr, f = self._get(), self._all(), self.fleet_analytics()
        done, failed = len(allr), self.failed_missions
        hours = sum(self._life(i)["total_flight_time"] + (self.cm[i]["flight_time"] if self.cm.get(i) and self.cm[i]["active"] else 0)
                    for i in fleet.drones) / 3600.0
        usage = {}
        for r in allr:
            usage[r["trajectory_type"]] = usage.get(r["trajectory_type"], 0) + 1
        buckets = []
        for name, lo, hi in (("Calm (<15 km/h)", 0, 15), ("Moderate (15-30)", 15, 30), ("Strong (>30)", 30, 1e9)):
            g = [r for r in allr if lo <= r["wind_kmh"] < hi]
            buckets.append(dict(bucket=name, n=len(g),
                                avg_time=round(sum(r["flight_time"] for r in g) / len(g), 1) if g else 0,
                                avg_energy=round(sum(r["energy_consumed"] for r in g) / len(g), 3) if g else 0))
        rms = {i: round(math.sqrt(t[0] / t[1] / 2), 2) if t[1] else 0 for i, t in self.tilt.items()}
        w = fleet.weather
        return dict(total_deliveries=done, completed_missions=done, failed_missions=failed,
                    success_rate_pct=round(100.0 * done / (done + failed), 1) if done + failed else 100.0,
                    flight_hours=round(hours, 3), total_energy_wh=f["fleet_energy_wh"],
                    avg_delivery_time_s=f["avg_delivery_time_s"], trajectory_usage=usage,
                    failed_per_drone={i: self._life(i)["failed_missions"] for i in fleet.drones},
                    controller_rms_tilt_deg=rms,
                    controller_mean_rms_tilt_deg=round(sum(rms.values()) / len(rms), 2) if rms else 0,
                    replans_per_delivery=round(f["fleet_replans"] / done, 2) if done else 0,
                    replans_per_hour=round(f["fleet_replans"] / hours, 1) if hours else 0,
                    weather=dict(wind_kmh=w.wind_speed_kmh, speed_modifier=w.speed_modifier,
                                 battery_drain_modifier=w.battery_drain_modifier), weather_buckets=buckets)

    # ---------------- views ----------------
    def queue_view(self, did=None):
        f = lambda m: dict(m.to_dict(), v6_status="QUEUED")
        ids = [did] if did else list(self._get().drones)
        return {i: [f(m) for m in self.queues.get(i, [])] for i in ids}

    def analytics(self):
        fleet = self._get()
        done = list(self.delivered.values())
        return dict(
            total_deliveries=len(done),
            avg_delivery_time_s=round(sum(x["elapsed_s"] or 0 for x in done) / len(done), 1) if done else 0,
            total_flight_distance_km=round(sum(x["dist_km"] for x in done), 2),
            trajectory_replans=self.counters["replans"],
            obstacle_avoidance_events=self.counters["obstacle_events"],
            energy_wh=round(sum(x["energy_wh"] for x in done), 2),
            weather=dict(wind_kmh=fleet.weather.wind_speed_kmh, speed_modifier=fleet.weather.speed_modifier),
            per_drone={did: dict(state=self.v6_state(d), battery=round(d.battery, 1),
                                 replans=getattr(d, "replan_count", 0),
                                 queued=len(self.queues.get(did, []))) for did, d in fleet.drones.items()},
            replan_log=self.replans[-50:])


def _locked(fn):
    def wrapper(self, *a, **k):
        with self._lock:
            return fn(self, *a, **k)
    return wrapper


for _n in ("current_metrics", "lifetime", "delivery_history", "fleet_analytics", "research_analytics", "queue_view",
           "analytics", "request_replan", "fail_drone", "check_battery", "set_method", "diagnostics"):
    setattr(V6Coordinator, _n, _locked(getattr(V6Coordinator, _n)))
