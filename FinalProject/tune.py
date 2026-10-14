import advanced_flight_controller as a, numpy as np, itertools, sys, json
n=496; D=3; TRAIN=[101,102]
def ev(mode,kw):
    r=[a._bench_simulate(mode,i,n,a._bench_wind(sd,n),sd,D,kw) for sd in TRAIN for i in range(6)]
    return float(np.mean([x['rmse'] for x in r]))
which=sys.argv[1]; out=[]
if which=="lqr":
    for qp,qv,r in itertools.product((0.05,0.1,0.2),(0.05,0.1,0.3),(2,6,20)):
        kw=dict(q_pos=qp,q_vel=qv,q_int=0.5,r=r,max_accel=6.5); out.append((ev("LQR",kw),kw))
else:
    # lag-aware 3-state MPC (command-rate input, so r is on d(v_cmd)/dt and sits lower than the legacy 2-state values)
    for H,qp,qv,r,tau in itertools.product((20,30),(1,3),(1,3),(0.1,0.3,1.0),(0.35,0.45)):
        kw=dict(horizon=H,q_pos=qp,q_vel=qv,q_terminal=qp*2.5,r=r,max_accel=6.5,delay_ticks=D,model="lag",tau=tau,gain=0.94); out.append((ev("MPC",kw),kw)); print(round(out[-1][0],3),kw,flush=True)
out.sort(key=lambda x:x[0]); json.dump(out[:8],open(f"tune_{which}.json","w"),indent=1); print(which,[round(o[0],3) for o in out[:5]]); print(out[0][1])
