import advanced_flight_controller as a, numpy as np
base=a.DELAYED_MPC_KW; rows=[]
cfgs=[("H=5",dict(horizon=5)),("H=10",dict(horizon=10)),("H=20 (chosen)",{}),("H=30",dict(horizon=30)),("tau .25",dict(tau=0.25)),("tau .50",dict(tau=0.50)),("no plant-accel limit",dict(plant_accel=99.0)),("legacy 2-state",a.LEGACY_MPC_KW),
      ("q_pos 4",dict(q_pos=4.0,q_terminal=10.0)),("r 0.3",dict(r=0.3)),("r 3",dict(r=3.0)),("no delay comp",dict(delay_ticks=0))]
for name,ov in cfgs:
    kw={**base,**ov} if name!="legacy 2-state" else dict(ov); r=a.run_delayed_study(seeds=range(21,27),modes=("MPC",),mpc_kw=kw)["MPC"]
    print(f"{name:<16} RMSE {r['rmse']:.3f}±{r['rmse_std']:.3f}  settled {r['settled']*100:.0f}%  viol/seed {r['viol']:.1f}",flush=True)
l=a.run_delayed_study(seeds=range(21,27),modes=("LQR",))["LQR"]; print(f"LQR ref seeds 21-26: {l['rmse']:.3f}")
