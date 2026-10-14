import advanced_flight_controller as a, json, sys
res=a.run_delayed_study(seeds=range(1,21))
a.print_delayed_study(res)
json.dump(res,open("study_main.json","w"),indent=1,default=float)
