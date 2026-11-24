import json, glob, os, sys
SET = ('_' + os.environ['EVOKED_SET']) if os.environ.get('EVOKED_SET') else ''
for f in sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results" + SET, "*", "*", "metrics.json"))):
    if "/oracle" in f:
        continue
    pass
for f in sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results" + SET, "*", "*_*", "metrics.json"))):
    d = json.load(open(f)); e = d["eval"]; z = e.get("no_synaptic_input", {})
    print(f"{d['cell']} S={d['somatic']} {d['model']:9s} {d['train_minutes']:5.1f}min  rmse {e['rmse_mV']:.2f} away {e['rmse_away_mV']:.2f} "
          f"L1 {e['rmse_L1_mV']:.2f}/{e['sd_L1_mV']:.2f} L23 {e['rmse_L23_mV']:.2f}/{e['sd_L23_mV']:.2f} Γ {e['gamma']:.2f} n {e['n_true']}/{e['n_model']} "
          f"|Δn| {e['count_abs_err']:.2f} r {e['count_r']:.2f} | D=0: away {z.get('rmse_away_mV', float('nan')):.2f} L1 {z.get('rmse_L1_mV', float('nan')):.2f} L23 {z.get('rmse_L23_mV', float('nan')):.2f}")
