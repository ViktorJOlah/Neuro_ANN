"""Smith et al. 2013 (Nature 503:115) protocol suite for the in vivo constrained model.

Runs smith_probe.py (in parallel subprocesses) for:
  * 8 grating orientations (0..157.5 deg), seed 0                       -> tuning (Fig 1f, ED Fig 2)
  * preferred orientation, seeds 0-2: control, MK-801 (NMDA block in the
    recorded cell) and -50 pA through the 'dendritic pipette' at basal 150  -> Fig 4, ED Fig 9
then analyses the recordings with Smith's own definitions (isolated spike = >= 50 ms from any other spike
at that site = presumed bAP; the rest are spikes in bursts; a burst = a run of non-isolated spikes) and
draws Smith-style figures into results/smith/.

Run: python invivo/smith_suite.py [--jobs 3] [--analyse-only]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import smith_probe as S  # noqa: E402

OUT = os.path.join(ROOT, "results", "smith")
ORIS = [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5]
HYP_SITE = "basal 150"
HYP_NA = -0.05


def jobs():
    j = [dict(ori=o, seed=0, cond="control") for o in ORIS]
    for s in (1, 2):
        j.append(dict(ori=0, seed=s, cond="control"))
    for s in (0, 1, 2):
        j.append(dict(ori=0, seed=s, cond="NMDA"))
        j.append(dict(ori=0, seed=s, cond="control", hyp=True))
    return j


def tag(j):
    t = f"cell82141_{j['cond']}_td0_s{j['seed']}" + (f"_hyp{abs(HYP_NA):g}" if j.get("hyp") else "") + f"_ori{j['ori']:g}"
    return t


def run_one(j):
    cmd = [sys.executable, os.path.join(HERE, "smith_probe.py"), "--cond", j["cond"], "--seed", str(j["seed"]),
           "--ori", str(j["ori"]), "--out", OUT, "--no-fig"]
    if j.get("hyp"):
        cmd += ["--hyper-nA", str(HYP_NA)]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    return tag(j)


def load(t):
    z = np.load(os.path.join(OUT, t + ".npz"), allow_pickle=True)
    return list(z["meta"]), z["V"].astype(float)


def events(v, dt=S.DT_REC):
    pk = S.detect(v, dt); t = pk * dt
    iso = S.classify(t) if len(t) else np.zeros(0, bool)
    bursts = []
    for k in np.nonzero(~iso)[0]:
        if bursts and t[k] - bursts[-1][-1] < 50:
            bursts[-1].append(t[k])
        else:
            bursts.append([t[k]])
    return t, iso, bursts


def osi(rates, oris):
    r = np.asarray(rates, float); th = np.radians(2 * np.asarray(oris))
    if r.sum() == 0:
        return np.nan, np.nan
    v = np.sum(r * np.exp(1j * th)) / r.sum()
    pref = (np.degrees(np.angle(v)) / 2) % 180
    ip = int(np.argmin(np.abs(((np.asarray(oris) - pref + 90) % 180) - 90)))
    io = (ip + len(oris) // 2) % len(oris)
    return float((r[ip] - r[io]) / (r[ip] + r[io])) if r[ip] + r[io] > 0 else np.nan, float(pref)


def analyse():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = {}
    stim = (S.DT_REC, 500.0, 2500.0)
    # ---------------------------------------------------------------- tuning over 8 orientations
    tun = {}
    for o in ORIS:
        meta, V = load(tag(dict(ori=o, seed=0, cond="control")))
        for (lab, name, x, d, kind, diam), v in zip(meta, V):
            t, iso, bursts = events(v)
            on = lambda tt: (tt >= 500) & (tt < 2500)
            tun.setdefault(lab, dict(d=d, kind=kind, spikes=[], baps=[], bursts=[]))
            tun[lab]["spikes"].append(on(t).sum() / 2.0)
            tun[lab]["baps"].append(on(t[iso]).sum() / 2.0)
            tun[lab]["bursts"].append(sum(1 for b in bursts if 500 <= b[0] < 2500) / 2.0)
    res["tuning"] = {lab: dict(d=v["d"], spikes_hz=v["spikes"], baps_hz=v["baps"], bursts_hz=v["bursts"],
                               osi_spikes=osi(v["spikes"], ORIS)[0], osi_bursts=osi(v["bursts"], ORIS)[0],
                               osi_baps=osi(v["baps"], ORIS)[0], pref_bursts=osi(v["bursts"], ORIS)[1],
                               pref_baps=osi(v["baps"], ORIS)[1]) for lab, v in tun.items()}
    # ---------------------------------------------------------------- rates (Fig 2a), all 8 orientations
    rates = {}
    for o in ORIS:
        meta, V = load(tag(dict(ori=o, seed=0, cond="control")))
        st = [S.site_stats(v, soma_v=V[0]) for v in V]
        for m, s in zip(meta, st):
            grp = "soma" if m[4] == "soma" else ("proximal" if m[3] < 50 else ("distal" if m[3] > 75 else None))
            if grp is None or m[4] == "trunk":
                continue
            rates.setdefault(grp, dict(maxinst=[], mean=[], fano=[]))
            rates[grp]["maxinst"].append(s["max_inst_rate_hz"]); rates[grp]["mean"].append(s["mean_rate_hz"]); rates[grp]["fano"].append(s["fano"])
    res["rates"] = {g: {k: float(np.nanmean(v)) for k, v in d.items()} for g, d in rates.items()}
    # ---------------------------------------------------------------- bAP amplitude / width vs distance
    bap = []
    for s_ in (0, 1, 2):
        meta, V = load(tag(dict(ori=0, seed=s_, cond="control")))
        for m, v in zip(meta, V):
            a, w, n = S.bap_at_site(v, V[0])
            bap.append(dict(site=m[0], d=m[3], kind=m[4], amp=a, fwhm=w, n=n))
    res["bap"] = bap
    def grp_mean(lo, hi, key):
        x = [b[key] for b in bap if lo <= b["d"] <= hi and b["kind"] != "trunk" and not np.isnan(b[key])]
        return float(np.mean(x)) if x else np.nan
    res["bap_groups"] = {"soma": dict(amp=grp_mean(0, 0, "amp"), fwhm=grp_mean(0, 0, "fwhm")),
                         "100-150": dict(amp=grp_mean(95, 155, "amp"), fwhm=grp_mean(95, 155, "fwhm")),
                         "175-300": dict(amp=grp_mean(170, 310, "amp"), fwhm=grp_mean(170, 310, "fwhm"))}
    # ---------------------------------------------------------------- hyperpolarisation and MK-801 at the preferred orientation
    def counts(cond, hyp, site):
        b_, ba_, sp_, som = [], [], [], []
        for s_ in (0, 1, 2):
            meta, V = load(tag(dict(ori=0, seed=s_, cond=cond, hyp=hyp)))
            i = [m[0] for m in meta].index(site)
            t, iso, bursts = events(V[i])
            ba_.append(iso.sum() / 3.0); b_.append(len(bursts) / 3.0); sp_.append(len(t) / 3.0)
            som.append(len(S.detect(V[0], prom=30.0)) / 3.0)
        return dict(bursts_hz=float(np.mean(b_)), baps_hz=float(np.mean(ba_)), spikes_hz=float(np.mean(sp_)), soma_hz=float(np.mean(som)))
    res["hyperpol"] = {"site": HYP_SITE, "I_nA": HYP_NA, "control": counts("control", False, HYP_SITE),
                       "hyperpol": counts("control", True, HYP_SITE)}
    c, hp = res["hyperpol"]["control"], res["hyperpol"]["hyperpol"]
    res["hyperpol"]["decrease_pct"] = dict(bursts=100 * (1 - hp["bursts_hz"] / c["bursts_hz"]) if c["bursts_hz"] else np.nan,
                                           baps=100 * (1 - hp["baps_hz"] / c["baps_hz"]) if c["baps_hz"] else np.nan)
    res["mk801"] = {site: dict(control=counts("control", False, site), NMDA_block=counts("NMDA", False, site))
                    for site in ("basal 150", "oblique 150", "basal 100")}
    json.dump(res, open(os.path.join(OUT, "smith_summary.json"), "w"), indent=1, default=float)
    figures(res)
    return res


def _trace(ax, t, v, col, lim=(-80, 0), lw=0.6):
    ax.plot(t, v, color=col, lw=lw)
    ax.set_ylim(*lim); ax.axis("off")


def figures(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dt = S.DT_REC
    blue, green = "#3b78c4", "#3a9a4a"
    # Fig 1-like: soma and dendrite, preferred vs orthogonal
    fig, ax = plt.subplots(6, 2, figsize=(10, 8))
    sites = ["soma", "oblique 150", "basal 150"]
    for col, o in enumerate((0, 90)):
        meta, V = load(tag(dict(ori=o, seed=0, cond="control")))
        labs = [m[0] for m in meta]
        for r, sname in enumerate(sites):
            v = V[labs.index(sname)]; t = np.arange(len(v)) * dt
            m = (t >= 300) & (t < 2700)
            for k, (lo, hi) in enumerate(((300, 1500), (1500, 2700))):
                mm = (t >= lo) & (t < hi)
                a = ax[2 * r + k, col]
                _trace(a, t[mm], v[mm], blue if sname == "soma" else green, (-80, 30) if sname == "soma" else (-80, 0))
                a.axvspan(max(lo, 500), min(hi, 2500), color="0.93", zorder=-1)
                if k == 0:
                    a.set_title(f"{sname}  ({'preferred' if o == 0 else 'preferred + 90°'})", fontsize=8, loc="left")
    a = ax[-1, -1]; a.plot([2500, 2700], [-75, -75], "k", lw=1.5); a.plot([2700, 2700], [-75, -55], "k", lw=1.5)
    a.text(2600, -79, "200 ms", ha="center", va="top", fontsize=7); a.text(2710, -65, "20 mV", fontsize=7)
    fig.suptitle("Model 'pipette' recordings during drifting gratings (grey: grating), cf. Smith et al. 2013 Fig 1c,d", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_s1_traces_pref_orth.png"), dpi=150); plt.close(fig)
    # Fig 1g / ED 1d-like: burst examples at increasing distance
    meta, V = load(tag(dict(ori=0, seed=0, cond="control")))
    rows = [(m, v) for m, v in zip(meta, V) if m[4] not in ("soma", "trunk") and m[3] >= 75]
    rows.sort(key=lambda r: r[0][3])
    fig, ax = plt.subplots(len(rows), 3, figsize=(8, 0.9 * len(rows) + 0.6))
    for i, (m, v) in enumerate(rows):
        t, iso, bursts = events(v)
        bs = sorted(bursts, key=lambda b: -len(b))[:3]
        for k in range(3):
            a = ax[i, k]
            if k < len(bs):
                c0 = bs[k][0] - 60
                idx = slice(int(c0 / dt), int((c0 + 300) / dt))
                _trace(a, np.arange(idx.stop - idx.start) * dt, v[idx], "k", (-80, 0), lw=0.5)
            else:
                a.axis("off")
        ax[i, 0].text(-20, -40, f"{m[3]:.0f} µm\n({m[4]})", ha="right", va="center", fontsize=7)
    a = ax[-1, -1]; a.plot([100, 300], [-78, -78], "k", lw=1.5); a.plot([300, 300], [-78, -58], "k", lw=1.5)
    a.text(200, -80, "200 ms", ha="center", va="top", fontsize=7); a.text(305, -68, "20 mV", fontsize=7)
    fig.suptitle("Largest dendritic bursts at each site (preferred orientation), cf. Smith ED Fig 1d", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_s2_burst_examples.png"), dpi=150); plt.close(fig)
    # summary panel: bAP vs distance, rates, tuning, hyperpol, MK-801
    fig, ax = plt.subplots(2, 3, figsize=(12, 7))
    b = [x for x in res["bap"] if x["kind"] != "trunk" and not np.isnan(x["amp"])]
    ax[0, 0].scatter([x["d"] for x in b], [x["amp"] for x in b], s=12, color=green, label="model (each site, 3 trials)")
    for (lo, hi), key in (((0, 5), "soma"), ((100, 150), "100-150"), ((175, 300), "175-300")):
        y = S.SMITH["bap_amp_mV"][key]
        ax[0, 0].plot([lo, hi], [y, y], color="crimson", lw=3, alpha=0.7)
    ax[0, 0].plot([], [], color="crimson", lw=3, label="Smith 2013 ED Fig 1e")
    ax[0, 0].set_xlabel("distance from soma (µm)"); ax[0, 0].set_ylabel("bAP amplitude (mV)"); ax[0, 0].legend(fontsize=7)
    ax[1, 0].scatter([x["d"] for x in b], [x["fwhm"] for x in b], s=12, color=green)
    for (lo, hi), key in (((0, 5), "soma"), ((100, 150), "100-150"), ((175, 300), "175-300")):
        y = S.SMITH["bap_fwhm_ms"][key]
        ax[1, 0].plot([lo, hi], [y, y], color="crimson", lw=3, alpha=0.7)
    ax[1, 0].set_xlabel("distance from soma (µm)"); ax[1, 0].set_ylabel("bAP FWHM (ms)")
    grp = ["soma", "proximal", "distal"]
    for k, (key, sk, lab) in enumerate((("maxinst", "max_inst_rate_hz", "max inst. rate (Hz)"), ("mean", "mean_rate_hz", "mean rate (Hz)"),
                                       ("fano", "fano", "variance/mean"))):
        a = ax[0, 1] if k == 0 else (ax[0, 2] if k == 1 else ax[1, 1])
        mv = [res["rates"].get(g, {}).get(key, np.nan) for g in grp]
        sv = [S.SMITH[sk][g] for g in grp]
        xx = np.arange(3)
        a.bar(xx - 0.2, mv, 0.4, color=green, label="model"); a.bar(xx + 0.2, sv, 0.4, color="crimson", alpha=0.7, label="Smith Fig 2a")
        a.set_xticks(xx); a.set_xticklabels(grp); a.set_ylabel(lab)
        if k == 0:
            a.legend(fontsize=7)
    hp = res["hyperpol"]["decrease_pct"]
    mk = res["mk801"]
    a = ax[1, 2]
    labels = ["bursts\n-50 pA", "bAPs\n-50 pA", "bursts\nMK-801", "bAPs\nMK-801"]
    mkb = np.mean([100 * (1 - v["NMDA_block"]["bursts_hz"] / v["control"]["bursts_hz"]) for v in mk.values() if v["control"]["bursts_hz"] > 0])
    mka = np.mean([100 * (1 - v["NMDA_block"]["baps_hz"] / v["control"]["baps_hz"]) for v in mk.values() if v["control"]["baps_hz"] > 0])
    a.bar(np.arange(4) - 0.2, [hp["bursts"], hp["baps"], mkb, mka], 0.4, color=green, label="model")
    a.bar(np.arange(4) + 0.2, [68, 30, 100, 98], 0.4, color="crimson", alpha=0.7, label="Smith (Fig 4a, ED Fig 9d)")
    a.set_xticks(range(4)); a.set_xticklabels(labels, fontsize=8); a.set_ylabel("decrease (%)"); a.legend(fontsize=7)
    fig.suptitle("Model vs Smith et al. 2013: bAP attenuation, spike statistics, hyperpolarisation and NMDA block", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_s3_summary.png"), dpi=150); plt.close(fig)
    # tuning
    tu = res["tuning"]
    labs = ["soma", "oblique 150", "basal 100", "basal 150"]
    fig, ax = plt.subplots(1, len(labs), figsize=(3 * len(labs), 3), subplot_kw=dict(projection="polar"))
    th = np.radians(np.r_[ORIS, np.add(ORIS, 180), ORIS[0]] * 2 / 2)
    for a, lab in zip(ax, labs):
        if lab not in tu:
            a.axis("off"); continue
        for key, col in (("bursts_hz", "#c0392b"), ("baps_hz", "0.4"), ("spikes_hz", blue)):
            r = np.r_[tu[lab][key], tu[lab][key], tu[lab][key][0]]
            if lab == "soma" and key != "spikes_hz":
                continue
            a.plot(th, r, color=col, lw=1.2, label=key.replace("_hz", ""))
        a.set_title(f"{lab}\nOSI bursts {tu[lab]['osi_bursts']:.2f}, bAPs {tu[lab]['osi_baps']:.2f}" if lab != "soma"
                    else f"soma\nOSI {tu[lab]['osi_spikes']:.2f}", fontsize=8)
        a.set_xticklabels([]); a.tick_params(labelsize=6)
    ax[-1].legend(fontsize=6, loc="lower right", bbox_to_anchor=(1.3, -0.1))
    fig.suptitle("Orientation tuning (2 s gratings, 8 orientations), cf. Smith Fig 1f, ED Fig 2", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_s4_tuning.png"), dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--analyse-only", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if not a.analyse_only:
        todo = [j for j in jobs() if not os.path.exists(os.path.join(OUT, tag(j) + ".npz"))]
        with ThreadPoolExecutor(a.jobs) as ex:
            for t in ex.map(run_one, todo):
                print("done", t, flush=True)
    r = analyse()
    print(json.dumps({k: r[k] for k in ("bap_groups", "rates", "hyperpol", "mk801")}, indent=1, default=float))
    print({k: (round(v["osi_spikes"], 2), round(v["osi_bursts"], 2), round(v["osi_baps"], 2)) for k, v in r["tuning"].items()})


if __name__ == "__main__":
    main()
