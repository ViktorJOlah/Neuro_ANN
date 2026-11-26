"""Run every validation protocol of the Experimental fully active L2/3 model, compare with targets.json,
and write figures + results/validation/summary.json.

    python validation/run_validation.py [--out results/validation] [--quick]

Figures:
  fig1_passive_hcn.png     steady state / sag vs current (pA/pF), ctrl vs HCN block (Olah 2025 Fig 2)
  fig2_epsp_pathways.png   L4 vs L1 EPSP half-widths in 4 conditions, model vs data (Olah 2025 Fig 4h)
  fig3_bap.png             bAP amplitude, latency and Ca2+ along the main apical path
  fig4_dendritic_spikes.png trunk-end current steps: Na+ spike, Ca2+ plateau, 4-AP unmasking (K+ brake)
  fig5_critical_freq.png   tuft Ca2+ vs bAP frequency (Larkum 2007)
  fig6_ca_waves.png        Ca2+ space-time along trunk -> tuft: bAP train, tuft NMDA spike, 4-AP Ca2+ spike
  fig7_nmda_io.png         basal branch input-output, proximal vs distal, with and without NMDA
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from neuron import h  # noqa: E402

import protocols as P  # noqa: E402
from model.l23_cell import L23Cell, load_params  # noqa: E402

C = dict(control="#222222", HCN="#2e8b57", NMDA="#7b3fa0", **{"HCN+NMDA": "#c0392b"}, TTX="#1f77b4",
         **{"TTX+4AP": "#e67e22", "TTX+4AP+Ca": "#999999", "4AP": "#d35400", "4APd": "#d35400"})


def passive_iv(cell, cm_ref_pF=65.0, dens=(-0.5, -1, -2, -3, -4, -5, -6)):
    out = {}
    for c in ("control", "HCN"):
        P._apply(cell, c); rows = []
        for d in dens:
            ic = cell.iclamp(cell.soma[0], 0.5, 100, 300, d * cm_ref_pF * 1e-3)
            vs = P._rec(cell.soma[0](0.5)); t = P._run(cell, 450); ic.amp = 0
            v = np.array(vs); vr = v[(t > 50) & (t < 100)].mean()
            vss = v[(t > 350) & (t < 400)].mean(); vmin = v[(t > 100) & (t < 300)].min()
            m = (t > 100) & (t < 250); x = (v[m] - vr) / (vss - vr)
            rows.append(dict(pA_pF=d, ss=vss - vr, sag=vss - vmin, tau=float(t[m][np.argmax(x >= 1 - np.exp(-1))] - 100)))
        out[c] = rows
    cell.unblock()
    return out


def fig_passive(iv, pm, tg, path):
    fig, ax = plt.subplots(1, 4, figsize=(14, 3.4))
    for c in ("control", "HCN"):
        r = iv[c]; x = [q["pA_pF"] for q in r]
        ax[0].plot(x, [q["ss"] for q in r], "o-", color=C[c], label="model " + ("ctrl" if c == "control" else "HCN block"))
        ax[1].plot(x, [-q["sag"] for q in r], "o-", color=C[c])
        ax[2].plot(x, [q["tau"] for q in r], "o-", color=C[c])
    ax[0].plot([-4, -4], [tg["ss_mV_at_-4pA_pF_ctrl"]["value"], tg["ss_mV_at_-4pA_pF_cs"]["value"]], "s", color="orange",
               ms=8, label="data (Olah 2025 Fig 2b)")
    ax[1].plot([-4], [-tg["sag_mV_at_-4pA_pF"]["value"]], "s", color="orange", ms=8)
    ax[2].axhline(tg["tau_ms_cs"]["value"], color="orange", ls="--", lw=1)
    ax[0].set(xlabel="current (pA/pF, 65 pF)", ylabel="steady-state ΔV (mV)"); ax[0].legend(fontsize=7)
    ax[1].set(xlabel="current (pA/pF)", ylabel="sag (mV)"); ax[2].set(xlabel="current (pA/pF)", ylabel="response τ (ms)")
    lab = ["Rin ctrl", "Rin HCN-", "Vrest ctrl", "ΔVrest"]
    mv = [pm["rin_ctrl"], pm["rin_hcn_block"], pm["vrest_ctrl"], pm["dvrest_hcn_block"]]
    dv = [tg["rin_ctrl_MOhm"]["value"], tg["rin_hcn_block_MOhm"]["value"], tg["vrest_ctrl_mV"]["value"], tg["dvrest_hcn_block_mV"]["value"]]
    xx = np.arange(4)
    ax[3].bar(xx - 0.2, np.abs(mv), 0.4, color="#555", label="model"); ax[3].bar(xx + 0.2, np.abs(dv), 0.4, color="orange", label="data")
    ax[3].set_xticks(xx); ax[3].set_xticklabels(lab, fontsize=7); ax[3].set_ylabel("|value| (MΩ or mV)"); ax[3].legend(fontsize=7)
    fig.suptitle("Passive properties and HCN (Ih_Olah, proximal Gaussian density)")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fig_epsp(m, traces, tg, path):
    conds = P.CONDS4
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
    for k, pth in enumerate(("L4", "L1")):
        xx = np.arange(len(conds)) + k * 5
        ax[0].bar(xx - 0.2, [m[f"hw_{pth}_{c}"] for c in conds], 0.4, color=[C[c] for c in conds])
        ax[0].bar(xx + 0.2, [tg[f"{pth}_{c}"]["value"] for c in conds], 0.4, color=[C[c] for c in conds], alpha=0.35,
                  hatch="//", edgecolor="k")
    ax[0].set_xticks(list(range(4)) + list(range(5, 9)))
    ax[0].set_xticklabels([f"L4\n{c}" for c in conds] + [f"L1\n{c}" for c in conds], fontsize=6)
    ax[0].set_ylabel("EPSP half-width (ms)"); ax[0].set_title("solid = model, hatched = Olah 2025 Fig 4h", fontsize=8)
    for k, pth in enumerate(("L4", "L1")):
        for c in conds:
            t, v = traces[(pth, c)]
            b = v[(t > 80) & (t < 99)].mean(); y = v - b
            ax[1 + k].plot(t - 100, y / max(y.max(), 1e-3), color=C[c], label=c)
        ax[1 + k].set(xlim=(-5, 80), xlabel="ms", title=f"{pth} stimulation (normalised, model)")
    ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fig_bap(b, btr, tg, path):
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
    for lab, r, col in (("longest apical path", b["control"], "k"), ("trunk", btr["control"], "tab:blue")):
        ax[0].plot(r["d"], r["amp"], ".-", color=col, label=lab)
        ax[1].plot(r["d"], r["delay"], ".-", color=col)
        ax[2].plot(r["d"], r["dca_nM"], ".-", color=col)
    dd = np.linspace(0, 350, 50)
    ax[0].plot(dd, b["control"]["amp"][0] * np.exp(-dd / tg["length_constant_um"]["value"]), "--", color="orange",
               label="λ = 317 µm (Waters 2003)")
    ax[0].plot([150], [tg["amp_at_150um_mV"]["value"]], "s", color="orange", ms=8)
    ax[0].set(xlabel="path distance (µm)", ylabel="bAP amplitude (mV)"); ax[0].legend(fontsize=7)
    ax[1].plot(dd, dd / (tg["velocity_m_s"]["value"] * 1000), "--", color="orange", label="0.22 m/s")
    ax[1].set(xlabel="path distance (µm)", ylabel="peak latency re soma (ms)"); ax[1].legend(fontsize=7)
    ax[2].axvspan(*tg["ca_peak_distance_um"]["value"], color="orange", alpha=0.2, label="data: peak 50–100 µm")
    ax[2].set(xlabel="path distance (µm)", ylabel="bAP Δ[Ca]i (nM)"); ax[2].legend(fontsize=7)
    fig.suptitle("Back-propagating AP (single somatic AP)")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fig_dspikes(dj, cth, path):
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
    for c in ("control", "TTX"):
        rows = dj[c]["rows"]
        k = min(range(len(rows)), key=lambda i: abs(rows[i]["amp_nA"] - 1.2))
        r = rows[k]
        ax[0].plot(r["t"] - 100, r["vd"], color=C[c], label=f"{c} dendrite ({r['amp_nA']:.1f} nA)")
        ax[0].plot(r["t"] - 100, r["vs"], color=C[c], ls=":", label=f"{c} soma")
    ax[0].set(xlim=(-5, 40), xlabel="ms", ylabel="Vm (mV)", title=f"EPSC-shaped current at trunk end ({dj['control']['site_um']:.0f} µm)")
    ax[0].legend(fontsize=6)
    for c, (amps, pk, som, tr) in cth.items():
        ax[1].plot(amps, pk, "o-", color=C.get(c, "gray"), label=c)
        ax[2].plot(amps, som, "o-", color=C.get(c, "gray"), label=c)
    ax[1].set(xlabel="50 ms step at trunk end (nA)", ylabel="dendritic plateau Vm (mV)", title="Ca2+ plateau and the K+ brake")
    ax[2].set(xlabel="step (nA)", ylabel="somatic APs"); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def ca_threshold_curves(cell, amps=np.arange(0.1, 1.21, 0.1), conds=("TTX+Ca", "TTX", "TTX+4AP", "control", "4APd")):
    tip = cell.trunk[-1]; out = {}
    for c in conds:
        P._apply(cell, c); pk, som, trs = [], [], None
        for a in amps:
            vd = P._rec(tip(0.5)); vs = P._rec(cell.soma[0](0.5))
            ic = cell.iclamp(tip, 0.5, 100, 50, a); t = P._run(cell, 250); ic.amp = 0
            v = np.array(vd); s = np.array(vs)
            pk.append(float(v[(t > 130) & (t < 150)].mean())); som.append(int(((s[1:] > -10) & (s[:-1] <= -10)).sum()))
        out[c] = (amps, pk, som, trs)
    cell.unblock()
    return out


def fig_cfreq(cf, path):
    fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))
    for c, rows in cf.items():
        ax.plot([r["freq"] for r in rows], [r["ca_integral_nMs"] for r in rows], "o-", color=C.get(c, "k"), label=c)
    ax.axvline(128, color="orange", ls="--", label="128 Hz (Larkum 2007)")
    ax.set(xlabel="5 bAPs at (Hz)", ylabel="tuft Ca2+ integral (nM·s)"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fig_waves(waves, path):
    n = len(waves)
    fig, ax = plt.subplots(2, n, figsize=(4.6 * n, 6), sharey="row")
    for j, (title, r) in enumerate(waves.items()):
        base = r["ca"][:, :int(95 / 0.25)].mean(1, keepdims=True)
        dca = (r["ca"] - base) * 1e6
        o = np.argsort(r["d"])
        im = ax[0, j].imshow(dca[o], aspect="auto", origin="lower", cmap="viridis",
                             extent=[0, r["t"][-1], r["d"][o][0], r["d"][o][-1]], vmin=0, vmax=max(np.percentile(dca, 99.5), 1))
        tb = r["d"][r["tuft"]].min() if r["tuft"].any() else None
        if tb:
            ax[0, j].axhline(tb, color="w", lw=0.6, ls="--")
        ax[0, j].set_title(title, fontsize=9); ax[0, j].set_xlim(80, 300)
        plt.colorbar(im, ax=ax[0, j], label="Δ[Ca]i (nM)")
        im = ax[1, j].imshow(r["v"][o], aspect="auto", origin="lower", cmap="magma", vmin=-80, vmax=20,
                             extent=[0, r["t"][-1], r["d"][o][0], r["d"][o][-1]])
        ax[1, j].set_xlim(80, 300); ax[1, j].set_xlabel("ms")
        plt.colorbar(im, ax=ax[1, j], label="Vm (mV)")
    ax[0, 0].set_ylabel("trunk → tuft path (µm)"); ax[1, 0].set_ylabel("trunk → tuft path (µm)")
    fig.suptitle("Dendritic Ca2+ along the trunk and longest tuft branch (dashed: tuft start)")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fig_nmda(io, path):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for (c, dd), r in io.items():
        ls = "-" if dd == "distal" else "--"
        ax[0].plot(r["expected"], r["measured"], ls, marker="o", ms=3, color=C.get(c, "k"), label=f"{c}, {dd} ({r['d'][0]:.0f}-{r['d'][1]:.0f} µm)")
        ax[1].plot(r["k"], r["dend_peak"], ls, marker="o", ms=3, color=C.get(c, "k"))
    lim = max(r["expected"].max() for r in io.values())
    ax[0].plot([0, lim], [0, lim], ":", color="gray")
    ax[0].set(xlabel="expected EPSP (mV, linear sum)", ylabel="measured EPSP (mV)", title=f"basal branch {next(iter(io.values()))['sec']}")
    ax[0].legend(fontsize=6); ax[1].set(xlabel="synapses", ylabel="local dendritic peak Vm (mV)")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "validation"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tg = json.load(open(os.path.join(HERE, "targets.json")))
    cell = L23Cell(load_params())
    S = {}; t0 = time.time()
    want = lambda k: a.only is None or k in a.only

    if want("passive"):
        pm, _ = P.passive(cell); iv = passive_iv(cell)
        S["passive"] = pm; S["passive_iv"] = iv
        fig_passive(iv, pm, tg["passive"], os.path.join(a.out, "fig1_passive_hcn.png"))
        print("passive", {k: round(v, 2) for k, v in pm.items()}, f"{time.time() - t0:.0f}s")
    if want("epsp"):
        dt0 = cell.p["dt"]; cell.p["dt"] = 0.05
        m, tr = P.epsp_pathways(cell, n_draws=3 if a.quick else 8)
        cell.p["dt"] = dt0
        S["epsp_pathways"] = m
        fig_epsp(m, tr, tg["epsp_pathway_halfwidth_ms"], os.path.join(a.out, "fig2_epsp_pathways.png"))
        print("epsp", {k: round(v, 1) for k, v in m.items() if k.startswith("hw")}, f"{time.time() - t0:.0f}s")
    if want("bap"):
        b = P.bap(cell)
        tr = [(s, seg.x, h.distance(seg.x, sec=s)) for s in cell.trunk for seg in s]
        btr = P.bap(cell, path=tr)
        r = b["control"]; d = r["d"]
        at = lambda x, arr: float(arr[np.argmin(abs(d - x))])
        k = (d > 30) & (d < 250)
        vel = np.polyfit(r["delay"][k], d[k], 1)[0] / 1000.0
        S["bap"] = dict(amp_at_150=at(150, r["amp"]), amp_at_250=at(250, r["amp"]), amp_at_340=at(340, r["amp"]),
                        velocity_m_s=float(vel), ca_peak_distance_um=float(d[np.nanargmax(r["dca_nM"])]),
                        ca_peak_nM=float(np.nanmax(r["dca_nM"])))
        tr10 = P.bap(cell, n_ap=10, freq=20.0)["control"]
        S["bap"]["train_note"] = "see fig3; 10 APs at 20 Hz"
        fig_bap(b, btr, tg["bap"], os.path.join(a.out, "fig3_bap.png"))
        print("bap", S["bap"], f"{time.time() - t0:.0f}s")
    if want("dspike"):
        dj = P.dendritic_injection(cell, site_um=h.distance(0.5, sec=cell.trunk[-1]), amps=np.arange(0.2, 3.01, 0.2),
                                   conds=("control", "TTX"))
        cth = ca_threshold_curves(cell)
        ref = np.array(cth["TTX+Ca"][1])
        S["dendritic"] = dict(
            site_um=dj["control"]["site_um"],
            first_soma_AP_nA=next((r["amp_nA"] for r in dj["control"]["rows"] if r["n_soma_ap"] > 0), None),
            max_soma_APs_brief=max(r["n_soma_ap"] for r in dj["control"]["rows"]),
            ttx_soma_APs=max(r["n_soma_ap"] for r in dj["TTX"]["rows"]),
            ca_boost_at_1nA_mV={c: float(np.array(v[1])[-3] - ref[-3]) for c, v in cth.items()},
            ca_threshold_nA={c: next((float(x) for x, y in zip(v[0], np.array(v[1]) - ref) if y > 10), None) for c, v in cth.items()},
            soma_APs_50ms_step_1nA={c: v[2][-3] for c, v in cth.items()})
        fig_dspikes(dj, cth, os.path.join(a.out, "fig4_dendritic_spikes.png"))
        print("dendritic", S["dendritic"], f"{time.time() - t0:.0f}s")
    if want("cfreq"):
        cf = P.critical_frequency(cell, conds=("control", "4APd"))
        S["critical_frequency"] = cf
        fig_cfreq(cf, os.path.join(a.out, "fig5_critical_freq.png"))
        print("cfreq", [(r["freq"], round(r["ca_integral_nMs"], 1), r["n_spikes"]) for r in cf["control"]], f"{time.time() - t0:.0f}s")
    if want("waves"):
        tip = cell.trunk[-1]
        tsec = P.trunk_tuft_path(cell)[-1][0]   # terminal tuft branch of the plotted trunk-tuft line

        def bap_train(c):
            for k in range(5):
                c.iclamp(c.soma[0], 0.5, 100 + k * 7.0, 2.0, 2.5)
            return None

        def tuft_cluster(c):
            return P.SynCluster(c, tsec, 20, 0.5e-3)

        def dend_step(c):
            c.iclamp(tip, 0.5, 100, 50, 0.5)
            return None
        waves = {}
        for title, fn, cond in (("5 bAPs at 143 Hz", bap_train, "control"),
                                ("tuft cluster, 20 syn (NMDA spike)", tuft_cluster, "control"),
                                ("tuft cluster, NMDA blocked", tuft_cluster, "NMDA"),
                                ("0.5 nA step at trunk end", dend_step, "control"),
                                ("same step, dendritic 4-AP", dend_step, "4APd")):
            waves[title] = P.ca_spacetime(cell, fn, conds=(cond,))[cond]
        S["ca_waves"] = {k: dict(peak_dca_nM=float(((v["ca"] - v["ca"][:, :300].mean(1, keepdims=True)) * 1e6).max()),
                                 extent_frac=float((((v["ca"] - v["ca"][:, :300].mean(1, keepdims=True)) * 1e6).max(1) > 30).mean()),
                                 soma_APs=int(((v["vsoma"][1:] > -10) & (v["vsoma"][:-1] <= -10)).sum()))
                         for k, v in waves.items()}
        fig_waves(waves, os.path.join(a.out, "fig6_ca_waves.png"))
        print("waves", S["ca_waves"], f"{time.time() - t0:.0f}s")
    if want("nmda"):
        io = P.nmda_spike_io(cell, n_max=12 if a.quick else 30, g=0.3e-3)
        S["nmda_io"] = {f"{c}_{d}": dict(supralinearity_pct=float(100 * (r["measured"] / r["expected"]).max()),
                                          max_mV=float(r["measured"].max())) for (c, d), r in io.items()}
        fig_nmda(io, os.path.join(a.out, "fig7_nmda_io.png"))
        print("nmda", S["nmda_io"], f"{time.time() - t0:.0f}s")
    prev = os.path.join(a.out, "summary.json")
    old = json.load(open(prev)) if os.path.exists(prev) and a.only else {}
    old.update(S)
    json.dump(old, open(prev, "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


if __name__ == "__main__":
    main()
