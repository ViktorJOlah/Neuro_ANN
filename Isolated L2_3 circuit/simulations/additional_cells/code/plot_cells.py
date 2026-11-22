"""Overview plots for the additional cells (and the first cell for comparison).

  example_traces.png   one row per cell: square / ramp / noise example, clean vs noisy Vm
  fi_curves.png        spikes in the 1 s square vs amplitude, all cells
  noise_model.png      measured baseline PSDs (real recordings) and a synthesised example

Usage: python plot_cells.py --data ../data --first ../../data --psd ../noise_model/noise_psd.npz --out ../plots
"""
import argparse
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from add_recording_noise import load_psd, synth  # noqa: E402


def rows(path):
    return list(csv.DictReader(open(path)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--first", required=True)
    ap.add_argument("--psd", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cells = sorted(d for d in os.listdir(a.data) if os.path.isdir(os.path.join(a.data, d)))

    # example traces
    fig, ax = plt.subplots(len(cells), 3, figsize=(16, 2.6 * len(cells)), squeeze=False)
    for r, c in enumerate(cells):
        meta = json.load(open(os.path.join(a.data, c, "metadata.json")))
        for col, p in enumerate(("square", "ramp", "noise")):
            x = np.load(os.path.join(a.data, c, "%s.npy" % p), mmap_mode="r")
            y = np.load(os.path.join(a.data, c, "%s_noisy.npy" % p), mmap_mode="r")
            ns = np.array([int(q["n_spikes"]) for q in rows(os.path.join(a.data, c, "%s_params.csv" % p))])
            k = int(np.argmin(np.abs(ns - np.percentile(ns, 70))))
            t = np.arange(x.shape[1]) / 1000.0
            ax[r, col].plot(t, y[k, :, 1], color="#d95f02", lw=0.6, label="noisy")
            ax[r, col].plot(t, x[k, :, 1], color="k", lw=0.5, label="clean")
            a2 = ax[r, col].twinx()
            a2.plot(t, x[k, :, 0], color="#1b9e77", lw=0.6)
            a2.set_ylabel("I (pA)", color="#1b9e77", fontsize=7)
            a2.tick_params(labelsize=7)
            ax[r, col].set_title("%s  %s #%d (%d spikes)" % (c, p, k, ns[k]), fontsize=8)
            ax[r, col].tick_params(labelsize=7)
            if col == 0:
                ax[r, col].set_ylabel("%s\nrheo %.0f pA\nVm (mV)" % (meta["model"]["specimen_name"].split(";")[0],
                                                                 meta["calibration"]["rheobase_pA"]), fontsize=8)
    ax[0, 0].legend(fontsize=7, loc="upper left")
    for col in range(3):
        ax[-1, col].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "example_traces.png"), dpi=110)
    plt.close(fig)

    # F-I
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    first = rows(os.path.join(a.first, "square_params.csv"))
    ax.plot([float(q["amplitude_pA"]) for q in first], [int(q["n_spikes"]) for q in first], "k-",
            lw=2, label="487661754 (first cell, perisomatic)")
    for c in cells:
        q = rows(os.path.join(a.data, c, "square_params.csv"))
        ax.plot([float(v["amplitude_pA"]) for v in q], [int(v["n_spikes"]) for v in q], "-o", ms=2, lw=1, label=c)
    ax.set_xlabel("1 s step amplitude (pA)")
    ax.set_ylabel("spikes in 1 s")
    ax.legend(fontsize=7)
    ax.set_title("F-I of the long-square simulations")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "fi_curves.png"), dpi=130)
    plt.close(fig)

    # noise model
    freq, psd, sessions = load_psd(a.psd)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for p, s in zip(psd, sessions):
        ax[0].semilogy(freq[1:], p[1:], lw=0.8, label=s.split("_")[0])
    ax[0].set_xlabel("Hz")
    ax[0].set_ylabel("PSD (mV$^2$/Hz), 1 ms point-sampled")
    ax[0].set_title("baseline Vm noise, real Allen recordings")
    ax[0].legend(fontsize=6)
    e = synth(3000, freq, psd[0], np.random.default_rng(0))
    ax[1].plot(np.arange(3000), e, lw=0.6)
    ax[1].set_title("synthesised noise (session %s), RMS %.3f mV" % (sessions[0].split("_")[0], e.std()))
    ax[1].set_xlabel("ms")
    ax[1].set_ylabel("mV")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "noise_model.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
