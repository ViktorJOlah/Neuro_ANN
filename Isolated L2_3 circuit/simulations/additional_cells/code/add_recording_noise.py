"""Add realistic patch-clamp recording noise to clean simulated Vm traces.

Noise model (measured by measure_noise.py from real Allen whole-cell current-clamp recordings,
DANDI 000020, Slc17a7 excitatory cells, MultiClamp 700B, 10 kHz Bessel, 50 kHz sampling):
  * Baseline Vm at zero command current, >= 0.3 s after any command change, spike-free,
    point-sampled every 1 ms like the simulations (so aliasing of >500 Hz noise is included).
  * One median Welch PSD per recording session (0-500 Hz, 1.95 Hz bins). It contains
    everything a real baseline has: amplifier/thermal noise, 60 Hz line hum and harmonics, and
    spontaneous membrane fluctuations of the cell in the slice.
  * For each simulated trace, one session is drawn at random and stationary Gaussian noise
    with exactly that PSD is synthesised (random phases), then added to Vm. Below 1.95 Hz the
    PSD is held at its 1.95 Hz value. Sessions with unhealthy rest (> -65 mV) are excluded.
The injected current is left clean: in a real recording the command waveform is known exactly.

Writes <protocol>_noisy.npy (same layout as the clean file, Vm + noise) and
<protocol>_noise_params.csv (session used, seed, realised noise RMS) next to the clean data.

Usage: python add_recording_noise.py --data ../data/521409057 --psd ../noise_model/noise_psd.npz
"""
import argparse
import csv
import json
import os

import numpy as np

PROTOCOLS = ("square", "ramp", "noise")


def load_psd(path, vrest_max=-65.0):
    z = np.load(path)
    meta = json.load(open(path.replace(".npz", ".json")))
    keep = [i for i, m in enumerate(meta) if m["vrest_mV"] < vrest_max]
    return z["freq"], z["psd"][keep], [meta[i]["session"] for i in keep]


def synth(n, freq, psd, rng, fs=1000.0):
    f = np.fft.rfftfreq(n, 1.0 / fs)
    s = np.interp(np.clip(f, freq[1], freq[-1]), freq, psd)  # one-sided, mV^2/Hz
    s[0] = 0.0
    w = np.fft.rfft(rng.standard_normal(n))  # unit white noise: one-sided density 2/fs
    return np.fft.irfft(w * np.sqrt(s * fs / 2.0), n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--psd", required=True)
    ap.add_argument("--seed", type=int, default=None,
                    help="default: 20261122 + specimen id, so each cell gets its own noise")
    a = ap.parse_args()
    if a.seed is None:
        a.seed = 20261122 + int(os.path.basename(os.path.normpath(a.data)))
    freq, psd, sessions = load_psd(a.psd)
    rng = np.random.default_rng(a.seed)
    for p in PROTOCOLS:
        x = np.load(os.path.join(a.data, "%s.npy" % p))
        y = x.copy()
        rows = []
        for k in range(len(x)):
            j = int(rng.integers(len(sessions)))
            seed = int(rng.integers(2**31))
            e = synth(x.shape[1], freq, psd[j], np.random.default_rng(seed))
            y[k, :, 1] += e.astype(np.float32)
            rows.append(dict(idx=k, master_seed=a.seed, noise_session=sessions[j], noise_seed=seed,
                             noise_rms_mV=float(e.std())))
        np.save(os.path.join(a.data, "%s_noisy.npy" % p), y)
        with open(os.path.join(a.data, "%s_noise_params.csv" % p), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(p, "noise RMS median %.3f mV" % np.median([r["noise_rms_mV"] for r in rows]))


if __name__ == "__main__":
    main()
