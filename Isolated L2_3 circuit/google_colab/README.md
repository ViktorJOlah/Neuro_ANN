# Hybrid L2/3 circuit on Google Colab (GPU)

This folder runs the hybrid L2/3 circuit (`circuit/hybrid.py`: 56,057 ANN pyramidal cells from the
Standard model + 9,892 GLIF interneurons in NEST) on a Colab GPU, and then makes the figures and
statistics of the run.

| file | what it is |
|---|---|
| `hybrid_l23_colab.ipynb` | the notebook, annotated from start to finish: GPU check → install NEST/BMTK → unpack the bundle → choose a condition → check the PC layer on the GPU and pick the faster front-end mode → run → statistics → figures → save to Drive |
| `make_bundle.py` | builds `hybrid_l23_bundle.zip` (≈1.3 GB), which holds everything the notebook needs: the circuit code, the Standard model, all fitted library models, the prepared circuit, the Allen files NEST reads, the replay and reference spikes, the cell classes and the Neuropixels rates |

## How to run it

1. Get `hybrid_l23_bundle.zip` (the current one is in the project files, `google_colab/`), or build it
   (see below).
2. Upload it to Google Drive as `MyDrive/hybrid_l23_colab/hybrid_l23_bundle.zip`.
3. Open `hybrid_l23_colab.ipynb` in Colab (File → Upload notebook, or open it from GitHub), set
   *Runtime → Change runtime type → GPU*, and run all cells. Pick the condition in Step 4:

| condition | PCs | gain onto PCs |
|---|---|---|
| `homogeneous` | every PC is the Standard model | 0.848 |
| `library_shuffled` | 38 library cells, placed at random | 0.8103 |
| `library_depth` | the 38 library cells, placed by depth | 0.8103 |
| `library_rheo_matched` | library, input scaled by rheobase / 90 pA | 0.8813 |
| `eight_fits_shuffled` | the earlier 8-fit set | 1.284 |

PC → interneuron synapses stay at Allen's weights (1.0) in every condition. Results (spikes, run info,
`summary.json`, `figures/`) go to `MyDrive/hybrid_l23_colab/runs/<condition>_<tstop>ms_<date>/`.

## What to expect

On the 4-core CPU cloud machine a 3 s run took 43 min (homogeneous) to about 60 min
(heterogeneous). About 93% of that is the PCs' CNN-LSTM front end: every millisecond it re-runs both
convolutions and the LSTM over the whole 64 ms window, for all 56k cells. That part moves to the GPU.
The NEST interneurons (≈90 s per 3 s) and spike routing (≈10 s) stay on the CPU, so they set the
floor: a 3 s run should take a few minutes on a T4, less on an L4 or A100. Step 5 of the notebook
measures the per-step time on your GPU and prints the expected run time before the run starts.

Heterogeneous conditions run one batch per fitted model, one after the other (38 batches of about
1,500 PCs for the library), so they take longer than the homogeneous run.

## Same model, same code

The bundle copies the repository code unchanged. The only addition is the `--no-front-jit` option of
`hybrid.py run`, which runs the front end without XLA (on some GPUs the cuDNN LSTM path is faster).
Both modes compute the same function, and the notebook picks the faster one. On a GPU the
floating-point operations run in a different order than on the CPU. The network amplifies those
tiny differences, so individual spike times can differ from a CPU run, but the rate statistics should
agree. Step 5 shows how many spikes of a 6-cell test agree within 1 ms.

The notebook was tested end to end on CPU (with a stand-in for Colab's Drive mount): the 300 ms
homogeneous run gives the same spikes as the earlier CPU run with the same settings.

## Rebuilding the bundle

After code or model changes (for example the synaptic-kernel-layer circuit), rebuild the bundle so
Colab runs the new version:

```bash
cd "Isolated L2_3 circuit"; export PYTHONPATH=$PWD
V1=/path/to/v1_point; X=/path/to/l23_extract; R=/path/to/streams    # see circuit/README.md
python -m circuit.extract_l23 --network-dir $V1/network --out $X
python -m circuit.hybrid prepare --config $V1/config.json --extract-dir $X --v1ext-spikes $R/v1ext_spikes.h5 --out H/prep
python google_colab/make_bundle.py --prep H/prep --extract-dir $X --streams $R \
    --invivo /path/to/npx_l23_rs_rates_all.npy --out hybrid_l23_bundle.zip
```

`v1ext_spikes.h5` and `l23_reference_spikes.h5` come from `circuit.simulate streams` (they are also in
the project files under `isolated_l23/gratings_ori315_trial0/`), and `npx_l23_rs_rates_all.npy` is
in `isolated_l23/invivo_comparison/`. Extracting and preparing take about 1.5 min together.
