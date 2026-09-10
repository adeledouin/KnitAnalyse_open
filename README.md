# KnitAnalyse_open

**Data analysis pipeline for seismic-like events in knitted fabrics**

*Adèle Douin, Samuel Poincloux, Jean-Philippe Bruneton & Frédéric Lechenault*
*Laboratoire de Physique, ENS Paris — PSL*

---

## Scientific context

When a knitted fabric is mechanically stretched, its mechanical response is not smooth: it exhibits sudden, abrupt force drops caused by collective rearrangements of the yarn contacts. These *slip events*, or *knitquakes*, are intermittent, scale-invariant, and statistically analogous to earthquakes — they follow the same power-law amplitude distribution and are spatially correlated with fault-like deformation fields in the fabric.

This makes knitted fabric an experimental system for studying seismic-like dynamics in controlled laboratory conditions, with full access to both the global mechanical signal and the spatially resolved displacement field of each stitch.

This repository provides the data analysis pipeline used to process, characterize and prepare these experimental signals — from raw force measurements and image sequences to structured event catalogues and normalized time series ready for machine learning.

---

## Project

The analysis is organized around three successive stages:

**Signal preprocessing (`scalar`)**
Loads raw force time series from individual experiments, removes the slowly varying affine response by linear detrending, and applies a two-step normalization (global per cycle, then local over a sliding window) to obtain a stationary, cycle-to-cycle comparable signal `f`. Force drop events are detected and characterized by their amplitude `Δf`.

**Statistical characterization (`scalarstats`)**
Computes event statistics across all experiments: amplitude distributions, inter-event time distributions, scaling exponents. Verifies the scale-invariant regime and defines event classes on a logarithmic scale (from noise to catastrophic quakes), as used in the classification step downstream.

**Event-field analysis (`scalarevent`)**
Aligns force events with the simultaneously acquired image sequences. For each detected event, extracts the corresponding stitch displacement field, computes vorticity profiles, and characterizes the spatial morphology of the associated fault-like structures. Produces synchronized event catalogues linking scalar measurements, image data, and event metadata.

A separate entry point (`Main_remote_NN.py`) prepares the structured output — stacked time series of `f`, `δf`, and past event labels — in the format expected by the neural network prediction models.

---

## Repository structure

```
KnitAnalyse_open/
├── Main_remote.py          # Main entry point — signal analysis pipeline
├── Main_remote_NN.py       # Entry point — prepares data for NN prediction
├── Config_exp.py           # Experiment-level configuration (per knit, per run)
├── Config_plot.py          # Figure configuration
├── classConfig.py          # Config class shared across modules
├── dictdata.py             # Data dictionaries and path management
├── memory.py               # Caching utilities
├── transfert_csv.py        # CSV export of processed signals
├── KnitAnalyse_autorun.sh  # Batch execution script
│
├── Datas/
│   ├── classSignal.py      # Force signal and image signal loaders
│   └── classEvent.py       # Event metadata and field extraction
│
├── Sub/
│   ├── sub_scalar.py       # Signal preprocessing routines
│   ├── sub_scalarstats.py  # Statistical analysis routines
│   ├── sub_scalarevent.py  # Event-field alignment routines
│   └── sub_transfert_csv.py
│
└── Utils/
    ├── classStat.py        # Histogram and distribution utilities
    └── classPlot.py        # Plotting wrappers
```

---

## Usage

Each experiment is identified by a knit reference, an experiment number, and a working version. The pipeline is run from the command line:

```bash
python Main_remote.py <data_path> <knit_ref> <exp_number> <version> [--scalar] [--scalarstats] [--scalarevent]
```

The `--remote` flag suppresses figure display (for cluster runs). Each stage (`--scalar`, `--scalarstats`, `--scalarevent`) can be run independently.

Example:
```bash
python Main_remote.py /data/knit K01 01 v1 --scalar --scalarstats --scalarevent
```

---

## Data availability

All raw and processed experimental data are available under CC-BY licence at:

**OSF:** [https://osf.io/nf847/](https://osf.io/nf847/) — KnitQuakesForecast Project (DOI: 10.17605/OSF.IO/NF847)

This repository is plug-and-play with the above dataset.

---

## Citing

If you use this code or the associated dataset, please cite:

> Adèle Douin, Samuel Poincloux, Jean-Philippe Bruneton & Frédéric Lechenault,
> **"Assessing seismic-like events prediction in model knits with unsupervised machine learning"**,
> *Extreme Mechanics Letters* 58 (2023) 101932.
> [https://doi.org/10.1016/j.eml.2022.101932](https://doi.org/10.1016/j.eml.2022.101932)

```bibtex
@article{douin2023knitcity,
  title   = {Assessing seismic-like events prediction in model knits
             with unsupervised machine learning},
  author  = {Douin, Ad\`ele and Poincloux, Samuel and Bruneton, Jean-Philippe
             and Lechenault, Fr\'ed\'eric},
  journal = {Extreme Mechanics Letters},
  volume  = {58},
  pages   = {101932},
  year    = {2023},
  doi     = {10.1016/j.eml.2022.101932}
}
```

---

## License

This code is released under the **MIT License**.
The associated dataset is released under **CC-BY 4.0**.
