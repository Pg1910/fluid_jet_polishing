# HVOF NiCr-Cr3C2 -- honest re-analysis

Re-analysis of Gui et al., *Materials* 2023, 16, 6279
(DOI: 10.3390/ma16186279). No data or code was released with that paper;
the dataset here is transcribed by hand from its Table 2.

## What this project is testing

1. Do the paper's reported accuracies survive honest cross-validation?
2. Is the input parameterization (raw O2 and CH4 flows) the right one,
   or does the oxygen-to-fuel ratio carry the thermal physics?
3. Is Eq. 5 (convective heating) consistent with the measured particle
   temperatures? (Preliminary answer: no -- temperature *rises* with
   stand-off distance, which convective cooling cannot produce.)

## Layout

```
hvof/
├── data/
│   └── hvof_nicr_cr3c2.csv     transcribed Table 2, n=40
├── src/
│   ├── data.py                 loading + derived features
│   └── evaluate.py             the evaluation harness
├── results/                    generated; safe to delete
├── run_step1.py                baselines
└── requirements.txt
```

Directory names matter: `data.py` locates the CSV relative to its own
file, so `src/` and `data/` must be siblings.

## Setup

```bash
cd hvof
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run order

```bash
python3 src/data.py      # 1. sanity-check the transcription
python3 run_step1.py     # 2. baselines under three protocols
```

Both scripts work from any working directory.

## Before trusting any of this

Spot-check the CSV against Table 2 in the PDF. A transcription error will
not raise an exception -- it will quietly change your conclusions. Check
at minimum rows 1, 20, and 40, and confirm the column order is
`CH4, SOD, O2` (the paper's table puts SOD in the middle, which is easy
to get wrong).

## Conventions

- Every model is an sklearn `Pipeline`, so scaling is refitted inside each
  CV fold and cannot leak.
- In-sample R2 is reported only next to out-of-fold R2, as an overfitting
  diagnostic. It is never quoted alone.
- `groupSOD_R2` (leave-one-stand-off-distance-out) is the headline metric.
  k-fold answers "predict a repeat run"; grouped answers "predict an
  untested process condition", which is what matters for process design.