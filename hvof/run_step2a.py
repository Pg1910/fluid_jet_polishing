"""
step 2: --- does the input parametrization matters?


the paper feeds three raw inputs to every model : q(ch4),q(o2),SOD
we test whether combinging the two flow rates into physically meaningfull
quantities predicts better , using the identical harness from step 1.

The two derived quantities :

o2_ch4_ratio = Q(o2)/Q(ch4)
    stoichiometric methane combustion is ch4 +2 02 -> co2 +2H2o, so a ratio of 2.0 is complete combustion and anything below it is fuel-rich. Flame temperature is governed by
    far the mizxture sits from stoichiometry , NOT by the absolute flow rates . A DOE study of HVOF pyrometry (OSTI 656793, 1995) found particle temperature is "primarlya function of stoichiometry"
    flow rates . A DOE study of HVOF pyrometry {OSTI 656793,1995} found particle temperature is "primarily a function of stoichiometry".

total_flow = Q(ch4) + Q(o2)
    A proxy for combustion chamber pressure . The same DOE study found particle velocity is "determined primarily by chamber pressure,while stoichiometry has a minor influence".

so the literatur make a falsifiable , *split* prediction :
    velocity should prefer total_flow
    temperature  should prefer o2_ch4_ratio
A feature set that helped both targets equally would be weaker evidence -- it would suggest we just added useful degrees of freedom
rather than captured the right physics.

Caution on interpretation
-------------------------
These feature sets were chosen after inspecting all  40 rows . Cross -validation does not protect against that kind of selection . The result below
is threfore *consistent* the mechanism , not independant confirmation of it . Report it as such

"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

import data 
from evaluate import benchmark
ALPHAS = np.logspace(-3, 3, 25)

MODELS = {
    "ridge": make_pipeline(StandardScaler (), RidgeCV(alphas =ALPHAS)),
    "ridge+poly2": make_pipeline(
        StandardScaler (), PolynomialFeatures(degree = 2, include_bias = False), RidgeCV(alphas = ALPHAS)
    ),
}

FEATURE_SETS = {
    "A paper (CH4,O2,SOD)": ["ch4_slpm", "o2_slpm", "sod_mm"],
    "B ratio (O2/CH4,SOD)": ["o2_ch4_ratio", "sod_mm"],
    "C flow  (total,SOD)": ["total_flow", "sod_mm"],
    "D ratio+flow (+SOD)": ["o2_ch4_ratio", "total_flow", "sod_mm"],
}

def main():
    df = data.load()
    groups = df["sod_mm"].values

    rows = []
    for target in data.INTERMEDIATE:
        y = df[target].values
        for fs_name , cols in FEATURE_SETS.items():
            X = df[cols].values
            t = benchmark(MODELS,X,y , groups=groups , target_name =    target)
            t.insert(1, "features",fs_name)
            t["n_features"] = len(cols)
            rows.append(t)

    out = pd.concat(rows, ignore_index = True)
    pd.set_option("display.width", 220)
    keep = ["features", "n_features", "kfold_R2", "kfold_R2_sd","kfold_MAE","groupSOD_R2","groupSOD_MAE"]

for target in data.INTERMEDIATE:
    for model in MODELS:
        sub = out[(out.target == target) & (out.model == model)]
        print(f"\n{'='*84}")
        print(f" {data.LABELS[target]}  |  model:{model}")
        print("="*84)
        print(sub[keep].set_index("features").round(3).to_string())

    outdir = ROOT / "results"
    outdir.mkdir(exists_ok = True)
    out.to_csv(outdir / "step2_features.csv", index = False)
    print(f"\nsaved -> {outdir / 'step2_features.csv'}")


if __name__ =="__main__":
    main()