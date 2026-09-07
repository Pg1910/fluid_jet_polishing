"""
step 2b -- completing step 2.

step 2a produced a table of scores . A table of scores is not a conclusion .
Two things are still missing:

    1. A valid comparison . Eyeballing overlapping error bars is the wrong test , because those error bars are dominnated by forld difficulty that affectd both models identically .w e use a paired comparison on identical folds, with the nadeau -bengio variance correction.
    2. interpretation . A model that predicts well but whose coffecients we habe never looked at is still a black box . if the ratio feature really encodes stoichiometry , its coeffecient should be positive for temperature (closer to stoichiometric -> hotter flame ) and weak for velocity. checking this is how we tell a physical effect from a lucky fit"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"src"))

from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


import data
from evaluate import paired_compare

ALPHAS = np.logspace(-3,3,25)
ridge = lambda: make_pipeline(StandardScaler(), RidgeCV(alphas = ALPHAS))

SETS = {
    "A paper (CH4,O2,SOD)": ["ch4_slpm", "o2_slpm", "sod_mm"],
    "B ratio (O2/CH4,SOD)": ["o2_ch4_ratio", "sod_mm"],
    "C flow  (total,SOD)": ["total_flow", "sod_mm"],
}
 
# Each comparison is stated as a directional hypothesis BEFORE running it.
COMPARISONS = [
    ("velocity_ms", "B ratio (o2/ch4/SOD)", "C flow (total,SOD)",
     "flow should beat ratio for velocity (chamber pressure drives v)"),
     ("temperature_k","C flow (total,SOD)","B ratio (o2/CH4/SOD)",
      "ratio should beat flow for temperature (stoichiometry drives T)"),
      ("temperature_k","A paper (CH4,o2,SOD)","B ratio (o2/CH4/SOD)",
       "does ratio beat the papers 3 raw inputs , using one fewer feature?"),
       ("velocity_ms", "A paper (CH4,o2,SOD)", "C flow (total,SOD)",
       "does total flow beat the paper's 3 raw inputs for velocity?"),
]

def significance_tests(df):
    print("="*88)
    print(" Paired comparisons (b minus a , positive => b is better)")
    print("="*88)
    rows = []
    for target , a_name , b_name , hypothesis in COMPARISONS:
        y = df[target].values
        Xa = df[SETS[a_name]].values
        Xb = df[SETS[b_name]].values
        r = paired_compare(ridge(), Xa , ridge(),Xb,y)

        print(f"\n{data.LABELS[target]}")
        print(f" hypothesis : {hypothesis}")
        print(f" A = {a_name}")
        print(f" B = {b_name}")
        print(f" mean dR2  = {r['mean_diff']:+.4f}")
        print(f" B wins in  = {r['win_rate']*100:.0f}% of the 20 repeats")
        print(f" SE naive  = {r['se_naive']:.4f}  (Nadeau-bengio)")
        print(f" t = {r['t_corrected']:+.2f}     p = {r['p_corrected']:.3f}"
              f"   {'SIGNIFICANT' if r['p_corrected']<0.05 else 'not significant'}")
        rows.append({"target": target, "A":a_name, "B": b_name, **r})
    return pd.DataFrame(rows)

def coeffecients(df):
    """Fit on all data and inspect standardised coeffecients.
    standardised (because standardised runs first) means coefficients.
    are directly comparable in magnitude: each is the change in the  target per one standard deviation of that input.
    
    NOTE: fitting on all 40 rows is fine here . we are not estimating predictive accuracy -- we already did that honestly . we are reading off the direction and relative size of effects."""

    print("\n"+ "="*88)
    print("  STANDARDISED RIDGE COEFFECIENTS  (fit on all n=40 , for reading only)")
    print("="*88)
    for target in data.INTERMEDIATE:
        print(f"\n{data.LABELS[target]}")
        for set_name , cols in SETS.items():
            m = ridge().fit(df[cols].values, df[target].values)
            coefs = m[-1].coef_
            terms = "  ".join(f"{c}={v:+8.1f}" for c , v in zip(cols, coefs))
            print(f" {set_name:<24} {terms}")


def main():
    df = data.load()
    res = significance_tests(df)
    coeffecients(df)

    outdir = ROOT/"results"
    outdir.mkdir(exist_ok = True)
    res.to_csv(outdir / "step2b_significance.csv", index = False)
    print(f"\nsaved -> {outdir / 'step2b_significance.csv'}")



if __name__ == "__main__":
    main()
    





































