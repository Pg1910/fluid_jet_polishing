"""
step1 -- establish the honest baseline.

we are NOT trying to build a good model yet . we are establishing the floor that every later model must clear,
and demonstrating that the protocol behaves sensibly on models whose behavious we already understand.

Ladded rationale;
   Mean-only   By construction gets out of fold R2 - 0. This is a sanity check on the harness: if it does not, somethign is broken.
   Linear      3 coeffecients. The simplest thing that could work.
   Ridge       Linear with L2 shrinkage, alpha chosen by inner CV. with n=40 and correlated inputs , OLS coeffecients are unstable;
   Ridge+poly2 Adds squares and pairwise interactions.Physics here is nonlinear (drag goes as velocity squared) , so this is the first model with a plausible functional form"""


import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent/"src"))

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures , StandardScaler

import data
from evaluate import benchmark

ALPHAS = np.logspace(-3, 3, 25)

MODELS = {
    "0 mean-only": DummyRegressor(strategy="mean"),
    "1 linear": make_pipeline(StandardScaler(), LinearRegression()),
    "2 ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS)),
    "3 ridge+poly2": make_pipeline(
        StandardScaler(), PolynomialFeatures(2), RidgeCV(alphas=ALPHAS)
    ),
}

def main():
    df = data.load()
    X = df[data.PROCESS_PARAMS].values
    groups = df["sod_mm"].values
    tables = []
    for target in data.INTERMEDIATE:
        y = df[target].values
        tables.append(benchmark(MODELS,X,y, groups = groups, target_name = target))

    out = pd.concat(tables , ignore_index=True)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)
    for target,g in out.groupby("target", sort = False):
        print(f"\n{'='*78}\n {data.LABELS[target]} <- CH4,O2 , SOD\n{'='*78}")
        show = g.drop(columns = ["target"]).set_index('model')
        print(show.round(3).to_string())


    Path("results").mkdir(exist_ok = True)
    out.to_csv("results/step1_baseline.csv", index = False)
    print("\nsaved -> results/step1_baseline.csv")

if __name__ == "__main__":
    main()