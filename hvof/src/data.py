"""
Data loading for the HVOF NiCr-Cr3C2 dataset.

Source: Gui et al., Materials 2023, 16, 6279, Table 2.
40 spraying experiments. Transcribed by hand -- verify against the PDF.

Column meanings
---------------
Process parameters (things the operator sets):
    ch4_slpm   fuel flow rate,     standard litres per minute
    o2_slpm    oxygen flow rate,   standard litres per minute
    sod_mm     stand-off distance, mm (gun-to-substrate)

Intermediate variables (measured in flight by AccuraSpray G3):
    velocity_ms      ensemble-average particle velocity at impact, m/s
    temperature_K    ensemble-average particle temperature at impact, K

Coating properties (measured after deposition):
    microhardness_HV03      Vickers, 300 gf load
    wear_rate_e5_mm3_Nm     dry sliding wear rate, x 1e-5 mm^3/N/m
    porosity_pct            cross-sectional porosity, % area
"""

from pathlib import Path
import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "hvof_nicr_cr3c2.csv"

# Column groups, so we never hard-code names in analysis scripts.
PROCESS_PARAMS = ["ch4_slpm", "o2_slpm", "sod_mm"]
INTERMEDIATE = ["velocity_ms", "temperature_K"]
COATING_PROPS = ["microhardness_HV03", "wear_rate_e5_mm3_Nm", "porosity_pct"]

# Nice labels for plots and tables.
LABELS = {
    "velocity_ms": "particle velocity (m/s)",
    "temperature_K": "particle temperature (K)",
    "microhardness_HV03": "microhardness (HV0.3)",
    "wear_rate_e5_mm3_Nm": "wear rate (1e-5 mm3/N/m)",
    "porosity_pct": "porosity (% area)",
}


def load(add_features: bool = True) -> pd.DataFrame:
    """Load the dataset, optionally adding physically motivated features.

    Parameters
    ----------
    add_features : bool
        If True, append the oxygen-to-fuel ratio and equivalence ratio.
        These are derived, not new information -- but the *form* in which
        you present information to a model matters enormously when n=40.

    Notes
    -----
    Methane combustion:  CH4 + 2 O2 -> CO2 + 2 H2O
    so the stoichiometric oxygen-to-fuel ratio is 2.0.

    The equivalence ratio phi is defined as
        phi = (fuel/oxidiser)_actual / (fuel/oxidiser)_stoichiometric
    Rearranged in terms of the O2/CH4 ratio this is  phi = 2.0 / ratio.
        phi > 1  fuel-rich   (excess fuel, incomplete combustion)
        phi = 1  stoichiometric
        phi < 1  fuel-lean   (excess oxygen)
    """
    df = pd.read_csv(DATA_PATH)

    if add_features:
        df["o2_ch4_ratio"] = df["o2_slpm"] / df["ch4_slpm"]
        df["phi"] = 2.0 / df["o2_ch4_ratio"]
        # Total reactant flow is a proxy for chamber pressure, which the
        # literature identifies as the main driver of particle velocity.
        df["total_flow"] = df["o2_slpm"] + df["ch4_slpm"]

    return df


def describe(df: pd.DataFrame) -> None:
    """Print a compact summary of the design and the response ranges."""
    print(f"n = {len(df)} experiments, {df.shape[1]} columns\n")

    print("Design coverage (rows = fuel flow, cols = stand-off distance):")
    print(pd.crosstab(df.ch4_slpm, df.sod_mm), "\n")

    print("Response ranges:")
    for c in INTERMEDIATE + COATING_PROPS:
        print(f"  {LABELS[c]:<32} {df[c].min():8.3f} to {df[c].max():8.3f}"
              f"   mean {df[c].mean():8.3f}")

    if "phi" in df:
        print(f"\nEquivalence ratio phi: {df.phi.min():.2f} to {df.phi.max():.2f}")
        rich = (df.phi > 1).sum()
        print(f"  fuel-rich conditions (phi > 1): {rich} of {len(df)}")


if __name__ == "__main__":
    describe(load())