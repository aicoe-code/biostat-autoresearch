"""
Biostat AutoResearch: data preparation and evaluation harness.
DO NOT MODIFY — this file is read-only for the AI agent.

Generates simulated Phase III clinical trial data and provides
a fixed evaluation harness for comparing statistical analysis methods.
"""

import time
import math
import numpy as np


# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

N_PER_ARM = 200
TRUE_EFFECT = -0.5
NOISE_SD = 1.0
ALPHA = 0.025
N_SIMS = 10000
TIME_BUDGET = 120

SITE_EFFECTS = np.array([0.0, 0.1, -0.1, 0.05])


# ---------------------------------------------------------------------------
# Data Generation
# ---------------------------------------------------------------------------

def generate_trial(n_per_arm: int = N_PER_ARM, true_effect: float = TRUE_EFFECT,
                   seed: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    n_total = 2 * n_per_arm

    arm = np.array([0] * n_per_arm + [1] * n_per_arm)
    age = rng.normal(55, 10, n_total)
    sex = rng.binomial(1, 0.5, n_total)
    baseline_severity = rng.normal(5, 1.5, n_total)
    site = rng.integers(0, 4, n_total)

    noise = rng.normal(0, NOISE_SD, n_total)
    outcome = (2.0
               + true_effect * arm
               + 0.3 * baseline_severity
               + 0.1 * (age - 55)
               + SITE_EFFECTS[site]
               + noise)

    logit_dropout = -3.0 + 0.4 * baseline_severity
    p_dropout = 1.0 / (1.0 + np.exp(-logit_dropout))
    is_missing = rng.binomial(1, p_dropout, n_total).astype(bool)
    outcome = outcome.astype(float)
    outcome[is_missing] = np.nan

    return {
        "arm": arm,
        "outcome": outcome,
        "age": age,
        "sex": sex,
        "baseline_severity": baseline_severity,
        "site": site,
    }
