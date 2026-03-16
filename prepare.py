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


# ---------------------------------------------------------------------------
# Evaluation Harness (DO NOT MODIFY)
# ---------------------------------------------------------------------------

def evaluate_power(analyze_fn: callable, n_sims: int = N_SIMS,
                   true_effect: float = TRUE_EFFECT, n_per_arm: int = N_PER_ARM,
                   alpha: float = ALPHA, time_budget: int = TIME_BUDGET,
                   seed: int | None = None) -> dict:
    base_rng = np.random.default_rng(seed)
    t_start = time.time()
    timed_out = False

    power_rejects = 0
    estimates = []
    coverage_hits = 0
    power_sims_done = 0

    for i in range(n_sims):
        if time.time() - t_start > time_budget:
            timed_out = True
            break
        trial_seed = int(base_rng.integers(0, 2**31))
        data = generate_trial(n_per_arm=n_per_arm, true_effect=true_effect, seed=trial_seed)
        try:
            result = analyze_fn(data)
            p_val = result["p_value"]
            est = result["estimate"]
            ci_lo = result["ci_lower"]
            ci_hi = result["ci_upper"]
            if p_val < alpha:
                power_rejects += 1
            estimates.append(est)
            if ci_lo <= true_effect <= ci_hi:
                coverage_hits += 1
        except Exception:
            pass
        power_sims_done += 1

    t1e_rejects = 0
    t1e_sims_done = 0

    if not timed_out:
        for i in range(n_sims):
            if time.time() - t_start > time_budget:
                timed_out = True
                break
            trial_seed = int(base_rng.integers(0, 2**31))
            data = generate_trial(n_per_arm=n_per_arm, true_effect=0.0, seed=trial_seed)
            try:
                result = analyze_fn(data)
                if result["p_value"] < alpha:
                    t1e_rejects += 1
            except Exception:
                pass
            t1e_sims_done += 1

    power = power_rejects / max(power_sims_done, 1)
    type1_error = t1e_rejects / max(t1e_sims_done, 1) if t1e_sims_done > 0 else float("nan")
    mean_est = float(np.mean(estimates)) if estimates else float("nan")
    bias = mean_est - true_effect if estimates else float("nan")
    coverage = coverage_hits / max(power_sims_done, 1)
    actual_sims = power_sims_done

    return {
        "power": power,
        "type1_error": type1_error,
        "bias": bias,
        "coverage": coverage,
        "mean_estimate": mean_est,
        "n_sims": actual_sims,
    }
