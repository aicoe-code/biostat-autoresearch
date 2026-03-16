"""
Biostat AutoResearch: statistical analysis method.
THIS IS THE FILE THE AI AGENT MODIFIES.

Baseline: two-sample t-test on complete cases.
The agent can replace this with any analysis method as long as
analyze_trial() returns {"p_value", "estimate", "ci_lower", "ci_upper"}.
"""

import numpy as np
from scipy import stats


def analyze_trial(data: dict) -> dict:
    arm = data["arm"]
    outcome = data["outcome"]

    mask = ~np.isnan(outcome)
    trt = outcome[mask & (arm == 1)]
    ctl = outcome[mask & (arm == 0)]

    t_stat, p_two = stats.ttest_ind(trt, ctl, equal_var=False)
    p_value = p_two / 2 if t_stat < 0 else 1.0 - p_two / 2

    estimate = trt.mean() - ctl.mean()

    se = np.sqrt(np.var(trt, ddof=1) / len(trt) + np.var(ctl, ddof=1) / len(ctl))
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se

    return {
        "p_value": p_value,
        "estimate": estimate,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }
