"""
Biostat AutoResearch: statistical analysis method.
THIS IS THE FILE THE AI AGENT MODIFIES.

ANCOVA: OLS regression adjusting for baseline severity, age, and site.
"""

import numpy as np
from scipy import stats
import statsmodels.api as sm


def analyze_trial(data: dict) -> dict:
    arm = data["arm"]
    outcome = data["outcome"]
    baseline_severity = data["baseline_severity"]
    age = data["age"]
    site = data["site"]

    # Complete case analysis
    mask = ~np.isnan(outcome)
    y = outcome[mask]
    x_arm = arm[mask]
    x_bl = baseline_severity[mask]
    x_age = age[mask]
    x_site = site[mask]

    # Site dummies (site 0 as reference)
    site_d1 = (x_site == 1).astype(float)
    site_d2 = (x_site == 2).astype(float)
    site_d3 = (x_site == 3).astype(float)

    # ANCOVA: outcome ~ arm + baseline_severity + age + site dummies
    X = np.column_stack([np.ones(len(y)), x_arm, x_bl, x_age, site_d1, site_d2, site_d3])
    model = sm.OLS(y, X).fit()

    # Treatment effect is coefficient of arm (index 1)
    estimate = model.params[1]
    se = model.bse[1]
    t_stat = estimate / se
    df = model.df_resid

    # One-sided p-value
    p_value = stats.t.cdf(t_stat, df)

    # 95% CI
    t_crit = stats.t.ppf(0.975, df)
    ci_lower = estimate - t_crit * se
    ci_upper = estimate + t_crit * se

    return {
        "p_value": p_value,
        "estimate": estimate,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }
