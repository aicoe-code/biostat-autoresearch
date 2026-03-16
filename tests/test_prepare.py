import numpy as np
from prepare import generate_trial, evaluate_power

def test_generate_trial_returns_required_keys():
    data = generate_trial(n_per_arm=50, true_effect=-0.5, seed=42)
    required = {"arm", "outcome", "age", "sex", "baseline_severity", "site"}
    assert required.issubset(set(data.keys()))

def test_generate_trial_correct_sample_size():
    data = generate_trial(n_per_arm=100, true_effect=-0.5, seed=42)
    assert len(data["arm"]) == 200
    assert np.sum(data["arm"] == 0) == 100
    assert np.sum(data["arm"] == 1) == 100

def test_generate_trial_has_dropout():
    data = generate_trial(n_per_arm=1000, true_effect=-0.5, seed=42)
    missing_rate = np.isnan(data["outcome"]).mean()
    assert 0.05 < missing_rate < 0.30

def test_generate_trial_treatment_effect():
    data = generate_trial(n_per_arm=10000, true_effect=-0.5, seed=42)
    observed = data["outcome"]
    arm = data["arm"]
    mask = ~np.isnan(observed)
    diff = observed[mask & (arm == 1)].mean() - observed[mask & (arm == 0)].mean()
    assert abs(diff - (-0.5)) < 0.1

def test_generate_trial_null_effect():
    data = generate_trial(n_per_arm=10000, true_effect=0.0, seed=42)
    observed = data["outcome"]
    arm = data["arm"]
    mask = ~np.isnan(observed)
    diff = observed[mask & (arm == 1)].mean() - observed[mask & (arm == 0)].mean()
    assert abs(diff) < 0.1

def test_generate_trial_reproducible():
    d1 = generate_trial(n_per_arm=50, true_effect=-0.5, seed=123)
    d2 = generate_trial(n_per_arm=50, true_effect=-0.5, seed=123)
    np.testing.assert_array_equal(d1["arm"], d2["arm"])
    np.testing.assert_array_equal(d1["age"], d2["age"])

def test_generate_trial_covariate_ranges():
    data = generate_trial(n_per_arm=1000, true_effect=-0.5, seed=42)
    assert 45 < data["age"].mean() < 65
    assert set(np.unique(data["sex"])) == {0, 1}
    assert set(np.unique(data["site"])).issubset({0, 1, 2, 3})


def _dummy_analyze(data):
    arm = data["arm"]
    outcome = data["outcome"]
    mask = ~np.isnan(outcome)
    trt = outcome[mask & (arm == 1)]
    ctl = outcome[mask & (arm == 0)]
    diff = trt.mean() - ctl.mean()
    from scipy import stats
    t_stat, p_two = stats.ttest_ind(trt, ctl, equal_var=False)
    p_val = p_two / 2 if t_stat < 0 else 1 - p_two / 2
    se = np.sqrt(np.var(trt, ddof=1) / len(trt) + np.var(ctl, ddof=1) / len(ctl))
    return {
        "p_value": p_val,
        "estimate": diff,
        "ci_lower": diff - 1.96 * se,
        "ci_upper": diff + 1.96 * se,
    }


def test_evaluate_power_returns_required_keys():
    result = evaluate_power(_dummy_analyze, n_sims=100, seed=42)
    required = {"power", "type1_error", "bias", "coverage", "mean_estimate", "n_sims"}
    assert set(result.keys()) == required


def test_evaluate_power_n_sims_matches():
    result = evaluate_power(_dummy_analyze, n_sims=200, seed=42)
    assert result["n_sims"] == 200


def test_evaluate_power_metrics_reasonable():
    result = evaluate_power(_dummy_analyze, n_sims=500, seed=42)
    assert 0.5 < result["power"] < 1.0
    assert 0.0 <= result["type1_error"] < 0.06
    assert abs(result["bias"]) < 0.2
    assert 0.8 < result["coverage"] < 1.0
