import numpy as np
from analysis import analyze_trial
from prepare import generate_trial


def test_analyze_trial_returns_required_keys():
    data = generate_trial(n_per_arm=100, true_effect=-0.5, seed=42)
    result = analyze_trial(data)
    required = {"p_value", "estimate", "ci_lower", "ci_upper"}
    assert set(result.keys()) == required


def test_analyze_trial_p_value_range():
    data = generate_trial(n_per_arm=100, true_effect=-0.5, seed=42)
    result = analyze_trial(data)
    assert 0.0 <= result["p_value"] <= 1.0


def test_analyze_trial_ci_contains_estimate():
    data = generate_trial(n_per_arm=100, true_effect=-0.5, seed=42)
    result = analyze_trial(data)
    assert result["ci_lower"] <= result["estimate"] <= result["ci_upper"]


def test_analyze_trial_handles_missing_data():
    data = generate_trial(n_per_arm=100, true_effect=-0.5, seed=42)
    data["outcome"][0:10] = np.nan
    result = analyze_trial(data)
    assert set(result.keys()) == {"p_value", "estimate", "ci_lower", "ci_upper"}


def test_analyze_trial_detects_effect():
    data = generate_trial(n_per_arm=5000, true_effect=-0.5, seed=42)
    result = analyze_trial(data)
    assert result["p_value"] < 0.025
    assert result["estimate"] < 0


def test_analyze_trial_no_effect():
    data = generate_trial(n_per_arm=100, true_effect=0.0, seed=42)
    result = analyze_trial(data)
    assert 0.0 <= result["p_value"] <= 1.0
