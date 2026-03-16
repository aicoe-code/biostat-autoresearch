import pathlib
import subprocess
import sys

PROJECT_ROOT = pathlib.Path(__file__).parent.parent


def test_runner_completes_and_prints_metrics():
    result = subprocess.run(
        [sys.executable, "run_experiment.py", "--n-sims", "50"],
        capture_output=True, text=True, timeout=120,
        cwd=PROJECT_ROOT,
    )
    assert result.returncode == 0
    output = result.stdout
    assert "power:" in output
    assert "type1_error:" in output
    assert "bias:" in output
    assert "coverage:" in output
    assert "mean_estimate:" in output
    assert "n_sims:" in output
    assert "time_seconds:" in output
    assert "---" in output
