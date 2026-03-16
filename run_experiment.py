"""
Biostat AutoResearch: experiment runner.
Calls evaluate_power with the current analysis method and prints results.

Usage:
    python run_experiment.py [--n-sims 10000] [--time-budget 120] [--seed 42]
"""

import argparse
import time

from prepare import evaluate_power
from analysis import analyze_trial


def main():
    parser = argparse.ArgumentParser(description="Run biostat autoresearch experiment")
    parser.add_argument("--n-sims", type=int, default=10000, help="Number of simulations per batch")
    parser.add_argument("--time-budget", type=int, default=120, help="Soft time limit in seconds")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    t_start = time.time()

    results = evaluate_power(
        analyze_fn=analyze_trial,
        n_sims=args.n_sims,
        time_budget=args.time_budget,
        seed=args.seed,
    )

    t_elapsed = time.time() - t_start

    print("---")
    print(f"power:            {results['power']:.4f}")
    print(f"type1_error:      {results['type1_error']:.4f}")
    print(f"bias:             {results['bias']:.3f}")
    print(f"coverage:         {results['coverage']:.3f}")
    print(f"mean_estimate:    {results['mean_estimate']:.3f}")
    print(f"n_sims:           {results['n_sims']}")
    print(f"time_seconds:     {t_elapsed:.1f}")


if __name__ == "__main__":
    main()
