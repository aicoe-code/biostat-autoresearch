# Biostat AutoResearch

Autonomous experimentation loop for clinical trial biostatistics, inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch).

An AI agent iteratively modifies a statistical analysis method for a simulated Phase III clinical trial, runs power simulations, and keeps or discards changes — exploring 50+ analysis strategies overnight while you sleep.

## Setup

```bash
pip install -e ".[dev]"
```

## Quick Start

```bash
# Run the current analysis method
python run_experiment.py

# Run with fewer sims for quick testing
python run_experiment.py --n-sims 500 --seed 42
```

## Using with an AI Agent

Point your AI agent at `protocol.md` and say: "Let's kick off a new experiment."

The agent will:
1. Create a branch, read all files, run baseline
2. Modify `analysis.py` with a new statistical approach
3. Run simulations, evaluate power and type I error
4. Keep improvements, discard regressions
5. Repeat autonomously

## Files

| File | Purpose | Who Edits |
|------|---------|-----------|
| `prepare.py` | Trial data simulator + evaluation harness | Nobody (fixed) |
| `analysis.py` | Statistical analysis method | AI agent |
| `protocol.md` | Agent instructions | Human |
| `run_experiment.py` | Experiment runner | Nobody |

## No GPU Required

Runs on any laptop — pure numpy/scipy simulations.
