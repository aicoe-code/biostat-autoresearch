# Biostat AutoResearch

Autonomous optimisation of statistical analysis methods for clinical trials using AI-driven experimentation loops.

An AI agent iteratively modifies a statistical analysis function for a simulated Phase III trial, evaluates operating characteristics via Monte Carlo simulation (power, type I error, bias, coverage), and retains or discards changes based on pre-specified acceptance criteria — running 50+ experiments autonomously.

## The Problem

Biostatisticians designing clinical trials face a combinatorial design space: choice of analysis model, covariate adjustment, missing data handling, variance estimation, and multiplicity correction. In practice, 3-10 approaches are manually evaluated over days or weeks.

This framework automates that exploration. The biostatistician defines constraints in `protocol.md`, the agent explores methods overnight, and the results are reviewed the next morning with full simulation evidence.

## How It Works

```
modify analysis.py → simulate 10,000 trials → evaluate power + type I error → accept/reject → repeat
```

| File | Purpose | Who Edits |
|------|---------|-----------|
| `prepare.py` | Trial data simulator + evaluation harness (fixed) | Nobody |
| `analysis.py` | Statistical analysis method | AI agent |
| `protocol.md` | Trial constraints + agent instructions | Human |
| `run_experiment.py` | Experiment runner | Nobody |

**Hard constraint:** Type I error must stay at or below 0.025 (one-sided). Any method that inflates alpha is auto-rejected, regardless of power gain. This mirrors regulatory reality.

## Results

Starting from a baseline Welch t-test (power 0.806), the agent explored 11 methods and identified ANCOVA with full covariate adjustment as optimal:

| Method | Power | Type I Error | Status |
|--------|-------|-------------|--------|
| Welch t-test (baseline) | 0.806 | 0.020 | keep |
| ANCOVA: baseline severity | 0.839 | 0.023 | keep |
| **ANCOVA: severity + age + site** | **0.986** | **0.021** | **keep** |
| ANCOVA + HC3 robust SE | 0.985 | 0.020 | discard (no improvement) |
| ANCOVA + mean imputation | 0.985 | 0.082 | discard (alpha inflated) |
| ANCOVA + regression imputation | 0.994 | 0.060 | discard (alpha inflated) |

Three imputation strategies were correctly rejected for inflating type I error — a result that validates the hard constraint mechanism.

See `paper.md` for the full writeup including assumptions, method details, and extension roadmap.

## Quick Start

**Requirements:** Python 3.10+. No GPU needed.

```bash
pip install -e ".[dev]"

# Run baseline
python run_experiment.py

# Quick test (fewer sims)
python run_experiment.py --n-sims 500 --seed 42

# Run tests
python -m pytest tests/ -v
```

## Running the Agent

Point an AI agent at `protocol.md`:

```
Read protocol.md and let's kick off a new experiment.
```

The agent will create a branch, establish a baseline, then loop autonomously — modifying `analysis.py`, running simulations, and keeping or discarding changes.

## Extending

The framework is designed for extension to:
- Time-to-event endpoints (survival analysis)
- Binary endpoints (responder analysis)
- Repeated measures (MMRM, GEE)
- Adaptive designs (interim analyses, sample size re-estimation)
- Multi-scenario robustness (minimax across data generating processes)

See Section 7 of `paper.md` for the full extension roadmap.

## Pattern Origin

This applies the autonomous experimentation loop from [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) — originally designed for LLM training optimisation — to clinical trial biostatistics. See `karpathy-autoresearch.md` for the mapping between domains.

## License

MIT
