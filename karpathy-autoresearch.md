# AutoResearch for Biostatistics: Autonomous Clinical Trial Design Optimization

**Inspired by:** [Karpathy's autoresearch](https://github.com/karpathy/autoresearch)

## The Pattern

```
protocol.md (trial constraints) → modify analysis.R → simulate → eval operating characteristics → accept/reject → repeat
```

## Direct Mapping

| Autoresearch | Clinical Trial Equivalent |
|---|---|
| `train.py` (agent modifies) | `analysis.py` — statistical model, endpoint handling, missing data strategy, multiplicity adjustments |
| `prepare.py` (fixed) | Data generation engine — fixed patient population simulator, enrollment model, dropout patterns |
| `program.md` (human guides) | Protocol constraints — regulatory requirements, therapeutic area context, acceptable type I error, clinically meaningful difference |
| `val_bpb` (single metric) | Power at target effect size while maintaining type I error control (or composite: power + bias + coverage) |
| 5-min time budget | Fixed number of simulation replicates (e.g., 10,000 per run) |

## Use Cases

### 1. Statistical Analysis Plan Optimization
Agent iterates over choices a biostatistician normally makes manually:
- MMRM vs ANCOVA vs Bayesian longitudinal model
- Covariate selection strategies
- Missing data handling (MNAR, MAR, LOCF, multiple imputation, pattern-mixture)
- Transformation of endpoints
- Accept/reject based on: power, bias, coverage probability, type I error

### 2. Adaptive Design Exploration
- Modify interim analysis decision rules, sample size re-estimation boundaries, futility thresholds
- Simulate full trial 10,000 times
- Evaluate: overall power, expected sample size, probability of early stopping

### 3. Multiplicity Strategy Optimization
- Multiple endpoints, multiple doses, subgroups — combinatorial explosion
- Agent explores: Hochberg, Holm, graphical approaches, gatekeeping, alpha recycling
- Metric: family-wise power across all clinically relevant scenarios

### 4. Sensitivity Analysis Automation
- Systematically vary dropout mechanisms (MCAR → MNAR)
- Test tipping-point analyses
- Explore different estimands
- Report which assumptions break the result

## The Shift

**Today:** Biostatistician manually tries 5-10 approaches over weeks, picks one, defends it in the SAP.

**With this pattern:** Biostatistician defines constraints in `protocol.md`, agent explores 100+ approaches overnight, biostatistician reviews top performers and selects with full simulation evidence.

The human's role moves from doing the simulations to defining the constraints and judging the results.
