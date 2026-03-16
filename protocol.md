# biostat-autoresearch

This is an experiment to have an AI agent autonomously optimize clinical trial statistical analysis methods.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar16`). The branch `biostat-autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b biostat-autoresearch/<tag>` from current main.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data simulator, evaluation harness. Do not modify.
   - `analysis.py` — the file you modify. Statistical analysis method.
4. **Initialize results.tsv**: Create `results.tsv` with just the header row. Leave it untracked by git — do NOT commit it.
5. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Trial Context

- **Phase III superiority trial**, two-arm (treatment vs placebo)
- **Continuous primary endpoint**: change from baseline
- **One-sided alpha**: 0.025
- **ITT population**: all randomized patients
- **N**: 200 per arm (400 total)
- **Dropout**: ~15% MAR (encoded as NaN in outcome)

## Experimentation

Each experiment runs a simulation-based evaluation. You launch it as: `python run_experiment.py`.

**What you CAN do:**
- Modify `analysis.py` — this is the only file you edit. Everything is fair game: statistical model, covariate adjustment, missing data handling, variance estimation, endpoint transformation.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed data simulator and evaluation harness.
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_power` function in `prepare.py` is the ground truth metric.

**The goal: maximize power while keeping type I error ≤ 0.025.**

Type I error is a **hard constraint**. Any method that inflates type I error above 0.025 is auto-rejected, regardless of power gain. This mirrors regulatory reality — no regulator accepts inflated alpha.

**The first run**: Your very first run should always be to establish the baseline, so you will run the experiment as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
power:            0.8120
type1_error:      0.0248
bias:             -0.002
coverage:         0.951
mean_estimate:    -0.498
n_sims:           10000
time_seconds:     45.2
```

You can extract the key metrics from the log file:

```
grep "^power:\|^type1_error:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated).

The TSV has a header row and 7 columns:

```
commit	power	type1_error	bias	coverage	status	description
```

1. git commit hash (short, 7 chars)
2. power achieved (e.g. 0.8120)
3. type1_error (e.g. 0.0248) — use 0.0000 for crashes
4. bias (e.g. -0.002) — use 0.000 for crashes
5. coverage (e.g. 0.951) — use 0.000 for crashes
6. status: `keep`, `discard`, or `crash`
7. short text description of what this experiment tried

Example:

```
commit	power	type1_error	bias	coverage	status	description
a1b2c3d	0.8120	0.0248	-0.002	0.951	keep	baseline: two-sample t-test
b2c3d4e	0.8560	0.0243	-0.001	0.948	keep	ANCOVA adjusting for baseline severity
c3d4e5f	0.7800	0.0310	-0.003	0.940	discard	bootstrap CI (type I error inflated)
d4e5f6g	0.0000	0.0000	0.000	0.000	crash	multiple imputation (import error)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `biostat-autoresearch/mar16`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `analysis.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `python run_experiment.py > run.log 2>&1`
5. Read out the results: `grep "^power:\|^type1_error:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If power improved AND type1_error ≤ 0.025, you "advance" the branch, keeping the git commit
9. If power is equal or worse, OR type1_error > 0.025, you git reset back: `git reset --hard HEAD~1`

**Simplicity criterion**: All else being equal, simpler is better. A small power improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better power is a great outcome. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude.

**Timeout**: Each experiment should take ~2 minutes. If a run exceeds 5 minutes, kill it and treat it as a failure.

**Crashes**: If a run crashes, use your judgment: if it's a typo or missing import, fix and re-run. If the idea is fundamentally broken, skip it, log "crash", and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — re-read the in-scope files for new angles, try combining previous near-misses, try more radical analysis approaches. The loop runs until the human interrupts you, period.

## Exploration suggestions

Here are some directions to explore (not exhaustive):

- **Covariate adjustment**: ANCOVA adjusting for baseline severity, age, site
- **Missing data**: LOCF, multiple imputation, inverse probability weighting
- **Model-based**: Mixed models, GEE, Bayesian approaches
- **Robust methods**: Rank-based tests, permutation tests, sandwich variance estimators
- **Combinations**: ANCOVA + multiple imputation, mixed model + robust SE
- **Transformations**: Rank-based endpoints, log transformation
