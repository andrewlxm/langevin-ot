# langevin-ot

## `gradient-ascent.py`

- `sample_rho(key, m, d)` draws `m` Gaussian samples in `d` dimensions.
- `compute_T_OT_eps(U, X, epsilon)` computes the entropic OT map from samples `U` to targets `X`.
- `v_theta(t, y, X)` defines the time-dependent velocity field used by the KM flow.
- `compute_T_KM_1_delta(U, X, delta)` solves the ODE for each sample and returns the transported points.
- `objective_fn(X, key, m, epsilon, delta)` measures the mean squared distance between OT and KM outputs, with a Gaussian weight on the samples.
- `update_step_box(...)` and `update_step_norm(...)` apply one optimizer step and then project `X` back to the box or unit-norm constraint.
- `run_optimization_box(...)` and `run_optimization_norm(...)` initialize `X`, run the loop for many steps, and collect the loss history.
- `visualize_results(...)` plots the loss curve, the target points, and the KM/OT cell assignments, then saves the figure in `results/`.

### How the optimization works

1. Start from random 2D target points.
2. Sample Gaussian inputs `U`.
3. Compute both transport maps from `U` to `X`.
4. Measure how far the two maps disagree.
5. Backpropagate through the full pipeline and update `X` with Adam.
6. Re-project `X` to satisfy either the box constraint or the norm constraint.

### Output

Running the script creates plots in `results/` for each seed. Each figure shows:

- the objective history,
- the initial and final target locations,
- the KM cell map,
- the OT cell map.
