import jax
import jax.numpy as jnp
import optax
import diffrax
import equinox as eqx
import matplotlib.pyplot as plt
from ott.geometry import pointcloud
from ott.solvers import linear
from ott.problems.linear import linear_problem

# Sample rho
def sample_rho(key, m, d):
    return jax.random.normal(key, shape=(m, d))

# Computes the Entropic OT map
def compute_T_OT_eps(U, X, epsilon):
    X_jitter = X + 1e-5 * jnp.ones_like(X)
    geom = pointcloud.PointCloud(U, X_jitter, epsilon=epsilon)
    prob_ot = linear_problem.LinearProblem(geom)
    sinkhorn_engine = linear.sinkhorn.Sinkhorn(
        threshold=1e-1,
        max_iterations=500,
        implicit_diff=None
    )
    out_ot = sinkhorn_engine(prob_ot)
    P_mat = out_ot.matrix
    P_row_sum = jnp.sum(P_mat, axis=1, keepdims=True) + 1e-10
    P_norm = P_mat / P_row_sum
    return jnp.dot(P_norm, X)

# Updated Velocity field for the Kim-Milman flow
def v_theta(t, y, X):
    s = t
    n = X.shape[0]
    
    # Calculate the exponent term: -|y - s*x_k|^2 / (2 * (1-s)^2)
    # We use a safety epsilon in the denominator to avoid division by zero as s -> 1
    denom_kernel = 2.0 * jnp.square(jnp.maximum(1.0 - s, 1e-4))
    
    # Compute distances between current position y and scaled targets s*X
    # y shape: (d,), X shape: (n, d)
    diffs = y - s * X  # (n, d)
    dist_sq = jnp.sum(jnp.square(diffs), axis=1)  # (n,)
    
    # Compute weights w_k using log-sum-exp trick for numerical stability
    logits = -dist_sq / denom_kernel
    weights = jax.nn.softmax(logits)  # (n,)
    
    # Compute the weighted sum: sum_k (x_k - s*y) * w_k
    # (X - s*y) shape: (n, d)
    term_sum = jnp.sum((X - s * y) * weights[:, None], axis=0)
    
    # Pre-factor: 1 / (1 - s^2)
    prefactor = 1.0 / jnp.maximum(1.0 - jnp.square(s), 1e-4)
    
    velocity = prefactor * term_sum
    return jnp.clip(velocity, -2.0, 2.0)

# Solve the ODE to compute the Kim-Milman map T_KM
def compute_T_KM_1_delta(U, X, delta):
    ode_term = diffrax.ODETerm(v_theta)
    diff_ode_solv_type = diffrax.Tsit5()
    t_start, t_end, step_size = 0.0, 1.0 - delta, 0.05

    def solve_single_particle(y_init):
        solution = diffrax.diffeqsolve(
            ode_term, diff_ode_solv_type, t0=t_start, t1=t_end, dt0=step_size, y0=y_init, args=X,
            adjoint=diffrax.RecursiveCheckpointAdjoint(),
            max_steps=500
        )
        return solution.ys[-1]

    return jax.vmap(solve_single_particle)(U)

# Objective
def objective_from_samples(X, U_samples, epsilon, delta, rho_weights=None):
    """Compute -E_rho[||T_OT(U)-T_KM(U)||^2] from provided samples.

    If `rho_weights` is None, samples are assumed to be drawn directly from rho,
    so a plain empirical mean is used.
    If `rho_weights` is provided, a weighted empirical expectation is used.
    """
    T_OT_U = compute_T_OT_eps(U_samples, X, epsilon)
    T_KM_U = compute_T_KM_1_delta(U_samples, X, delta)
    sq_err = jnp.sum((T_OT_U - T_KM_U) ** 2, axis=1)
    sq_err = jnp.where(jnp.isfinite(sq_err), sq_err, 0.0)

    if rho_weights is None:
        # U_samples ~ rho  =>  E_rho[f(U)] ≈ (1/m) * sum_i f(U_i)
        exp_val = jnp.mean(sq_err)
    else:
        # Importance/weighted quadrature estimate of E_rho[f(U)].
        w = jnp.where(jnp.isfinite(rho_weights), rho_weights, 0.0)
        w = jnp.maximum(w, 0.0)
        w = w / (jnp.sum(w) + 1e-12)
        exp_val = jnp.sum(w * sq_err)

    return -exp_val


def objective_fn(X, key, m, epsilon, delta):
    d_dim = X.shape[1]
    U_samples = sample_rho(key, m, d_dim)
    return objective_from_samples(X, U_samples, epsilon, delta)

loss_and_grad_fn = eqx.filter_value_and_grad(objective_fn)

@eqx.filter_jit
def update_step(X, opt_state, key, m, epsilon, delta, optimizer_obj):
    loss_val, grad_X = loss_and_grad_fn(X, key, m, epsilon, delta)
    grad_X = jnp.where(jnp.isfinite(grad_X), grad_X, 0.0)
    grad_X = jnp.clip(grad_X, -0.05, 0.05)
    updates, opt_state_new = optimizer_obj.update(grad_X, opt_state, X)
    X_updated = optax.apply_updates(X, updates)
    return X_updated, opt_state_new, loss_val

# Main execution loop
def run_optimization():
    n_p, d_p, m_s = 5, 2, 128
    eps_val, del_val, lr, steps = 0.05, 0.01, 0.01, 200

    prng_key = jax.random.PRNGKey(1234)
    prng_key, sub_k = jax.random.split(prng_key)
    X_init = jax.random.normal(sub_k, shape=(n_p, d_p))
    X_curr = X_init

    opt_instance = optax.adam(lr)
    opt_state_curr = opt_instance.init(X_curr)
    history = []

    print("Starting optimize")
    try:
        for i in range(steps):
            prng_key, sub_k = jax.random.split(prng_key)
            X_curr, opt_state_curr, l_out = update_step(X_curr, opt_state_curr, sub_k, m_s, eps_val, del_val, opt_instance)
            loss_scalar = float(-l_out)
            history.append(loss_scalar)
            if i % 20 == 0:
                print(f"Step {i} | F(X) = {loss_scalar:.4f}")
    except Exception as exc:
        print(f"\nStopped early: {exc}")

    if history:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1); plt.plot(history); plt.title("The Objective value")
        plt.subplot(1, 2, 2)
        plt.scatter(X_init[:, 0], X_init[:, 1], alpha=0.3, label='Start')
        plt.scatter(X_curr[:, 0], X_curr[:, 1], color='red', label='End')
        plt.legend(); plt.show()

    return X_curr

X = run_optimization()
print(X)
print(f'\nShape: {X.shape}')
