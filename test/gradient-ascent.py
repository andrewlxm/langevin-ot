import jax
import jax.numpy as jnp
import numpy as np
import optax
import diffrax
import equinox as eqx
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from scipy.optimize import linear_sum_assignment
from ott.geometry import pointcloud
from ott.solvers import linear
from ott.problems.linear import linear_problem

REFERENCE_TARGETS = {}

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

# Velocity field for the Kim-Milman flow
def v_theta(t, y, X):
    s = t
    denom_kernel = 2.0 * jnp.square(jnp.maximum(1.0 - s, 1e-4))
    diffs = y - s * X
    dist_sq = jnp.sum(jnp.square(diffs), axis=1)
    logits = -dist_sq / denom_kernel
    weights = jax.nn.softmax(logits)
    term_sum = jnp.sum((X - s * y) * weights[:, None], axis=0)
    prefactor = 1.0 / jnp.maximum(1.0 - jnp.square(s), 1e-4)
    velocity = prefactor * term_sum
    return jnp.clip(velocity, -2.0, 2.0)

# Solve the ODE
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

#Define the objective function as the negative of the weighted MSE between the two maps, where the weights are given by a Gaussian density on U
def objective_fn(X, key, m, epsilon, delta):
    d_dim = X.shape[1]
    U_samples = sample_rho(key, m, d_dim)
    T_OT_U = compute_T_OT_eps(U_samples, X, epsilon)
    T_KM_U = compute_T_KM_1_delta(U_samples, X, delta)
    diff = jnp.sum((T_OT_U - T_KM_U) ** 2, axis=1)
    weights = jnp.exp(-jnp.sum(U_samples**2, axis=1)/2)
    loss = jnp.mean(weights * diff)
    return -loss

loss_and_grad_fn = eqx.filter_value_and_grad(objective_fn) #compute both the loss and its gradient with respect to X

#update step for box constraint, projects onto the box [-1, 1]^2 after the update
@eqx.filter_jit
def update_step_box(X, opt_state, key, m, epsilon, delta, optimizer_obj):
    loss_val, grad_X = loss_and_grad_fn(X, key, m, epsilon, delta)
    grad_X = jnp.where(jnp.isfinite(grad_X), grad_X, 0.0)
    grad_X = jnp.clip(grad_X, -0.05, 0.05)
    updates, opt_state_new = optimizer_obj.update(grad_X, opt_state, X)
    X_updated = optax.apply_updates(X, updates)
    X_projected = jnp.clip(X_updated, -1.0, 1.0)
    return X_projected, opt_state_new, loss_val

#update step for norm constraint, projects onto the unit circle after the update
@eqx.filter_jit
def update_step_norm(X, opt_state, key, m, epsilon, delta, optimizer_obj):
    loss_val, grad_X = loss_and_grad_fn(X, key, m, epsilon, delta)
    grad_X = jnp.where(jnp.isfinite(grad_X), grad_X, 0.0)
    grad_X = jnp.clip(grad_X, -0.05, 0.05)
    updates, opt_state_new = optimizer_obj.update(grad_X, opt_state, X)
    X_updated = optax.apply_updates(X, updates)
    X_projected = X_updated / (jnp.linalg.norm(X_updated) + 1e-8)
    return X_projected, opt_state_new, loss_val

# optimization with box constraint, initialized uniformly in the box [-1, 1]^2
def run_optimization_box(seed=228, n_p=5):
    d_p, m_s = 2, 128
    eps_val, del_val, lr, steps = 0.001, 0.01, 0.02, 500
    prng_key = jax.random.PRNGKey(seed)
    prng_key, sub_k = jax.random.split(prng_key)
    X_curr = jax.random.uniform(sub_k, (n_p, d_p), minval=-1.0, maxval=1.0)
    X_init = X_curr
    opt_instance = optax.adam(lr)
    opt_state_curr = opt_instance.init(X_curr)
    history = []
    for i in range(steps):
        prng_key, sub_k = jax.random.split(prng_key)
        X_curr, opt_state_curr, l_out = update_step_box(X_curr, opt_state_curr, sub_k, m_s, eps_val, del_val, opt_instance)
        history.append(-float(l_out))
    visualize_results(X_init, X_curr, history, f"Box (Seed {seed}, N={n_p})", is_box=True)
    return X_curr

#Optimization with norm constraint, initialized on the unit circle
def run_optimization_norm(seed=9922, n_p=5, steps=500):
    d_p, m_s = 2, 128
    eps_val, del_val, lr = 0.001, 0.01, 0.04
    prng_key = jax.random.PRNGKey(seed)
    prng_key, sub_k = jax.random.split(prng_key)
    X_curr = jax.random.normal(sub_k, (n_p, d_p))
    X_curr = X_curr / jnp.linalg.norm(X_curr)
    X_init = X_curr
    opt_instance = optax.adam(lr)
    opt_state_curr = opt_instance.init(X_curr)
    history = []
    for i in range(steps):
        prng_key, sub_k = jax.random.split(prng_key)
        X_curr, opt_state_curr, l_out = update_step_norm(X_curr, opt_state_curr, sub_k, m_s, eps_val, del_val, opt_instance)
        history.append(-float(l_out))
    visualize_results(X_init, X_curr, history, f"Norm (Seed {seed}, N={n_p})", is_box=False)
    return X_curr

#Try to match the canonical target ids across runs for better visualization
def canonical_target_ids(X_final, is_box):
    x_np = np.asarray(X_final)
    n_p = x_np.shape[0]
    ref_key = ("box" if is_box else "norm", int(n_p))
    ref_targets = REFERENCE_TARGETS.get(ref_key)

    # The first run for each (mode, n) becomes the reference labeling.
    if ref_targets is None:
        REFERENCE_TARGETS[ref_key] = x_np.copy()
        return np.arange(n_p, dtype=np.int32)

    # Match current targets to reference targets via optimal assignment.
    cost = np.sum((x_np[:, None, :] - ref_targets[None, :, :]) ** 2, axis=-1)
    row_ind, col_ind = linear_sum_assignment(cost)

    canon_ids = np.empty(n_p, dtype=np.int32)
    canon_ids[row_ind] = col_ind.astype(np.int32)
    return canon_ids

def visualize_results(X_init, X_final, history, title, is_box=True):
    eps_val, del_val = 0.001, 0.01
    n_p = X_final.shape[0]
    d_p = X_final.shape[1]

    viz_key = jax.random.PRNGKey(9221)
    U_viz = sample_rho(viz_key, 1000, d_p)
    T_OT_viz = compute_T_OT_eps(U_viz, X_final, eps_val)
    T_KM_viz = compute_T_KM_1_delta(U_viz, X_final, del_val)

    labels_KM = jnp.argmin(jnp.sum((T_KM_viz[:, None, :] - X_final[None, :, :])**2, axis=-1), axis=1)
    labels_OT = jnp.argmin(jnp.sum((T_OT_viz[:, None, :] - X_final[None, :, :])**2, axis=-1), axis=1)

    canon_ids = canonical_target_ids(X_final, is_box)
    target_ids = canon_ids[np.arange(n_p, dtype=np.int32)]

    labels_KM_plot = canon_ids[np.asarray(labels_KM, dtype=np.int32)]
    labels_OT_plot = canon_ids[np.asarray(labels_OT, dtype=np.int32)]

    label_cmap = plt.get_cmap('tab10', max(n_p, 1))
    label_norm = mcolors.BoundaryNorm(np.arange(-0.5, n_p + 0.5, 1.0), label_cmap.N)

    fig, axs = plt.subplots(1, 4, figsize=(20, 4))
    axs[0].plot(history); axs[0].set_title(title)

    if is_box:
        axs[1].add_patch(plt.Rectangle((-1, -1), 2, 2, fill=False, color='gray', linestyle='--'))
    else:
        theta = jnp.linspace(0, 2*jnp.pi, 100)
        axs[1].plot(jnp.cos(theta), jnp.sin(theta), color='gray', linestyle='--', alpha=0.3)
    axs[1].scatter(X_init[:, 0], X_init[:, 1], alpha=0.4)
    axs[1].scatter(X_final[:, 0], X_final[:, 1], color='red')
    axs[1].set_xlim([-1.5, 1.5]); axs[1].set_ylim([-1.5, 1.5])
    axs[2].scatter(U_viz[:, 0], U_viz[:, 1], c=labels_KM_plot, s=2, cmap=label_cmap, norm=label_norm, alpha=0.3)
    axs[2].scatter(X_final[:, 0], X_final[:, 1], c=np.asarray(target_ids), cmap=label_cmap, norm=label_norm, edgecolors='black')
    axs[3].scatter(U_viz[:, 0], U_viz[:, 1], c=labels_OT_plot, s=2, cmap=label_cmap, norm=label_norm, alpha=0.3)
    axs[3].scatter(X_final[:, 0], X_final[:, 1], c=np.asarray(target_ids), cmap=label_cmap, norm=label_norm, edgecolors='black')
    #save fig to results folder
    plt.savefig(f"results/{title}.png")
    #plt.show()
    # #save the plot
    # plt.savefig(f"{title}.png")


if __name__ == "__main__":
    #run_optimization_box(seed=228, n_p=5)
    for seed in range(10):
        run_optimization_norm(seed=seed, n_p=7, steps=500)
