import jax
import jax.numpy as jnp
import diffrax
from ott.geometry import pointcloud
from ott.solvers import linear
from ott.problems.linear import linear_problem

#sample rho
def sample_rho(key, m, d):
    return jax.random.normal(key, shape=(m, d))

#compute Entropic OT map
def compute_T_OT_eps(U, X, epsilon):
    X_jitter = X + 1e-5 * jnp.ones_like(X)
    geom = pointcloud.PointCloud(U, X_jitter, epsilon=epsilon)
    prob_ot = linear_problem.LinearProblem(geom)
    sinkhorn_engine = linear.sinkhorn.Sinkhorn(
        threshold=1e-1,
        max_iterations=500,
        implicit_diff=None,
    )
    out_ot = sinkhorn_engine(prob_ot)
    P_mat = out_ot.matrix
    P_row_sum = jnp.sum(P_mat, axis=1, keepdims=True) + 1e-10
    P_norm = P_mat / P_row_sum
    return jnp.dot(P_norm, X)

#compute velocity of KM map with delta cutoff
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

#compute KM map with delta cutoff
def compute_T_KM_1_delta(U, X, delta):
    ode_term = diffrax.ODETerm(v_theta)
    diff_ode_solv_type = diffrax.Tsit5()
    t_start, t_end, step_size = 0.0, 1.0 - delta, 0.05

    def solve_single_particle(y_init):
        solution = diffrax.diffeqsolve(
            ode_term,
            diff_ode_solv_type,
            t0=t_start,
            t1=t_end,
            dt0=step_size,
            y0=y_init,
            args=X,
            adjoint=diffrax.RecursiveCheckpointAdjoint(),
            max_steps=500,
        )
        return solution.ys[-1]

    return jax.vmap(solve_single_particle)(U)