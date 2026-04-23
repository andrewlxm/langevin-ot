import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from scipy.optimize import linear_sum_assignment

from core import sample_rho, compute_T_OT_eps, compute_T_KM_1_delta


REFERENCE_TARGETS = {}

# Function to compute canonical target IDs for consistent coloring in visualizations
def canonical_target_ids(X_final, is_box):
    x_np = np.asarray(X_final)
    n_p = x_np.shape[0]
    ref_key = ("box" if is_box else "norm", int(n_p))
    ref_targets = REFERENCE_TARGETS.get(ref_key)

    if ref_targets is None:
        REFERENCE_TARGETS[ref_key] = x_np.copy()
        return np.arange(n_p, dtype=np.int32)

    cost = np.sum((x_np[:, None, :] - ref_targets[None, :, :]) ** 2, axis=-1)
    row_ind, col_ind = linear_sum_assignment(cost)

    canon_ids = np.empty(n_p, dtype=np.int32)
    canon_ids[row_ind] = col_ind.astype(np.int32)
    return canon_ids


#Visualize the results
def visualize_results(X_init, X_final, history, title, is_box=True):
    eps_val, del_val = 0.001, 0.01
    n_p = X_final.shape[0]
    d_p = X_final.shape[1]

    viz_key = jax.random.PRNGKey(9221)
    U_viz = sample_rho(viz_key, 1000, d_p)
    T_OT_viz = compute_T_OT_eps(U_viz, X_final, eps_val)
    T_KM_viz = compute_T_KM_1_delta(U_viz, X_final, del_val)

    labels_KM = jnp.argmin(jnp.sum((T_KM_viz[:, None, :] - X_final[None, :, :]) ** 2, axis=-1), axis=1)
    labels_OT = jnp.argmin(jnp.sum((T_OT_viz[:, None, :] - X_final[None, :, :]) ** 2, axis=-1), axis=1)

    canon_ids = canonical_target_ids(X_final, is_box)
    target_ids = canon_ids[np.arange(n_p, dtype=np.int32)]
    labels_KM_plot = canon_ids[np.asarray(labels_KM, dtype=np.int32)]
    labels_OT_plot = canon_ids[np.asarray(labels_OT, dtype=np.int32)]

    label_cmap = plt.get_cmap('tab10', max(n_p, 1))
    label_norm = mcolors.BoundaryNorm(np.arange(-0.5, n_p + 0.5, 1.0), label_cmap.N)

    fig, axs = plt.subplots(1, 4, figsize=(20, 4))
    axs[0].plot(history)
    axs[0].set_title(title)

    if is_box:
        axs[1].add_patch(plt.Rectangle((-1, -1), 2, 2, fill=False, color='gray', linestyle='--'))
    else:
        theta = jnp.linspace(0, 2 * jnp.pi, 100)
        axs[1].plot(jnp.cos(theta), jnp.sin(theta), color='gray', linestyle='--', alpha=0.3)
    axs[1].scatter(X_init[:, 0], X_init[:, 1], alpha=0.4)
    axs[1].scatter(X_final[:, 0], X_final[:, 1], color='red')
    axs[1].set_xlim([-1.5, 1.5])
    axs[1].set_ylim([-1.5, 1.5])
    
    axs[2].scatter(U_viz[:, 0], U_viz[:, 1], c=labels_KM_plot, s=2, cmap=label_cmap, norm=label_norm, alpha=0.3)
    axs[2].scatter(X_final[:, 0], X_final[:, 1], c=np.asarray(target_ids), cmap=label_cmap, norm=label_norm, edgecolors='black')
    axs[2].set_title('KM cells')

    axs[3].scatter(U_viz[:, 0], U_viz[:, 1], c=labels_OT_plot, s=2, cmap=label_cmap, norm=label_norm, alpha=0.3)
    axs[3].scatter(X_final[:, 0], X_final[:, 1], c=np.asarray(target_ids), cmap=label_cmap, norm=label_norm, edgecolors='black')
    axs[3].set_title('OT cells')

    plt.savefig(f"{title}.png")
    #plt.show()