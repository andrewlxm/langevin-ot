import jax
import jax.numpy as jnp
import optax
import equinox as eqx

from core import sample_rho, compute_T_OT_eps, compute_T_KM_1_delta, compute_T_OT_weighted
from visualize import visualize_results

#Define the objective function between OT and KM maps, with importance weighting based on rho distribution
def objective_fn(X, key, m, epsilon, delta):
    d_dim = X.shape[1]
    U_samples = sample_rho(key, m, d_dim)
    T_OT_U = compute_T_OT_eps(U_samples, X, epsilon)
    T_KM_U = compute_T_KM_1_delta(U_samples, X, delta)
    diff = jnp.sum((T_OT_U - T_KM_U) ** 2, axis=1)
    weights = jnp.exp(-jnp.sum(U_samples**2, axis=1) / 2)
    loss = jnp.mean(weights * diff)
    return -loss


loss_and_grad_fn = eqx.filter_value_and_grad(objective_fn) #Compute both loss and gradient for optimization

#update step for box-constrained optimization
@eqx.filter_jit
def update_step_box(X, opt_state, key, m, epsilon, delta, optimizer_obj):
    loss_val, grad_X = loss_and_grad_fn(X, key, m, epsilon, delta)
    grad_X = jnp.where(jnp.isfinite(grad_X), grad_X, 0.0)
    grad_X = jnp.clip(grad_X, -0.05, 0.05)
    updates, opt_state_new = optimizer_obj.update(grad_X, opt_state, X)
    X_updated = optax.apply_updates(X, updates)
    X_projected = jnp.clip(X_updated, -1.0, 1.0)
    return X_projected, opt_state_new, loss_val

#update step for norm-constrained optimization
@eqx.filter_jit
def update_step_norm(X, opt_state, key, m, epsilon, delta, optimizer_obj):
    loss_val, grad_X = loss_and_grad_fn(X, key, m, epsilon, delta)
    grad_X = jnp.where(jnp.isfinite(grad_X), grad_X, 0.0)
    grad_X = jnp.clip(grad_X, -0.05, 0.05)
    updates, opt_state_new = optimizer_obj.update(grad_X, opt_state, X)
    X_updated = optax.apply_updates(X, updates)
    X_projected = X_updated / (jnp.linalg.norm(X_updated) + 1e-8) #Project back to unit sphere to maintain norm constraint
    return X_projected, opt_state_new, loss_val

#main optimization loop for box-constrained case
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
    for _ in range(steps):
        prng_key, sub_k = jax.random.split(prng_key)
        X_curr, opt_state_curr, l_out = update_step_box(
            X_curr, opt_state_curr, sub_k, m_s, eps_val, del_val, opt_instance
        )
        history.append(-float(l_out))
    visualize_results(X_init, X_curr, history, f"Box (Seed {seed}, N={n_p})", is_box=True)
    return X_curr

#main optimization loop for norm-constrained case
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
    for _ in range(steps):
        prng_key, sub_k = jax.random.split(prng_key)
        X_curr, opt_state_curr, l_out = update_step_norm(
            X_curr, opt_state_curr, sub_k, m_s, eps_val, del_val, opt_instance
        )
        history.append(-float(l_out))
    visualize_results(X_init, X_curr, history, f"Norm (Seed {seed}, N={n_p})", is_box=False)
    return X_curr

#------------------------------------Second part------------------------------------------------
#Joint optimization of positions and weights for OT map, with custom weight parameterization and projection to maintain constraints
def objective_fn_joint(params, key, m, epsilon, delta):
    X, w_logits = params
    d_dim = X.shape[1]
    U_samples = sample_rho(key, m, d_dim)
    
    # Map weights to simplex
    weights_X = jax.nn.softmax(w_logits)
    
    # Compute OT map with custom weights
    T_OT_U = compute_T_OT_weighted(U_samples, X, weights_X, epsilon)
    T_KM_U = compute_T_KM_1_delta(U_samples, X, delta)
    
    diff = jnp.sum((T_OT_U - T_KM_U) ** 2, axis=1)
    rho_weights = jnp.exp(-jnp.sum(U_samples**2, axis=1)/2)
    loss = jnp.mean(rho_weights * diff)
    return -loss

loss_grad_joint = eqx.filter_value_and_grad(objective_fn_joint)

@eqx.filter_jit
def update_step_joint(X, w_logits, opt_state, key, m, epsilon, delta, optimizer):
    params = (X, w_logits)
    loss_val, grads = loss_grad_joint(params, key, m, epsilon, delta)
    grad_X, grad_W = grads
    
    # Clean and clip gradients
    grad_X = jnp.where(jnp.isfinite(grad_X), grad_X, 0.0)
    grad_W = jnp.where(jnp.isfinite(grad_W), grad_W, 0.0)
    
    updates, opt_state_new = optimizer.update((grad_X, grad_W), opt_state, params)
    X_new, W_new = optax.apply_updates(params, updates)
    
    # Project X to unit norm as before
    X_proj = X_new / (jnp.linalg.norm(X_new) + 1e-8)
    return X_proj, W_new, opt_state_new, loss_val

def run_optimization_joint(seed=42, n_p=5, steps=500):
    d_p, m_s = 2, 128
    eps_val, del_val, lr = 0.001, 0.01, 0.01
    
    key = jax.random.PRNGKey(seed)
    k1, k2, k3 = jax.random.split(key, 3)
    
    # Init positions and weight logits
    X = jax.random.normal(k1, (n_p, d_p))
    X = X / jnp.linalg.norm(X)
    X_init = X # Store initial positions
    W_logits = jnp.zeros((n_p,))
    
    opt = optax.adam(lr)
    opt_state = opt.init((X, W_logits))
    
    history = []
    for i in range(steps):
        k3, sub_k = jax.random.split(k3)
        X, W_logits, opt_state, l = update_step_joint(X, W_logits, opt_state, sub_k, m_s, eps_val, del_val, opt)
        history.append(-float(l))
        
    final_weights = jax.nn.softmax(W_logits)
    print(f"Final Weights: {final_weights}")
    visualize_results(X_init, X, history, f"Opt W and X with n={n_p} and seed={seed}", is_box=False)
    return X, final_weights