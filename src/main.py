from optimize import run_optimization_box, run_optimization_norm, run_optimization_joint


if __name__ == "__main__":
    for seed in range(10):
        #run_optimization_norm(seed=seed, n_p=7, steps=500)
        run_optimization_joint(seed=seed, n_p=7, steps=1000)