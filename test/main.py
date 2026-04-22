from optimize import run_optimization_box, run_optimization_norm


if __name__ == "__main__":
    #for seed in range(10):
    seed = 9922
    run_optimization_norm(seed=seed, n_p=7, steps=500)
