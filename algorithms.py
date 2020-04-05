# simple example: minimize a quadratic around some solution point
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt


def nes(fun, w_0, n_iter, pop_size = 50, sigma = 1, alpha = 0.01, gamma = 0, params = []):
    def f(x):
        if params:
            return fun(x, params)
        else:
            return fun(x)
    R_history = np.zeros(n_iter)
    R = np.zeros(pop_size)
    w = w_0
    pop = np.zeros((pop_size, len(w_0)))
    best_history = []
    for i in range(n_iter):
        #noise = RNG.normal(scale = sigma, size=(pop_size, len(w)))
        noise = np.random.randn(pop_size, len(w))
        for j_p in range(pop_size):
            w_try = w + sigma*noise[j_p]
            pop[j_p] = w_try
            R[j_p] = f(w_try) - gamma*np.sum(np.abs(w_try))
        #A = (R - np.mean(R)) / np.std(R)
        A = R
        w = w + alpha/(pop_size*sigma) * np.dot(noise.T, A)
        R_history[i] = f(w)
        best_ind = np.argmax(R)
        best_history.append(pop[best_ind])
    return w, pop, R_history, best_history

def call_f(fun, x, params = None):
    if params:
        return fun(x, params)
    else:
        return fun(x)

def simple_ga(fun, w_0, n_iter, pop_size = 50, elite_frac = 0.1, sigma = 1, params = None):
    def crossover(parents):
        cross_ind = parents.shape[1] // 2
        return np.concatenate((parents[0,:cross_ind], parents[1,cross_ind:]))
    RNG = np.random.default_rng()
    R = np.zeros(pop_size)
    n_params = len(w_0)
    w = w_0
    pop = np.zeros((pop_size, n_params))
    n_elite = int(pop_size*elite_frac)
    R_history = np.zeros(n_iter)
    best_history = np.zeros((n_iter, n_params))
    for i_gen in range(n_iter):
        noise = np.random.randn(pop_size, n_params)
        for i_w in range(pop_size):
            pop[i_w,:] = w + sigma*noise[i_w]
            R[i_w] = call_f(fun, pop[i_w,:], params)
        #pick top 10 solutions
        elite = np.argsort(R)[-n_elite:]
        #crossover
        parents = RNG.choice(elite, size = 2, replace = False)
        alpha = i_gen/(1.1*n_iter)
        #alpha = 0
        w = w*alpha + (1-alpha)*crossover(pop[parents,:])
        best = pop[elite[-1],:]
        R_history[i_gen] = call_f(fun, w, params)
        #best_history.append(best)
        best_history[i_gen,:] = w
    return best, pop, R_history, best_history

        


if __name__ == "__main__":
    solution = np.array([0.5, 0.1, -0.3, 20, 50, -1000, 0])
    def f(w): 
        return -np.sum((w - solution)**2)
    n_param = solution.size
    n_iter = 300
    npop = 50      # population size
    sigma = 1    # noise standard deviation
    alpha = 0.2  # learning rate
    w = np.random.randn(n_param) # initial guess
    #w, pop, R_hist, w_hist = nes(f, w, n_iter, npop, sigma, alpha)
    w, pop, R_hist, w_hist = simple_ga(f, w, n_iter, npop, elite_frac = 0.1, sigma = 10)
    top = np.argmax(R_hist)
    best = w_hist[top]
    plt.plot(R_hist)
    plt.show()
    print(f"Solution found: {best}")
    print(f"True solution: {solution}")
    print(f"Reward: {R_hist[top]}")
    #print(w_hist)