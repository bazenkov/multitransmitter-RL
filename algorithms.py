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

if __name__ == "__main__":
    solution = np.array([0.5, 0.1, -0.3])
    def f(w): 
        return -np.sum((w - solution)**2)

    n_iter = 100
    npop = 50      # population size
    sigma = 0.1    # noise standard deviation
    alpha = 0.001  # learning rate
    w = np.random.randn(3) # initial guess
    w, pop, R_w = nes(f, w, n_iter, npop, sigma, alpha)

    plt.plot(R_w)
    plt.show()
    print(f"Solution found: {w}")
    print(f"True solution: {solution}")