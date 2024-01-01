import numpy as np
from scipy import optimize

def quadratic_penalty(f, mu):
    return lambda x: f(x) + 1/(2*mu) * np.sum((x * (x-1)) ** 2)

def augmented_lagrangian(f, lam, mu):
    return lambda x: f(x) + np.dot(x * (x - 1), lam) + 1/(2*mu) * np.sum((x * (x-1)) ** 2)

def QP(obj_fn, x0, mu0, n_iter=50, gamma=0.9):
    x = x0
    mu = mu0
    for _ in range(n_iter):
        x1 = optimize.minimize(quadratic_penalty(obj_fn, mu), x).x
        x = x1
        mu *= gamma
    return np.where(x > 0.5, 1, 0)

def ALM(obj_fn, x0, lam0, mu0, n_iter=50, gamma=0.9):
    n = len(x0)
    bnds = [(0,1)] * n
    x = x0
    lam = lam0
    mu = mu0
    for _ in range(n_iter):
        x1 = optimize.minimize(augmented_lagrangian(obj_fn, lam, mu), x, bounds=bnds).x
        x = x1
        mu *= gamma
        lam += x1 * (x1 - 1) / mu
    return np.where(x > 0.5, 1, 0)
