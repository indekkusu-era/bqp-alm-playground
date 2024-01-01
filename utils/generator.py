import numpy as np

def generate_positive_definite_matrix(n):
    A = np.random.normal(size=(n, n))  # Generate a random matrix
    A = np.dot(A, A.T)        # Make it symmetric

    # Add a multiple of the identity matrix to make it positive definite
    A += np.eye(n) * 0.1

    return A / 10

def make_quadratic_function(Q, c):
    return lambda x: 1/2 * x.T @ Q @ x + np.dot(c, x)

def generate_quadratic_function(n):
    Q = generate_positive_definite_matrix(n)
    c = np.random.normal(0, 1, size=n)
    return make_quadratic_function(Q, c), Q, c
