import numpy as np
from pypoly import Polynomial

# input format: xs[j][i] gives the value of the jth dimension of the ith point.
# e.g. the ith point in xs is given by xs[:, i]

def AckleyProblem(xs):
    out_shape = xs[0].shape
    a = np.exp(-0.2 * np.sqrt(1. / len(xs) * np.square(np.linalg.norm(xs, axis=0))))
    b = - np.exp(1. / len(xs) * np.sum(np.cos(2 * np.pi * xs), axis=0))
    return np.array(-20 * a + b + 20 + np.exp(1)).reshape(out_shape)


def GradAckleyProblem(xs):
    """del H/del xi = -20 * -0.2 * (xi * 1/n) / sqrt(1/n sum_j xj^2) * a + 2 pi sin(2 pi xi)/n * b"""
    out_shape = xs.shape
    a = np.exp(-0.2 * np.sqrt(1. / len(xs) * np.square(np.linalg.norm(xs, axis=0))))
    b = -np.exp(1. / len(xs) * np.sum(np.cos(2 * np.pi * xs), axis=0))
    a_p = -0.2 * (xs * 1. / len(xs)) / np.sqrt(1. / len(xs) * np.square(np.linalg.norm(xs, axis=0)))
    b_p = -2 * np.pi * np.sin(2 * np.pi * xs) / len(xs)
    return np.nan_to_num(
        -20 * a_p * a + b_p * b).reshape(out_shape)  # only when norm(x) == 0 do we have nan and we know the grad is zero there


def QuadraticFunctionInit(A, b):
    def QuadraticFunction(xs):
        out_shape = xs[0].shape
        xs = np.array([x.flatten() for x in xs])
        return (np.diag(np.dot(xs.T, np.dot(A, xs))) + b).reshape(out_shape)
    return QuadraticFunction

def GradQuadraticFunctionInit(A):
    def GradQuadraticFunction(xs):
        out_shape = xs.shape
        xs = np.array([x.flatten() for x in xs])
        grad = np.dot(xs.T, A + A.T).T
        return np.array([g for g in grad]).reshape(out_shape)
    return GradQuadraticFunction


def Gibbs(x, U, sig):
    return np.exp(-U(np.array(x)) / sig ** 2)


def GradGibbs(x, U, grad_U, sig):
    return -grad_U(x) * 1./sig**2 * Gibbs(x, U, sig)


def grad_poly_problem(inp):
    orig_shape = inp.shape

    if len(inp.shape) == 1:
        inp = inp.reshape((inp.shape[0], 1, 1))

    X = [[0, -2], [1, 1], [2, 0.5], [3, 1.5], [2.5, 1], [-2.5, 1], [-1, 1], [-2, 0.5], [-3, 1.5]]

    equations = np.array([[point[0] ** i for i in range(len(X))] for point in X])
    values = np.array([point[1] for point in X])
    coefficients = np.linalg.solve(equations, values)

    coefficients = coefficients[1:]
    coefficients = [coefficients[i] * (i + 1) for i in range(len(coefficients))]

    p = Polynomial(*coefficients)

    res = [np.array([[p(x) for x in row] for row in inp[0]])]
    for s in inp[1:]:
        res.append(2 * 3 / 9. * s)
    return np.array(res).reshape(orig_shape)


def poly_problem(inp):
    orig_shape = inp.shape
    if len(inp.shape) == 1:
        inp = inp.reshape((inp.shape[0], 1, 1))

    X = [[0, -2], [1, 1], [2, 0.5], [3, 1.5], [2.5, 1], [-2.5, 1], [-1, 1], [-2, 0.5], [-3, 1.5]]

    equations = np.array([[point[0] ** i for i in range(len(X))] for point in X])
    values = np.array([point[1] for point in X])
    coefficients = np.linalg.solve(equations, values)

    p = Polynomial(*coefficients)

    res = np.array([[p(x) for x in row] for row in inp[0]])
    for s in inp[1:]:
        res += 3 / 9. * s ** 2
    return res.reshape(orig_shape[1:])


if __name__ == "__main__":
    x = np.array([[0, 1], [0, 2]])
    g_out = Gibbs(x, AckleyProblem, 1)
    grad_g_out = GradGibbs(x, AckleyProblem, GradAckleyProblem, 1)
    print(g_out)
    print(grad_g_out)
