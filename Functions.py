import numpy as np

# input format: xs[j][i] gives the value of the jth dimension of the ith point.
# e.g. the ith point in xs is given by xs[:, i]

def AckleyProblem(xs):
    out_shape = xs[0].shape
    a = np.exp(-0.2 * np.sqrt(1. / len(xs) * np.square(np.linalg.norm(xs, axis=0))))
    b = - np.exp(1. / len(xs) * np.sum(np.cos(2 * np.pi * xs), axis=0))
    return -20 * a + b + 20 + np.exp(1).reshape(out_shape)


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