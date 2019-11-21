import numpy as np

def grad_descent(func, grad_func, x_curr, eps, gamma, start_t=0, end_t=float("inf"), verbose=False):
    x_curr = np.array(x_curr, dtype=np.float)
    path = [x_curr]
    t = start_t
    while t < end_t:
        x_next = x_curr - gamma(t) * grad_func(x_curr)
        path.append(x_next)

        if np.abs(func(x_next) - func(x_curr)) < eps:  # TODO check what happens with more samples
            if verbose:
                print(grad_func(x_curr))
            break
        if (t % 50) == 0 and verbose:
            print("Iteration", t)
            print("diff", np.abs(func(x_next) - func(x_curr)))
        x_curr = x_next
        t += 1
    return np.array(path)


def simulated_annealing_janky(func, grad_func, x_curr, eps, gamma, temperature, start_t=0, end_t=float("inf"), verbose=False):
    x_curr = np.array(x_curr, dtype=np.float)
    path = [x_curr]
    t = start_t
    while t < end_t:
        x_next = x_curr + gamma(t) * (
                    -grad_func(x_curr) + temperature(t) * np.array([[np.random.normal()] for _ in range(x_curr.shape[0])]).reshape(x_curr.shape))
        path.append(x_next)

        if np.abs(func(x_next) - func(x_curr)) < eps:  # TODO check what happens with more samples
            if verbose:
                print(grad_func(x_curr))
            break

        if (t % 50) == 0 and verbose:
            print("Iteration", t)
            print("diff", np.abs(func(x_next) - func(x_curr)))

        x_curr = x_next
        t += 1
    return np.array(path)
