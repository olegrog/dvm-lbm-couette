#!/usr/bin/env python
import numpy as np

class Grid():
    pass

class Uniform(Grid):
    def __init__(self, radius, N):
        idx = lambda N: np.arange(2*N) - N + .5
        self.x = idx(N) * radius / N
        self.w = radius / N * np.ones_like(self.x)

class Hermite(Grid):
    def __init__(self, radius, N):
        x, w = np.polynomial.hermite.hermgauss(2*N)
        w /= np.exp(-x**2)
        self.x, self.w = np.array((x, w)) * 2 * radius / np.sum(w)

class Polynomial(Grid):
    def __init__(self, radius, N, w_min, p=2):
        if radius-N*w_min < 0:
            raise NameError('q = %g is too big' % q)
        A = (radius-N*w_min) / np.sum(np.arange(N)**p)
        w = w_min + A*np.arange(N)**p
        x = np.append(0, np.cumsum(w))
        self.x, self.w = symm_x(x), symm_w(w)

class Geometric(Grid):
    def __init__(self, radius, N, q):
        if q == 1:
            return Uniform.__init__(self, radius, N)
        w1 = radius*(q-1)/(q**N-1)
        self.x = w1 * symm_x((q**np.arange(N+1)-1)/(q-1))
        self.w = w1 * symm_w(q**np.arange(N))

symm_w = lambda x: np.hstack((x[::-1], x))
semi_sum = lambda x: .5*(x[:-1] + x[1:])
symm_x = lambda x: np.hstack((-semi_sum(x)[::-1], semi_sum(x)))

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    f = lambda x: np.exp(-x**2)/np.pi**.5
    grids = map(lambda c: c.__name__ , Grid.__subclasses__())
    params = {
            'Polynomial': { 'w_min': 0.1, 'p': 2 },
        'Geometric': { 'q': 1.15 },
    }
    for n, name in enumerate(grids):
        radius, N = 4, 16
        g = globals()[name](radius, N, **params.get(name, {}))
        print(name, np.sum(f(g.x)*g.w))
        plt.plot(g.x, f(g.x)+n, '+-', label=name)

    plt.legend()
    plt.show()
