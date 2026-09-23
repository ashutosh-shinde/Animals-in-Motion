import numpy as np


def sphere(x):
    return np.sum(x**2)
# Globle minima at origin = 0, use S = 5.
# Shall take (1, D) Array as input and return value (distance from origin)^2.
# Basic check function before anuthing hard.


def ackley(x):
    D = len(x)
    term1 = -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / D))
    term2 = -np.exp(np.sum(np.cos(2*np.pi*x)) / D)
    return term1 + term2 + 20 + np.e
# Gobal minima : at origin = 0, use S = 32.768 (standard bound).
# Good test of whether your swarm escapes local traps.


def rastrigin(x):
    D = len(x)
    return 10*D + np.sum(x**2 - 10*np.cos(2*np.pi*x))
# Global minima : at origin = 0, use S = 5.12
# complemetory to ackley, just uniform local minima.


def goldstein_price(x):
    x1, x2 = x[0], x[1]
    a = 1 + (x1 + x2 + 1)**2 * (19 - 14*x1 + 3 *
                                x1**2 - 14*x2 + 6*x1*x2 + 3*x2**2)
    b = 30 + (2*x1 - 3*x2)**2 * (18 - 32*x1 + 12 *
                                 x1**2 + 48*x2 - 36*x1*x2 + 27*x2**2)
    return a * b
# D = 2 must.
# Global minima : at (0, -1) = 3, Use S = 2.
# Highly non-convex with large plateaus.
# Tests sensitivity to c (step size),
# since gradients vary wildly in magnitude across the space.


def levy_n13(x):
    x1, x2 = x[0], x[1]
    term1 = np.sin(3*np.pi*x1)**2
    term2 = (x1 - 1)**2 * (1 + np.sin(3*np.pi*x2)**2)
    term3 = (x2 - 1)**2 * (1 + np.sin(2*np.pi*x2)**2)
    return term1 + term2 + term3
# D = 2 must.
# Global minima : at (1, 1) = 0, use S = 10.
# Many local minima in a ring around the global one.
# Good test of exploration vs. premature convergence.


# print(sphere(np.zeros(3)))
# print(ackley(np.zeros(2)))
# print(rastrigin(np.zeros(5)))
# print(goldstein_price(np.array([0, -1])))
# print(levy_n13(np.array([1, 1])))
