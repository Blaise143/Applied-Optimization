import numpy as np
from typing import Final, Callable, Union
import matplotlib.pyplot as plt

v: Final[np.array] = np.array([-3, 8])

phi = lambda t: np.array([np.sin(t), np.cos(t)])

# def func(t):
#     parametrization = np.array([np.sin(t), np.cos(t)])
#     return np.linalg.norm(v - parametrization)#/np.sqrt(2)

func = lambda t: np.linalg.norm(v-phi(t))/np.sqrt(2)

def grad(f: Callable, t: Union[float, np.array], v: Union[float, np.array]) -> float|np.array:
    """
    Computes the gradient of f at point p
    """
    ft = f(t)
    grad_sin = -2 * (v[0] - np.sin(t)) * np.cos(t)
    grad_cos = -2 * (v[1] - np.cos(t)) * (-np.sin(t))
    gradient = grad_cos+grad_sin#np.array([np.cos(t), -np.sin(t)])
    return gradient


def optimize(f: Callable,
             starting_point: float,
             learning_rate: float=1e-1,
             num_steps: int=1000,
             tolerance: float = 1e-6) -> float:
    """
    Minimize f using gradient descent
    """
    current = f(starting_point)#.copy()
    # print(current)
    # exit()
    for _ in range(num_steps):
        delta = grad(f, current)
        step = delta*learning_rate
        # if np.all(np.abs(step) < tolerance):
        #     break
        current -= step
        print(current)
        # print(current)
    return current

optimal = optimize(func, 60.)
print(optimal)
print(func(optimal))
print(phi(optimal))
