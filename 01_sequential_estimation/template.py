import numpy as np
from tools import scatter_2d_data, bar_per_axis

# Author: Arnar Ingi Njardarson
# Date:
# Project: 
# Acknowledgements: 

import matplotlib.pyplot as plt



def gen_data(n: int, k: int, mean: np.ndarray, std: np.float64) -> np.ndarray:
    '''Generate n values samples from the k-variate
    normal distribution
    '''
    cov = np.eye(k) * std ** 2  # Covariance matrix
    return np.random.multivariate_normal(mean, cov, n)


def update_sequence_mean(mu: np.ndarray, x: np.ndarray, n: int) -> np.ndarray:
    '''Performs the mean sequence estimation update
    '''
    return mu + (x - mu) / n


def _plot_sequence_estimate():
    data = gen_data(200, 3, np.array([1, 2, 3]), 0.5)  # Example data
    estimates = [np.array([1, 2, 3])]
    for i in range(data.shape[0]):
        estimates.append(update_sequence_mean(estimates[-1], data[i], i + 1))
    plt.plot([e[0] for e in estimates], label='First dimension')
    plt.plot([e[1] for e in estimates], label='Second dimension')
    plt.plot([e[2] for e in estimates], label='Third dimension')
    plt.legend(loc='upper center')
    plt.show()


def _square_error(y, y_hat):
    return (y - y_hat) ** 2


def _plot_mean_square_error():
    pass


# Naive solution to the independent question.

def gen_changing_data(n: int, k: int, start_mean: np.ndarray, end_mean: np.ndarray, std: np.float64) -> np.ndarray:
    data = []
    for i in range(n):
        current_mean = start_mean + (end_mean - start_mean) * i / n - 1
        data.append(gen_data(1, k, current_mean, std))
    return np.vstack(data)


def _plot_changing_sequence_estimate():
    # remove this if you don't go for the independent section
    pass


if __name__ == "__main__":
    # Example 1: Small dataset
    np.random.seed(1234)
    example1 = gen_data(2, 3, np.array([0, 1, -1]), 1.3)
    print("Example 1 output:")
    print(example1)
    print("\n")

    # Example 2: Larger dataset
    np.random.seed(1234)
    example2 = gen_data(5, 1, np.array([0.5]), 0.5)
    print("Example 2 output:")
    print(example2)
    print("\n")
    _plot_sequence_estimate()
