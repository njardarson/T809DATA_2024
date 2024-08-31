import numpy as np
import matplotlib.pyplot as plt

# Author: Arnar Njarðarson
# Date:
# Project: 
# Acknowledgements: 

def gen_data(n: int, k: int, mean: np.ndarray, std: np.float64) -> np.ndarray:
    '''
    Generate n values samples from the k-variate normal distribution
    Parameters:
        n (int): Number of samples
        k (int): Dimension of the distribution
        mean (np.ndarray): Mean vector of the distribution
        std (np.float64): Standard deviation of the distribution
    Returns:
        np.ndarray: An array of sampled points
    '''
    cov = np.eye(k) * std ** 2  # Covariance matrix
    return np.random.multivariate_normal(mean, cov, n)

def update_sequence_mean(mu: np.ndarray, x: np.ndarray, n: int) -> np.ndarray:
    '''
    Performs the mean sequence estimation update
    Parameters:
        mu (np.ndarray): Current mean estimate
        x (np.ndarray): New data point
        n (int): Sequence index (1-based)
    Returns:
        np.ndarray: Updated mean estimate
    '''
    return mu + (x - mu) / n

def _plot_sequence_estimate():
    data = gen_data(100, 2, np.array([0, 0]), np.sqrt(3))
    estimates = [np.array([0, 0])]
    
    for i in range(data.shape[0]):
        new_estimate = update_sequence_mean(estimates[-1], data[i], i + 1)
        estimates.append(new_estimate)
    
    estimates = np.array(estimates)
    plt.plot(estimates[:, 0], label='Dimension 1')
    plt.plot(estimates[:, 1], label='Dimension 2')
    plt.xlabel('Sequence Index')
    plt.ylabel('Mean Estimate')
    plt.title('Sequential Mean Estimation')
    plt.legend()
    plt.show()

def _square_error(y, y_hat):
    return (y - y_hat) ** 2

def _plot_mean_square_error():
    data = gen_data(100, 2, np.array([0, 0]), np.sqrt(3))
    true_mean = np.array([0, 0])
    estimates = [true_mean]
    squared_errors = []
    
    for i in range(data.shape[0]):
        new_estimate = update_sequence_mean(estimates[-1], data[i], i + 1)
        estimates.append(new_estimate)
        squared_errors.append(np.mean(_square_error(true_mean, new_estimate)))
    
    squared_errors = np.array(squared_errors)
    plt.plot(squared_errors, label='Mean Squared Error')
    plt.xlabel('Sequence Index')
    plt.ylabel('Mean Squared Error')
    plt.title('Mean Squared Error Over Time')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # SECTION 1: EXAMPLES

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

    # SECTION 3
    np.random.seed(1234)  # Set seed for reproducibility
    X = gen_data(300, 2, np.array([-1, 2]), 2)
    mean = np.mean(X, 0)  # Calculate the initial mean of the generated data
    new_x = gen_data(1, 2, np.array([0, 0]), 1)
    updated_mean = update_sequence_mean(mean, new_x, X.shape[0] + 1)
    
    print("Updated mean:", updated_mean)

    # SECTION 4
    _plot_sequence_estimate()
    # SECTION 5
    _plot_mean_square_error()
