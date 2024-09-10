# Author: 
# Date:
# Project: 
# Acknowledgements: 
#

import torch
import matplotlib.pyplot as plt

from tools import load_regression_iris
from scipy.stats import multivariate_normal


def mvn_basis(
    features: torch.Tensor,
    mu: torch.Tensor,
    var: float
) -> torch.Tensor:
    '''
    Multivariate Normal Basis Function
    The function transforms (possibly many) data vectors <features>
    to an output basis function output <fi>
    Inputs:
    * features: [NxD] is a data matrix with N D-dimensional
    data vectors.
    * mu: [MxD] matrix of M D-dimensional mean vectors defining
    the multivariate normal distributions.
    * var: All normal distributions are isotropic with sigma^2*I covariance
    matrices (where I is the MxM identity matrix)
    Output:
    * fi - [NxM] is the basis function vectors containing a basis function
    output fi for each data vector x in features
    '''
    N, D = features.shape
    M = mu.shape[0]
    fi = torch.zeros(N, M)
    
    covariance_matrix = var * torch.eye(D)
    
    for i in range(M):
        mvn = multivariate_normal(mean=mu[i].numpy(), cov=covariance_matrix.numpy())
        fi[:, i] = torch.tensor(mvn.pdf(features.numpy()))
    
    return fi

def _plot_mvn(fi: torch.Tensor):
    plt.figure(figsize=(8, 6))
    
    # Plot each basis function output (each column in fi is a basis function)
    for i in range(fi.shape[1]):
        plt.plot(fi[:, i].numpy(), label=f'Basis {i+1}')
    
    # Add labels, legend, and show plot
    plt.xlabel('Data Index')
    plt.ylabel('Basis Function Output')
    plt.title('Output of Each Basis Function')
    plt.legend(loc='upper right')
    plt.savefig('2_1.png')  # Save the plot as '2_1.png'
    plt.show()


def max_likelihood_linreg(
    fi: torch.Tensor,
    targets: torch.Tensor,
    lamda: float
) -> torch.Tensor:
    '''
    Estimate the maximum likelihood values for the linear model

    Inputs :
    * Fi: [NxM] is the array of basis function vectors
    * t: [Nx1] is the target value for each input in Fi
    * lamda: The regularization constant

    Output: [Mx1], the maximum likelihood estimate of w for the linear model
    '''
    N,M = fi.shape
    I = torch.eye(M)
    
    regularized_g_matrix = fi.T @ fi + lamda * I
    inverse_g_matrix = torch.inverse(regularized_g_matrix)
    
    w_ml = inverse_g_matrix @ fi.T @ targets
    
    return w_ml


def linear_model(
    features: torch.Tensor,
    mu: torch.Tensor,
    var: float,
    w: torch.Tensor
) -> torch.Tensor:
    '''
    Inputs:
    * features: [NxD] is a data matrix with N D-dimensional data vectors.
    * mu: [MxD] matrix of M D dimensional mean vectors defining the
    multivariate normal distributions.
    * var: All normal distributions are isotropic with sigma^2*I covariance
    matrices (where I is the MxM identity matrix).
    * w: [Mx1] the weights, e.g. the output from the max_likelihood_linreg
    function.

    Output: [Nx1] The prediction for each data vector in features
    '''
    fi = mvn_basis(features, mu, var)
    
    predictions = fi @ w
    
    return predictions


if __name__ == "__main__":
    """
    Keep all your test code here or in another file.
    """
    X, t = load_regression_iris()
    N, D = X.shape
    
    # Define the number of basis functions (M) and variance (var)
    M, var = 10, 10
    mu = torch.zeros((M, D))
    
    # Define the means of the Gaussian basis functions
    for i in range(D):
        mmin = torch.min(X[:, i])
        mmax = torch.max(X[:, i])
        mu[:, i] = torch.linspace(mmin, mmax, M)
    
    # Compute the basis functions
    fi = mvn_basis(X, mu, var)
    
    # Set the regularization constant
    lamda = 0.1
    
    w = max_likelihood_linreg(fi, t, lamda)
    
    predictions = linear_model(X, mu, var, w)
    
    _plot_mvn(fi)
    
    print("fi:\n", fi)

    lamda = 0.001
    w_ml = max_likelihood_linreg(fi, t, lamda)
    print("Wights with regularization:\n", w_ml)
    wml = max_likelihood_linreg(fi, t, lamda) # as before
    prediction = linear_model(X, mu, var, wml)
    print("Predictions with regularization:\n", prediction) 