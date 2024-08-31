# Author:
# Date:
# Project:
# Acknowledgements:

from tools import load_iris, split_train_test

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

def gen_data(n: int, locs: np.ndarray, scales: np.ndarray) -> np.ndarray:
    '''
    Return n data points, their classes and a unique list of all classes, from each normal distribution
    shifted and scaled by the values in locs and scales.
    '''
    data = []
    targets = []
    for i, (loc, scale) in enumerate(zip(locs, scales)):
        data.append(norm(loc, scale).rvs(size=n))
        targets.extend([i] * n)
    return np.concatenate(data), np.array(targets), list(range(len(locs)))

def mean_of_class(features: np.ndarray, targets: np.ndarray, selected_class: int) -> np.ndarray:
    '''
    Estimate the mean of a selected class given all features and targets in a dataset.
    '''
    class_features = features[targets == selected_class]
    return np.mean(class_features)

def covar_of_class(features: np.ndarray, targets: np.ndarray, selected_class: int) -> np.ndarray:
    '''
    Estimate the covariance of a selected class given all features and targets in a dataset.
    '''
    class_features = features[targets == selected_class]
    if class_features.size == 0:
        raise ValueError("No data available for the selected class.")
    if class_features.ndim > 1:
        # If features are multidimensional, compute the covariance matrix
        return np.cov(class_features, rowvar=False)
    else:
        # If features are one-dimensional, return the variance
        return np.var(class_features, ddof=1)

def likelihood_of_class(feature: np.ndarray, class_mean: np.ndarray, class_covar: np.ndarray) -> float:
    '''
    Estimate the likelihood that a sample is drawn from a multivariate normal distribution.
    '''
    # Ensure the covariance matrix is a 2D array
    class_covar = np.atleast_2d(class_covar)
    from scipy.stats import multivariate_normal

    # Check if the covariance matrix is singular
    if np.linalg.det(class_covar) == 0:
        raise ValueError("Covariance matrix is singular and cannot be used for density estimation.")

    # Calculate the probability density for each feature point
    return multivariate_normal(mean=class_mean, cov=class_covar).pdf(features)

def maximum_likelihood(train_features: np.ndarray, train_targets: np.ndarray, test_features: np.ndarray, classes: list) -> np.ndarray:
    '''
    Calculate the maximum likelihood for each test point in test_features.
    '''
    means, covs = [], []
    for class_label in classes:
        means.append(mean_of_class(train_features, train_targets, class_label))
        covs.append(covar_of_class(train_features, train_targets, class_label))

    likelihoods = []
    for test_feature in test_features:
        class_likelihoods = [likelihood_of_class(test_feature, mean, cov) for mean, cov in zip(means, covs)]
        likelihoods.append(class_likelihoods)
    return np.array(likelihoods)

def predict(likelihoods: np.ndarray):
    '''
    Make a prediction for each datapoint by choosing the highest likelihood.
    '''
    return np.argmax(likelihoods, axis=1)

def plot_data(features, targets):
    '''
    Plot data points with different markers for each class.
    '''
    markers = ['o', 's', 'x']
    colors = ['blue', 'orange', 'green']  # Ensure you have enough colors for all classes
    plt.figure(figsize=(10, 6))
    for class_index in np.unique(targets):
        plt.scatter(features[targets == class_index], np.zeros_like(features[targets == class_index]),
                    marker=markers[class_index % len(markers)],
                    color=colors[class_index % len(colors)],
                    label=f'Class {class_index}')
    plt.ylim(-0.05, 0.05)
    plt.xlim(features.min() - 1, features.max() + 1)
    plt.axhline(0, color='grey', linewidth=0.5)
    plt.xlabel('Feature value')
    plt.ylabel('Zero Line')
    plt.title('Data Points Colored by Class with Different Markers')
    plt.legend()
    plt.grid(True)
    plt.show()
    
if __name__ == "__main__":
    # SECTION 1: Generate and split the dataset
    # Generate and split the dataset
    features, targets, classes = gen_data(50, [-1, 1], [5, 5])
    (train_features, train_targets), (test_features, test_targets) = split_train_test(features, targets, train_ratio=0.8)

    print("Training features:", train_features)
    print("Test features:", test_features)
    print("Training targets:", train_targets)
    print("Test targets:", test_targets)
    
    # SECTION 2: Calculate the maximum likelihood
    plot_data(features, targets)
    
    #SECTION 3: MEAN OF CLASS
    # Calculate and store means for each class
    class_means = {mean_of_class(train_features, train_targets, 0)}
    
    class_mean = mean_of_class(train_features, train_targets, 0)
    print(f"Mean of class 0: {class_mean:.4f}")
    
    # Calculate covariance for a specific class, e.g., class 0
    class_cov = covar_of_class(train_features, train_targets, 0)
    print(f"Covariance of class 0: {class_cov:.4f}")
    
    print(f"Likelihoods of the features: {likelihood_of_class(test_features[0:3], class_mean, class_cov)}")
    
    maximum_likelihood(train_features, train_targets, test_features, classes)
    print(f"Maximum Likelihood: {maximum_likelihood(train_features, train_targets, test_features, classes)}")
    
    likelihoods = maximum_likelihood(train_features, train_targets, test_features, classes)
    print(f"predict(likelihoods) -> {predict(likelihoods)}")