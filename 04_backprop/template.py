from typing import Union
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from tools import load_iris, split_train_test


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    '''
    Calculate the sigmoid of x
    '''
    return 1 / (1 + torch.exp(-x))


def d_sigmoid(x: torch.Tensor) -> torch.Tensor:
    '''
    Calculate the derivative of the sigmoid of x.
    '''
    return sigmoid(x) * (1 - sigmoid(x))


def perceptron(
    x: torch.Tensor,
    w: torch.Tensor
) -> Union[torch.Tensor, torch.Tensor]:
    '''
    Return the weighted sum of x and w as well as
    the result of applying the sigmoid activation
    to the weighted sum
    '''
    weighted_sum = torch.dot(x, w)
    output = sigmoid(weighted_sum)
    
    return weighted_sum, output


def ffnn(
    x: torch.Tensor,
    M: int,
    K: int,
    W1: torch.Tensor,
    W2: torch.Tensor,
) -> Union[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    '''
    Computes the output and hidden layer variables for a
    single hidden layer feed-forward neural network.
    '''
    z0 = torch.cat([torch.Tensor([1.0]), x])
    a1 = torch.matmul(z0, W1)
    z1_non_bias = sigmoid(a1)
    z1 = torch.cat([torch.Tensor([1.0]), z1_non_bias])
    a2 = torch.matmul(z1, W2)
    y = sigmoid(a2)
    
    return y, z0, z1, a1, a2


def backprop(
    x: torch.Tensor,
    target_y: torch.Tensor,
    M: int,
    K: int,
    W1: torch.Tensor,
    W2: torch.Tensor
) -> Union[torch.Tensor, torch.Tensor, torch.Tensor]:
    '''
    Perform the backpropagation on given weights W1 and W2
    for the given input pair x, target_y
    '''
    y, z0, z1, a1, a2 = ffnn(x, M, K, W1, W2)
    delta_k = y - target_y 
    delta_j = d_sigmoid(a1) * torch.matmul(W2[1:], delta_k)

    # Initialize gradients
    dE1 = torch.zeros_like(W1)
    dE2 = torch.zeros_like(W2)

    # Compute gradients for W1 and W2
    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            dE1[i, j] = delta_j[j] * z0[i]

    for j in range(W2.shape[0]):
        for k in range(W2.shape[1]):
            dE2[j, k] = delta_k[k] * z1[j]

    return y, dE1, dE2


def cross_entropy_loss(y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    '''
    Compute the binary cross-entropy loss for multi-class classification.
    '''
    return -torch.sum(t * torch.log(y + 1e-10) + (1 - t) * torch.log(1 - y + 1e-10))


def misclassification_rate_fn(predicted: torch.Tensor, targets: torch.Tensor) -> float:
    '''
    Calculate the misclassification rate.
    '''
    predictions = torch.argmax(predicted, dim=1)
    incorrect = torch.sum(predictions != targets).item()
    return incorrect / targets.shape[0]


def train_nn(
    X_train: torch.Tensor,
    t_train: torch.Tensor,
    M: int,
    K: int,
    W1: torch.Tensor,
    W2: torch.Tensor,
    iterations: int,
    eta: float
) -> Union[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    '''
    Train a neural network with backpropagation.
    
    Outputs:
    - W1tr, W2tr: Updated weight matrices
    - Etotal: Array containing error after each iteration
    - misclassification_rate: Array of misclassification rates per iteration
    - last_guesses: The final output guesses from the network
    '''
    N = X_train.shape[0]
    Etotal = torch.zeros(iterations)
    # Initialize weight matrices
    W1tr = W1.clone()
    W2tr = W2.clone()
    misclassification_rate = torch.zeros(iterations)
    
    
    for it in range(iterations):
        dE1_total = torch.zeros_like(W1tr)
        dE2_total = torch.zeros_like(W2tr)
        total_loss = 0
        
        # Collect gradients and calculate the total error for each data point
        for i in range(N):
            x = X_train[i, :]
            target_y = torch.zeros(K)
            target_y[t_train[i]] = 1.0  # One-hot encode the target
            
            # Get backprop results
            y, dE1, dE2 = backprop(x, target_y, M, K, W1tr, W2tr)
            dE1_total += dE1
            dE2_total += dE2
            total_loss += cross_entropy_loss(y, target_y)
        
        
        # Store the total error
        Etotal[it] = total_loss / N
        
        # Calculate the misclassification rate
        last_guesses = test_nn(X_train, M, K, W1tr, W2tr)
        misclassification_rate[it] = misclassification_rate_fn(last_guesses, t_train)
        # Update weights
        W1tr -= eta * dE1_total / N
        W2tr -= eta * dE2_total / N
    # After the final iteration, get the final guesses
    last_guesses = torch.argmax(last_guesses, dim=1)
    
    return W1tr, W2tr, Etotal, misclassification_rate, last_guesses


def test_nn(
    X: torch.Tensor,
    M: int,
    K: int,
    W1: torch.Tensor,
    W2: torch.Tensor
) -> torch.Tensor:
    '''
    Return the predictions made by a network for all features
    in the test set X.
    '''
    N = X.shape[0]
    predictions = torch.zeros(N, K)
    
    for i in range(N):
        x = X[i, :]
        y, _, _, _, _ = ffnn(x, M, K, W1, W2)
        predictions[i, :] = y

    return predictions

def plot_confusion_matrix(cm, classes):
    """
    Plots the confusion matrix using matplotlib.
    """
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = range(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Display the numbers inside the matrix
    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, cm[i, j], horizontalalignment="center")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

if __name__ == "__main__":
    """
    You can test your code inside this scope without having to comment it out
    everytime before submitting. It also makes it easier for you to
    know what is going on in your code.
    """
    value1 = torch.Tensor([0.5])
    value2 = torch.Tensor([0.2])
    
    print(f"Sigmoid test: {sigmoid(value1).item():.4f}")
    print(f"Sigmoid derivative test: {d_sigmoid(value2).item():.4f}")
    print(f"Weighted sum and sigmoid weighted sum Ex.1: {perceptron(torch.Tensor([1.0, 2.3, 1.9]), torch.Tensor([0.2, 0.3, 0.1]))}")
    print(f"Weighted sum and sigmoid weighted sum Ex.2: {perceptron(torch.Tensor([0.2, 0.4]), torch.Tensor([0.1, 0.4]))}")
    
    # Load the Iris data
    torch.manual_seed(4321)  # For reproducibility
    features, targets, classes = load_iris()
    (train_features, train_targets), (test_features, test_targets) = split_train_test(features, targets)

    # Initialize the random generator to get repeatable results
    torch.manual_seed(1234)

    # Take one point from the training set
    x = train_features[0, :]  # The first training example
    K = 3  # Number of output neurons (classes)
    M = 10  # Number of hidden layer neurons
    D = 4  # Number of input features (dimensions)

    # Initialize random weight matrices for W1 and W2
    W1 = 2 * torch.rand(D + 1, M) - 1  # W1 has shape (D+1, M) for input to hidden layer
    W2 = 2 * torch.rand(M + 1, K) - 1  # W2 has shape (M+1, K) for hidden to output layer

    # Call the FFNN function
    y, z0, z1, a1, a2 = ffnn(x, M, K, W1, W2)

    # Print the outputs
    print(f"y  : {y}")
    print(f"z0 : {z0}")
    print(f"z1 : {z1}")
    print(f"a1 : {a1}")
    print(f"a2 : {a2}")
    
    torch.manual_seed(42)
    K = 3
    M = 6
    D = 4
    
    x = features[0, :]

    # create one-hot target for the feature
    target_y = torch.zeros(K)
    target_y[targets[0]] = 1.0

    # Initialize two random weight matrices
    W1 = 2 * torch.rand(D + 1, M) - 1
    W2 = 2 * torch.rand(M + 1, K) - 1

    y, dE1, dE2 = backprop(x, target_y, M, K, W1, W2)
    
    # Output results
    print("Output (y):", y)
    print("Gradient dE1:", dE1)
    print("Gradient dE2:", dE2)
    
    # Initialize the random seed to get predictable results
    torch.manual_seed(1234)
    
    K = 3  # number of classes
    M = 6
    D = train_features.shape[1]
    
    # Initialize two random weight matrices
    W1 = 2 * torch.rand(D + 1, M) - 1
    W2 = 2 * torch.rand(M + 1, K) - 1
    W1tr, W2tr, Etotal, misclassification_rate, last_guesses = train_nn(
        train_features[:20, :], train_targets[:20], M, K, W1, W2, 500, 0.1)
    
    # Print the outputs
    print(f"W1tr:\n{W1tr}")
    print(f"W2tr:\n{W2tr}")
    print(f"Etotal:\n{Etotal}")
    print(f"Misclassification Rate:\n{misclassification_rate}")
    print(f"Last guesses:\n{last_guesses}")
    
    # Example use case
    X_test = test_features  # The test set (features) you want to classify
    W1_trained = W1tr  # Trained weight matrix from the training process
    W2_trained = W2tr  # Trained weight matrix from the training process

    guesses = test_nn(X_test, M, K, W1_trained, W2_trained)

    print(f"Predicted classes: {guesses}")
    torch.manual_seed(66666)  # Set seed for reproducibility

    # Load the Iris data
    features, targets, classes = load_iris()
    (train_features, train_targets), (test_features, test_targets) = split_train_test(features, targets, train_ratio=0.8)

    # Network configuration
    K = 3  # Number of classes
    M = 6  # Number of hidden neurons
    D = train_features.shape[1]  # Number of input features

    # Initialize random weight matrices
    W1 = 2 * torch.rand(D + 1, M) - 1
    W2 = 2 * torch.rand(M + 1, K) - 1

    # Train the network
    W1_trained, W2_trained, E_total, misclassification_rate, _ = train_nn(
        train_features, train_targets, M, K, W1, W2, iterations=500, eta=0.1
    )

    # Test the network
    test_guesses = test_nn(test_features, M, K, W1_trained, W2_trained)
    test_guesses = torch.argmax(test_guesses, dim=1)

    # 1. Calculate accuracy
    accuracy = torch.sum(test_guesses == test_targets).item() / test_targets.shape[0]
    print(f"Accuracy on the test set: {accuracy:.4f}")

    # 2. Confusion matrix
    cm = confusion_matrix(test_targets, test_guesses)
    plot_confusion_matrix(cm, classes)
    plt.show()

    # 3. Plot E_total
    plt.figure()
    plt.plot(E_total.numpy())
    plt.title('E_total as a function of iterations')
    plt.xlabel('Iterations')
    plt.ylabel('E_total')
    plt.grid(True)
    plt.show()

    # 4. Plot misclassification rate
    plt.figure()
    plt.plot(misclassification_rate.numpy())
    plt.title('Misclassification Rate as a function of iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Misclassification Rate')
    plt.grid(True)
    plt.show()
    
