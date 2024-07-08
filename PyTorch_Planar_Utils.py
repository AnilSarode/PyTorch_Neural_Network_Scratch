import torch
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model



def load_planar_dataset():
    torch.manual_seed(1)
    m = 400  # number of examples
    N = int(m / 2)  # number of points per class
    D = 2  # dimensionality
    X = torch.zeros((m, D), dtype=torch.float32)  # data matrix where each row is a single example
    Y = torch.zeros((m, 1), dtype=torch.long)  # labels vector (0 for red, 1 for blue)
    a = 4  # maximum ray of the flower

    for j in range(2):
        ix = range(N * j, N * (j + 1))
        t = torch.linspace(j * 3.12, (j + 1) * 3.12, N) + torch.randn(N) * 0.2  # theta
        r = a * torch.sin(4 * t) + torch.randn(N) * 0.2  # radius
        X[ix] = torch.stack((r * torch.sin(t), r * torch.cos(t)), dim=1)
        Y[ix] = j

    X = X.t()
    Y = Y.t()

    return X, Y



def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min().item() - 1, X[0, :].max().item() + 1
    y_min, y_max = X[1, :].min().item() - 1, X[1, :].max().item() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = torch.meshgrid(torch.arange(x_min, x_max, h), torch.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    xy_tensor = torch.cat((xx.reshape(-1, 1), yy.reshape(-1, 1)), dim=1)
    Z = model(xy_tensor)
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :].numpy(), X[1, :].numpy(), c=y.numpy(), cmap=plt.cm.Spectral)



def plot_decision_boundary_PyTorch(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min().item() - 1, X[0, :].max().item() + 1
    y_min, y_max = X[1, :].min().item() - 1, X[1, :].max().item() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = torch.meshgrid(torch.arange(x_min, x_max, h), torch.arange(y_min, y_max, h), indexing='ij')
    # Predict the function value for the whole grid
    xy_tensor = torch.cat((xx.reshape(-1, 1), yy.reshape(-1, 1)), dim=1)
    Z = model(xy_tensor).detach().cpu().numpy() # important step
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :].numpy(), X[1, :].numpy(), c=y.numpy(), cmap=plt.cm.Spectral)
def sigmoid(x):
    """
    Compute the sigmoid of x
    Arguments:
    x -- A scalar or PyTorch tensor of any size.
    Return:
    s -- sigmoid(x)
    """
    s = 1 / (1 + torch.exp(-x))
    return s


