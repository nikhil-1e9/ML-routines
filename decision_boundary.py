import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_decision_boundary(clf, X, y, grid_step=0.01):
    '''
    Plots a decision boundary in 2D separating the positive and negative classes based on the classifier fitted to the data
    
    Parameters:
    clf -> classifier fitted to the data
    X(numpy.ndarray) -> Feature vector representing the data
    y(numpy.ndarray) -> Target vector
    grid_step(scalar) -> Size of steps to go from minimum to maximum value of the grid
                        (smaller values produce smoothed boundaries)
    
    Returns:
    No value, just plots the decision boundary and the scatter plots of the two classes
    '''
    
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, grid_step), np.arange(y_min, y_max, grid_step))
    
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contour(xx, yy, Z, cmap=plt.cm.Paired)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c="blue", label="positive")
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c="orange", label="negative")
    plt.legend()
    plt.show()
