# Setup
# Library for tensor manipulation.
import numpy as np

# Library for tabular data manipulation.
import pandas as pd

# Plotting libraries.
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Utils.
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# Data, Features and Target
# Training set.
# Features.
X_train = np.genfromtxt("..\data\X_train.csv", delimiter=',')
# Target.
y_train = np.genfromtxt("..\data\y_train.csv", delimiter=',')

# Test set.
# Features.
X_test = np.genfromtxt("..\data\X_test.csv", delimiter=',')
# Target.
y_test = np.genfromtxt("..\data\y_test.csv", delimiter=',')

# Data Visualization
# Scatter plot of training data.
plt.figure(figsize=(10, 10))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.coolwarm)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title("Training Data")
plt.legend(handles=[mpatches.Patch(color='#b40426', label='+1'), mpatches.Patch(color='#3b4cc0', label='-1')])
plt.show()


def train_AdaBoost(X, y, k):
    n = X.shape[0]
    
    D = np.concatenate((X, y.reshape(-1, 1)), axis=1)
    
    w = np.full(n, 1/n)

    classifiers = []
    alphas = []
    for i in range(k):
        # Sample from the dataset according to weights.
        idx_i = np.random.choice(a=np.arange(0, n), size=n, replace=True, p=w)
        D_i = D[idx_i]

        # Fit a decision stump.
        clf_i = DecisionTreeClassifier(max_depth=1, random_state=0)
        clf_i.fit(D_i[:, :-1], D_i[:, -1])
        
        # Predictions.
        preds = clf_i.predict(D_i[:, :-1])
        
        # Calculate the error rate.
        error_rate = np.sum(np.where(y == preds, 0, 1) * w)
        
        # Calculate the weight of classifier's vote.
        alpha_i = 0.5 * np.log2((1 - error_rate) / error_rate)
        
        # Increase the weight of misclassified points and vice-versa.
        w = w * np.exp(-1 * alpha_i * y * preds)
        w = w / np.sum(w)
        
        # Append your classifier to the list classifiers.
        classifiers.append(clf_i)
        
        # Append your alpha to the list alphas.
        alphas.append(alpha_i)
    
    return classifiers, alphas


def predict_AdaBoost(X, classifiers, alphas):
    n = X.shape[0]
    
    preds = np.zeros((n, 1))
    for classifier, alpha in zip(classifiers, alphas):
        preds_i = classifier.predict(X).reshape(-1, 1)
        
        preds += preds_i * alpha
    
    y_pred = np.sign(preds).reshape(-1,)
    
    return y_pred


def plot_AdaBoost(X, y, classifiers, alphas):
    # Get limits of x and y for plotting the decision surface.
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    
    # Get points at a distance of h between the above limits .
    h = .02    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    temp = np.c_[xx.ravel(), yy.ravel()]
    
    # Classify the all the points.
    P = predict_AdaBoost(temp, classifiers, alphas).reshape(yy.shape)
    
    # Plot the decision boundary and margin
    plt.pcolormesh(xx, yy, P, cmap=plt.cm.coolwarm, shading='auto')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolor='k')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.legend(handles=[mpatches.Patch(color='#b40426', label='+1'), mpatches.Patch(color='#3b4cc0', label='-1')])
    plt.show()


if __name__ == "__main__":
    for k in [5, 100, 500, 1000]:
        print("="*25)

        classifiers, alphas = train_AdaBoost(X_train, y_train, k)

        y_preds = predict_AdaBoost(X_test, classifiers, alphas)

        print("k: ", k, " Accuracy Score: ", accuracy_score(y_test, y_preds))

        plot_AdaBoost(X_test, y_test, classifiers, alphas)

        print("="*25)