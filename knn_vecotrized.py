import numpy as np
from collections import Counter
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


class KNearestNeighbors:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = self._predict(X)
        return np.array(y_pred)

    def _predict(self, x):
        # Compute distances between x and all examples in the training set

        distances = np.sqrt(np.sum((self.X_train - x[:, np.newaxis])**2, axis=2))

        # Sort by distance and return indices of the first k neighbors
        k_indices = np.argsort(distances)[:, :self.k]

        # Extract the labels of the k nearest neighbor training samples
        k_nearest_labels = self.y_train[k_indices]

        max_per_sample = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), 1, k_nearest_labels)

        # return the most common class label
        return max_per_sample



# Load the iris dataset
data = load_iris()
X, y = data.data, data.target

# Split the dataset into a train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model with k=3
model = KNearestNeighbors(k=3)

# Fit the model
model.fit(X_train, y_train)

# Predict the labels for test data
y_pred = model.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))
