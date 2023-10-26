import numpy as np
from collections import Counter
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


class KMeans:
    def __init__(self, k=3, iters=10):
        self.k = k
        self.X_train = None
        self.iters = iters

    def compute_distances_to_cluster(self, X: np.ndarray, means: np.ndarray) -> np.ndarray:
        distances = np.sqrt(np.sum((means-X[:, np.newaxis])**2), axis=2)
        labels = np.argmin(distances, axis=1)
        return labels


    def compute_means(self, assigned_cluster: np.ndarray, X: np.ndarray) -> np.ndarray:
        new_means = [X[assigned_cluster == k].mean(axis=1) for c in range(self.k)]
        




    def fit(self, X):
        self.X_train = X
        self.mean = np.random.random((self.k, X.shape[1]))

        for i in range(self.iters):
            ...





    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)



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
