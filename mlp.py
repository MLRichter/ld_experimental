import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, x, 0)


def relu_backwards(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, 1, 0)

def softmax(x: np.ndarray) -> np.ndarray:
    return np.exp(x) / np.exp(x).sum(axis=1)


class ReLU():

    def forward(self, x: np.ndarray):
        return np.maximum(0, x)

    def backward(self, output_error: np.ndarray, learning_rate):
        return np.where(output_error > 0, 1, 0)


class FCLayer():

    def __init__(self, units: int, units_in: int):
        self.units = units
        self.units_in = units_in
        self.weights: np.ndarray = np.random.random(size=(self.units_in, units))
        self.bias: np.ndarray = np.random.random(size=(self.units))

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        self.output =  x@self.weights+self.bias
        return self.output

    def backward(self, output_error: np.ndarray, learning_rate):
        input_error = output_error@self.weights.T
        weights_err = self.input.T @ output_error

        self.weights -= learning_rate * weights_err
        self.bias -= learning_rate * output_error
        return input_error

class MLP:

    def __init__(self, lr=0.01):
        self.layer = FCLayer(10, 20)


    def forward(self, x: np.ndarray):
        self.l1_out = self.l1act(x @ self.l1 + self.l1b)
        self.l2_out = self.l2act(x @ self.l2 + self.l2b)
        return self.l2_out


    def backward(self, x: np.ndarray, y: np.ndarray):
        l2_delta = self.l1_out.T (self.l2_out - y)





        self.l1 -= (self.lr * (output_delta))
        self.l2 -= (self.lr * (hidden_delta))
