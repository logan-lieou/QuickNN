import numpy as np

sigmoid = lambda x: 1 / (1 + np.exp(-x))

class Neuron():
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def forward(self, inputs):
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)

class QuickNN():
    def __init__(self):
        weights = np.array([0, 1])
        bias = 0

        self.h1 = Neuron(weights, bias)
        self.h2 = Neuron(weights, bias)
        self.o1 = Neuron(weights, bias)

    def forward(self, x):
        out_h1 = self.h1.forward(x)
        out_h2 = self.h2.forward(x)
        out_o1 = self.o1.forward(np.array([out_h1, out_h2]))

        return out_o1

mse = lambda yt, yp: ((yt - yp)**2).mean()

# TODO add training
"""
how to go about doing this: random init weights and biases using
np.random.normal() then after that generate hidden layers after
that you want to then create def train(epochs) where you do a
train loop this involves d_sigmoid because we are decsenting the
gradient find d loss or what not minimize or something
"""

def train(epochs: int, lr: float) -> QuickNN:
    pass

if __name__ == "__main__":
    model = QuickNN()
    x = np.array([3, 2])
    print("model output: ", model.forward(x))
