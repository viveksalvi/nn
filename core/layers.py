from typing import Protocol
from core.activations import Activation
import numpy as np


class Layer(Protocol):

    def __init__(self, activation:Activation, inputs:int, nodes: int):
        """initializes Layer"""

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Runs feedward on layer and returns Z=Activation(W*X)"""

    def get_last_input(self) -> np.ndarray:
        """returns last inputs for which forward was run"""

    def get_weights(self) -> np.ndarray:
        "returns weights matrix"

    def get_activations(self) -> np.ndarray:
        "returns activtions"

    def get_activations_derivative(self) -> np.ndarray:
        "returns activtions derivatives"

    def adjust_weights(self, adjustments:np.ndarray) -> None:
        "Does adjustments to weights"

class DenseLayer:

    def __init__(self, activation:Activation, inputs:int=2, nodes: int=3):
        """initializes Layer"""
        self.activation = activation
        self._inputs = inputs + 1
        self._nodes = nodes
        self.weights = np.random.normal(size=(self._nodes, self._inputs)) # np.arange(6).reshape(2,3) #
        self.weights[:, -1] = 0 
        self.x = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Runs feedward on layer and returns Z=Activation(W*X)"""
        self.x = np.append(x, 1).reshape(self._inputs, -1)
        # print("-------------")
        self.z = np.dot(self.weights, self.x)
        self.a = self.activation.activate(self.z)
        self.a_prime = self.activation.derivative(self.z) 
        # print(f"w={self.weights}")
        # print(f"x={self.x}")
        # print(f"a={self.a}")
        return self.a

    def get_weights(self) -> np.ndarray:
        "returns weights matrix"
        return self.weights

    def get_last_input(self) -> np.ndarray:
        """returns last inputs for which forward was run"""
        return self.x

    def get_activations(self) -> np.ndarray:
        "returns activtions"
        return self.a

    def get_activations_derivative(self) -> np.ndarray:
        "returns activtions derivatives"
        return self.a_prime

    def adjust_weights(self, adjustments:np.ndarray) -> None:
        "Does adjustments to weights"
        adjustments= adjustments.reshape(self._nodes, self._inputs)
        self.weights = self.weights + adjustments
