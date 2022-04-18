
from typing import Protocol
import numpy as np

class Activation(Protocol):

    def activate(self, z: np.ndarray) -> np.ndarray:
        """Returns activation of given np.ndarray"""
    
    def derivative(self, z: np.ndarray) -> np.ndarray:
        """Returns derivative of activation of given np.ndarray"""


class SigmoidActivation:

    def activate(self, z: np.ndarray) -> np.ndarray:
        """Returns activation of given np.ndarray"""
        return 1/(1+np.exp(-z))

    def derivative(self, z: np.ndarray) -> np.ndarray:
        """Returns derivative of activation of given np.ndarray"""
        activation = self.activate(z)
        return  activation * (1-activation)


class LinearActivation:

    def activate(self, z: np.ndarray) -> np.ndarray:
        """Returns activation of given np.ndarray"""
        return z

    def derivative(self, z: np.ndarray) -> np.ndarray:
        """Returns derivative of activation of given np.ndarray"""
        return  np.ones(shape=z.shape)
    

class LogActivation:

    def activate(self, z: np.ndarray) -> np.ndarray:
        """Returns activation of given np.ndarray"""
        return np.log(z)

    def derivative(self, z: np.ndarray) -> np.ndarray:
        """Returns derivative of activation of given np.ndarray"""
        return 1/z


class ExpActivation:

    def activate(self, z: np.ndarray) -> np.ndarray:
        """Returns activation of given np.ndarray"""
        return np.exp(z)

    def derivative(self, z: np.ndarray) -> np.ndarray:
        """Returns derivative of activation of given np.ndarray"""
        return np.exp(z)


class SoftmaxActivation:

    def activate(self, z: np.ndarray) -> np.ndarray:
        """Returns activation of given np.ndarray"""
        exps = np.exp(z - np.max(z))
        return exps / np.sum(exps)

    def derivative(self, z: np.ndarray) -> np.ndarray:
        """Returns derivative of activation of given np.ndarray"""
        return 1 # I dont know yet

