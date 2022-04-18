from typing import Protocol
import numpy as np

class Loss(Protocol):

    def get_error(self, actual:np.ndarray, expected: np.ndarray) -> np.ndarray:
        """returns error in actual output against expected"""

    def get_error_derivative(self, actual:np.ndarray, expected: np.ndarray) -> np.ndarray:
        """returns derivative of error in actual output against expected"""

class MeanSquaredLoss:

    def get_error(self, actual:np.ndarray, expected: np.ndarray) -> np.ndarray:
        """returns error in actual output against expected"""
        return 0.5 * np.square(actual - expected)

    def get_error_derivative(self, actual:np.ndarray, expected: np.ndarray) -> np.ndarray:
        """returns derivative of error in actual output against expected"""
        return actual - expected


