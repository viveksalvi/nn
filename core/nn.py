from typing import List
import numpy as np

from core.loss import Loss, MeanSquaredLoss
from core.activations import ExpActivation, LinearActivation, LogActivation, SigmoidActivation
from core.layers import DenseLayer, Layer


class NeuralNetowrk:
    def __init__(self, loss: Loss, layers: List[Layer], learning_rate = 0.1):
        """Initalize neural netwrk"""
        self.loss = loss
        self.layers = layers
        self.learning_rate = learning_rate
        self.out = None

    def print_weights(self):
        for n in range(len(self.layers)):
            print(f"L{n} : {self.layers[n].get_weights()}")

    def feed_forward(self, x: np.ndarray) -> np.ndarray:
        """Feed formward inputs x"""
        x_ = x
        for l in self.layers:
            x_ = l.forward(x_)
        self.out = x_
        return self.out

    def back_propogate(self, expected: np.ndarray):
        """Adjust weights according to error delta"""
        if self.out is None:
            raise Exception("Please call feed_forward before calling back_propogate")
        de = self.loss.get_error_derivative(self.out, expected)
        delta = de * self.layers[-1].get_activations_derivative()
        self.layers.reverse()

        all_adjustments = []
        
        for n in range(0, len(self.layers)):
            if n>0:
                w_n_1 = self.layers[n-1].get_weights()[:, :-1].transpose()
                d_activation = self.layers[n].get_activations_derivative() 
                delta = np.dot(w_n_1, delta) * d_activation

            last_input = self.layers[n].get_last_input().transpose()
            grad = delta * last_input
            
            adjustments = -self.learning_rate * grad.transpose()
            all_adjustments.append(adjustments)
            
        for n in range(0, len(self.layers)):
            self.layers[n].adjust_weights(all_adjustments[n])

        err = self.loss.get_error(self.out, expected)

        self.layers.reverse()
        self.out = None
        return err

