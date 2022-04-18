import numpy as np
from core.activations import *
from core.nn import NeuralNetowrk
from core.loss import *
from core.layers import DenseLayer

sigmoid_a = SigmoidActivation()
linear_a = LinearActivation()
log_a = LogActivation()
exp_a = ExpActivation()

nn = NeuralNetowrk(
    loss=MeanSquaredLoss(),
    layers=[
        # DenseLayer(activation=linear_a, inputs=2, nodes=2),
        # DenseLayer(activation=linear_a, inputs=2, nodes=5),
        # DenseLayer(activation=linear_a, inputs=5, nodes=2),
        DenseLayer(activation=linear_a, inputs=2, nodes=1)
    ],
    learning_rate=0.000001)


x_y = [
    [[3,4], 7], #12],
    [[2,4], 6], #8],
    [[3,2], 5], # 6],
    # [[3,40], 120],
    # [[6,4], 24],
    # [[5,4], 20],
    # [[3,11], 33],
    # [[17,4], 68],
    # [[3,14], 42],
]

exp_w = np.ones(shape=(2, 1), dtype=np.float64)

s = 0
initial_weights = nn.layers[0].get_weights()
for _ in range(333): # Epochs
    for datum in x_y:
        print(f"==== SAMPLE {s}====")
        
        s += 1
        # x = np.array(datum[0])
        # y = np.array([datum[1]])

        x0 = np.random.random_integers(0, 100)
        x1 = np.random.random_integers(0, 100)
        
        x = np.array([x0, x1])
        y = np.array([2*x0+x1])

        print(f"X = {x}")
        print(f"Y = {y}")
        nn.print_weights()

        out = nn.feed_forward(np.array(x))
        print(f"out => {out}")    
        err = nn.back_propogate(y)
        print(f"ERROR = {err}")
        nn.print_weights()
    #     if np.allclose(np.array(err), np.zeros(shape=(1,))):
    #         break
    # if np.allclose(np.array(err), np.zeros(shape=(1,))):
    #     break


nn.print_weights()
print(f"initial w : {initial_weights}")
test_data = [
    [14, 15],
    [1, 25],
    [0, 0],
    [0, 5],
    [5, 5],
    [123, 512], # 635
]

for td in test_data:
    print(f"TEST {td} = {nn.feed_forward(np.array(td))}")
