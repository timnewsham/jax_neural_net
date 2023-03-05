#!/usr/bin/env python
from neural_net import *

def test():
    # train for the AND function
    training_data = np.array([
        [[0,0], [1,0]],
        [[0,1], [1,0]],
        [[1,0], [1,0]],
        [[1,1], [0,1]],
    ], dtype=np.float32)

    net = train((2,2), training_data, iters=1000)
    print(f"\ntrained network {net}\n")

    # try it out
    for input in training_data[:,0]:
        outputs = network(input, net)
        best = outputs.argmax()
        print(f"{input} -> {outputs} best {best}")

test()
