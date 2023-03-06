#!/usr/bin/env python
from neural_net import *
from andfunc import *

def test():
    # train for the AND function
    net = train((2,2), I, EO, iters=1000)
    print(f"\ntrained network {net}\n")

    # try it out
    for input in I:
        outputs = network(input, net)
        best = outputs.argmax()
        print(f"{input} -> {outputs} best {best}")

test()
