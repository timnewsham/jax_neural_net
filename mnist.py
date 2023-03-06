#!/usr/bin/env python
"""
MNIST data reader

To get data:
  git clone https://github.com/fgnt/mnist
"""
import gzip
import random
import struct
import numpy as np

from neural_net import *

def reader(basename) :
    # read labels and compute expected vectors
    fnIdx = basename + "-labels-idx1-ubyte.gz"
    fIdx = gzip.open(fnIdx)
    mag,cnt = struct.unpack("!II", fIdx.read(8))
    assert mag == 0x801
    labBytes = np.frombuffer(fIdx.read(cnt), dtype=np.uint8)
    expected = np.eye(10, dtype=np.float32)[labBytes]

    # read image data and compute flat image vectors
    fnImg = basename + "-images-idx3-ubyte.gz"
    fImg = gzip.open(fnImg)
    mag,cnt2,h,w = struct.unpack("!IIII", fImg.read(16))
    assert mag == 0x803
    assert cnt == cnt2
    imgBytes = np.frombuffer(fImg.read(w*h*cnt),dtype=np.uint8).reshape((cnt,28*28)) / 255.0
    imgs = imgBytes.reshape((cnt, 28*28))

    return imgs, expected

def showImg(flat) :
    def pixChar(p) :
        if p < 0.25 : return ' '
        if p < 0.50 : return '.'
        if p < 0.75 : return 'o'
        return '@'

    matrix = flat.reshape((28,28))
    for row in matrix:
        print(''.join(pixChar(val) for val in row))

def pick_training(seed, n, I, EO):
    """randomly pick n from (imgs[], expected[])"""
    key = jax.random.PRNGKey(seed)
    idx = np.arange(I.shape[0])
    idx = jax.random.permutation(key, idx, independent=True)[:n]
    return I[idx], EO[idx]

def evaluate(net, I, EO):
    """Evaluate the network over I and EO."""
    O = network(I, net)
    correct = 100.0 * sum(1.0
        for o,eo in zip(O, EO)
        if o.argmax() == eo.argmax()) / O.shape[0]
    print(f"got {correct}% right")

def showIncorrect(net, I, EO):
    O = network(I, net)
    for i, o, eo in zip(I, O, EO):
        want = eo.argmax()
        got = o.argmax()
        if want != got:
            print(f"Want {want} got {got}")
            print(f"output {o}")
            showImg(i)

# train in batches
#set_activation(sigmoid)
I, EO = reader('mnist/t10k')
net = random_net((28*28, 16, 16, 10))
for batch in range(50):
    print(f"batch {batch+1}: ", end='')
    TI, TEO = pick_training(batch, 100, I, EO)
    net = train_net(net, TI, TEO, iters=1000, training_rate=0.005)
    #evaluate(net, I, EO)

evaluate(net, I, EO)
#showIncorrect(net, I, EO)
