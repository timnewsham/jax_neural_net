#!/usr/bin/env python
import jax.numpy as np
import jax

@jax.jit
def sigmoid(x):
     return 1 / (1 + np.exp(-x))

@jax.jit
def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

activation = gelu

def set_activation(f):
    """
    Set the activation function.
    Due to the way jax jit works, this can only be done once
    before evaluating or training the network.
    """
    global activation
    activation = f

@jax.jit
def prebias(x):
    """Bias boolean inputs to the range -1,1"""
    return x * 2.0 - 1.0

@jax.jit
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

@jax.jit
def network(I, layers):
    """
    Compute a neural network's output probabilities for an input.
    I: vector-N
    layers: list of W, B
        I: vector-N
        W: matrix-NxM
        B: vector-M
        O: vector-M
    O: vector-M
    """
    # without prebias, test_train_sigmoid*.py dont work well...
    #I = prebias(I)

    for W, B in layers:
        Z = I @ W + B
        I = activation(Z)
    return softmax(I)

@jax.jit
def square_error(x, y):
    return np.sum(np.square(x - y))

@jax.jit
def loss(layers, I, EO):
    return square_error(network(I, layers), EO)

@jax.jit
def gradient_descent(layers, I, EO, rate):
    gradient = jax.grad(loss)(layers, I, EO)
    return [
        (W - rate * dw, B - rate * db)
        for (W, B), (dw, db) in zip(layers, gradient)
    ]

random_seed = 0
random_key = jax.random.PRNGKey(random_seed)

def random_layer(input_size, output_size):
    global random_key

    random_key, subkey = jax.random.split(random_key)
    W = jax.random.uniform(subkey, shape=(input_size, output_size))

    random_key, subkey = jax.random.split(random_key)
    B = jax.random.uniform(subkey, shape=(output_size,))

    # change range of W's and B's to be -0.5,0.5
    # then scale W's according to number of inputs to avoid saturation
    W = (W - 0.5) / np.sqrt(input_size)
    B = B - 0.5
    return W, B

def random_net(sizes):
    return [
        random_layer(input_size, output_size)
        for input_size, output_size in zip(sizes, sizes[1:])
    ]

def train_net(layers, I, EO, iters=100, training_rate=0.1):
    """
    sizes is a list of the number of neurons in each layer.
    """
    print("Training...")
    for n in range(iters):
        loss0 = loss(layers, I, EO)
        print(f"{n+1} Loss: {loss0}\r", end='')

        layers = gradient_descent(layers, I, EO, training_rate)
    print()
    return layers

def train(sizes, I, EO, iters=1000, training_rate=0.1):
    layers = random_net(sizes)
    return train_net(layers, I, EO, iters, training_rate)
