#!/usr/bin/env python
import jax.numpy as np
import jax

@jax.jit
def sigmoid(x):
     return 1 / (1 + np.exp(-x))

@jax.jit
def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

# TODO: why doesnt this train well with sigmoid?!
#activation = sigmoid

activation = gelu

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

random_key = jax.random.PRNGKey(0)

def random_layer(input_size, output_size):
    W = jax.random.uniform(random_key, shape=(input_size, output_size))
    B = jax.random.uniform(random_key, shape=(output_size,))
    return W, B

def random_layers(sizes):
    return [
        random_layer(input_size, output_size)
        for input_size, output_size in zip(sizes, sizes[1:])
    ]

def train(sizes, training_data, iters=100, training_rate=0.1):
    """
    sizes is a list of the number of neurons in each layer.
    training_data has two columns, inputs and expected outputs.
    """
    I = training_data[:,0]
    EO = training_data[:,1]

    layers = random_layers(sizes)
    print("Training...")
    for n in range(iters):
        loss0 = loss(layers, I, EO)
        print(f"{n} Loss: {loss0}\r", end='')

        layers = gradient_descent(layers, I, EO, training_rate)
    print()
    return layers
