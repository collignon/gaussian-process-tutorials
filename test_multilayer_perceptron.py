import numpy as np
from multilayer_perceptron import Neuron, Layer

def identity(x):
    return x

def test_single_input_neuron():
    '''Outputs the identity of the input value.'''
    neuron = Neuron(neuron_type='input')

    for x in [0, -10, np.pi]:
        assert neuron.compute_output(x) == x

def test_single_output_neuron():
    '''
    Outputs the linear combination of the input
    values, weighted by the connection weights.'''
    connection_weights = [1, -2, 1./3]
    neuron = Neuron(neuron_type='output', bias=1,
                connection_weights=connection_weights)

    input_values = [[-1, -.5, -3],
                    [2, 1, 0],
                    [2, 2, 9]]

    # 0 = 1 + 1*-1 + -2*-.5 + 1./3*3
    # 1 = 1 + 1*2 + -2*1 + 1./3*0
    # 2 = 1 + 1*2 + -2*2 + 1./3*9
    output_values = [0, 1, 2]

    for x, y in zip(input_values, output_values):
        assert neuron.compute_output(x) == y

def test_single_hidden_neuron_step():
    '''
    Outputs the linear combination of the input
    values, weighted by the connection weights,
    and passed through the activation function.'''

    connection_weights = [1, -2, 1./3]
    neuron = Neuron(neuron_type='hidden', bias=0,
                connection_weights=connection_weights)

    input_values = [[-1, -.5, -3],
                    [2, 1, 0],
                    [2, 2, 9]]

    # -1 = step(0 + 1*-1 + -2*-.5 + 1./3*3) = step(-1)
    # 1 = step(0 + 1*2 + -2*1 + 1./3*0) = step(0)
    # 1 = step(0 + 1*2 + -2*2 + 1./3*9) = step(1)
    output_values = [-1, 1, 1]

    for x, y in zip(input_values, output_values):
        assert neuron.compute_output(x) == y

def test_input_layer():
    '''
    Test output values for input layer with two neurons.
    '''
    input_layer = Layer(layer_type='input', num_neurons=2)
    input_values = [[0, 2], [-10, -20], [np.pi, 2*np.pi]]
    for x in input_values:
        assert input_layer.compute_output(x) == x

def test_output_layer():
    '''
    Test output values for output layer with two neurons.
    '''
    layer_biases = [0, -1]
    layer_weights = [[1, -2, .5], [0, -.1, 10]]
    output_layer = Layer(layer_type='output', num_neurons=2,
                         layer_biases=layer_biases,
                         layer_weights=layer_weights)

    input_values = [[[2, 1, 4], [10, 10, 0]],
                    [[-2, 2, -2], [2, -100, -1]]]
    output_values = [[2, -2], [-7, -1]]

    for x, y in zip(input_values, output_values):
        assert output_layer.compute_output(x) == y

def test_hidden_layer():
    '''
    Test output values for hidden layer with two neurons.
    '''
    layer_biases = [0, -1]
    layer_weights = [[1, -2, .5], [0, -.1, 10]]
    hidden_layer = Layer(layer_type='hidden', num_neurons=2,
                         layer_biases=layer_biases,
                         layer_weights=layer_weights)

    input_values = [[[2, 1, 4], [10, 10, 0]],
                    [[-2, 2, -2], [2, -100, -1]]]
    output_values = [[1, -1], [-1, -1]]

    for x, y in zip(input_values, output_values):
        assert hidden_layer.compute_output(x) == y
