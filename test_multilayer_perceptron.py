import numpy as np
from multilayer_perceptron import Neuron, Layer, MultilayerPerceptron

def identity(x):
    return x

def test_single_input_neuron():
    '''Outputs the identity of the input value.'''
    neuron = Neuron(neuron_type='input')
    for x in [0, -10, np.pi]:
        assert neuron.compute_output(x) == x

    layer = Layer(layer_type='input')
    for x in [0, -10, np.pi]:
        assert layer.compute_output([x]) == [x]

def test_single_output_neuron():
    '''
    Outputs the linear combination of the input
    values, weighted by the connection weights.'''
    bias = 1
    connection_weights = [1, -2, 1./3]
    input_values = [[-1, -.5, -3],
                    [2, 1, 0],
                    [2, 2, 9]]

    # 0 = 1 + 1*-1 + -2*-.5 + 1./3*3
    # 1 = 1 + 1*2 + -2*1 + 1./3*0
    # 2 = 1 + 1*2 + -2*2 + 1./3*9
    output_values = [0, 1, 2]

    neuron = Neuron(neuron_type='output', bias=bias,
                connection_weights=connection_weights)
    for x, y in zip(input_values, output_values):
        assert neuron.compute_output(x) == y

    layer = Layer(layer_type='output', layer_biases=[bias],
                  layer_weights=[connection_weights])
    for x, y in zip(input_values, output_values):
        assert layer.compute_output(x) == [y]

def test_single_hidden_neuron_step():
    '''
    Outputs the linear combination of the input
    values, weighted by the connection weights,
    and passed through the activation function.'''
    bias = 0
    connection_weights = [1, -2, 1./3]
    input_values = [[-1, -.5, -3],
                    [2, 1, 0],
                    [2, 2, 9]]

    # -1 = step(0 + 1*-1 + -2*-.5 + 1./3*3) = step(-1)
    # 1 = step(0 + 1*2 + -2*1 + 1./3*0) = step(0)
    # 1 = step(0 + 1*2 + -2*2 + 1./3*9) = step(1)
    output_values = [-1, 1, 1]

    neuron = Neuron(neuron_type='hidden', bias=bias,
                connection_weights=connection_weights)
    for x, y in zip(input_values, output_values):
        assert neuron.compute_output(x) == y

    layer = Layer(layer_type='hidden', layer_biases=[bias],
                  layer_weights=[connection_weights])
    for x, y in zip(input_values, output_values):
        assert layer.compute_output(x) == [y]


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

    input_values = [[2, 1, 4], [10, 10, 0],
                    [-2, 2, -2], [2, -100, -1]]
    output_values = [[2, 38.9], [-10, -2],
                     [-7, -21.2], [201.5,-1]]

    for x, y in zip(input_values, output_values):
        assert np.allclose(output_layer.compute_output(x), y)

def test_hidden_layer():
    '''
    Test output values for hidden layer with two neurons.
    '''
    layer_biases = [0, -1]
    layer_weights = [[1, -2, .5], [0, -.1, 10]]
    hidden_layer = Layer(layer_type='hidden', num_neurons=2,
                         layer_biases=layer_biases,
                         layer_weights=layer_weights)
    input_values = [[2, 1, 4], [10, 10, 0],
                    [-2, 2, -2], [2, -100, -1]]
    output_values = [[1, 1], [-1, -1],
                     [-1, -1], [1, -1]]

    for x, y in zip(input_values, output_values):
        assert np.allclose(hidden_layer.compute_output(x), y)

def test_linear_single_perceptron():
    '''Test network without hidden layer and
    unidimensional input and output.'''
    num_layers = 2
    num_neurons = [1, 1]
    net_biases = [[2]]
    net_weights = [[[-3]]]

    input_values = [1, 0, -1]
    output_values = [-1, 2, 5]

    net = MultilayerPerceptron(
        num_layers, num_neurons, net_biases, net_weights)
    for x, y in zip(input_values, output_values):
        assert net.compute_output([x]) == [y]


def test_nonlinear_single_perceptron():
    '''Test network with a single hidden layer with one neuron and
    unidimensional input and output.'''
    num_layers = 3
    num_neurons = [1, 1, 1]
    net_biases = [[-1],[1]]
    net_weights = [[[1]], [[2]]]

    input_values = [1, 0, -1]
    output_values = [3, -1, -1]

    net = MultilayerPerceptron(
        num_layers, num_neurons, net_biases, net_weights)
    for x, y in zip(input_values, output_values):
        assert net.compute_output([x]) == [y]

def test_nonlinear_perceptron():
    '''Test network with a single hidden layer with three neurons and
    unidimensional input and output.'''
    num_layers = 3
    num_neurons = [1, 3, 1]
    net_biases = [[1, -1, 0], [10]]
    net_weights = [[[-1], [0], [1]], [[2, 4, -2]]]

    input_values = [1, 0, -1]
    output_values = [6, 6, 10]

    net = MultilayerPerceptron(
        num_layers, num_neurons, net_biases, net_weights)
    for x, y in zip(input_values, output_values):
        assert net.compute_output([x]) == [y]
