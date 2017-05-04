import numpy as np
from multilayer_perceptron import Neuron

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
