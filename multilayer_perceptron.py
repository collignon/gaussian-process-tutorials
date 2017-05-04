import numpy as np

# class MultilayerPerceptron():

class Neuron():
    def __init__(self, neuron_type, bias=None, connection_weights=None,
                 activation_function=None):
        identity = lambda x: x
        step_function = lambda x: -1 if x < 0 else 1

        # Neuron type properties
        if neuron_type not in ['input', 'hidden', 'output']:
            raise ValueError('Unknown Neuron Type %s' %(neuron_type,))
        if neuron_type == 'input':
            bias = 0
            connection_weights = [1]
            activation_function = identity
        if neuron_type == 'output':
            activation_function = identity

        # Default initializations
        if bias is None:
            bias = 0
        if connection_weights is None:
            connection_weights = [1]
        if activation_function is None:
            activation_function = step_function

        self.neuron_type = neuron_type
        self.bias = bias
        self.connection_weights = np.array(connection_weights)
        self.activation_function = activation_function

    def compute_output(self, x):
        '''
        Linear combination of input values passed into activation function.
        '''
        combined_input = self.bias + self.connection_weights.dot(x)
        return self.activation_function(combined_input)

class Layer():
    def __init__(self, layer_type, num_neurons=None, layer_biases=None,
                 layer_weights=None, activation_function=None):

        # Defaul Initializations
        if num_neurons is None:
            num_neurons = 1
        if layer_biases is None:
            layer_biases = [None] * num_neurons
        if layer_weights is None:
            layer_weights = [None] * num_neurons

        # Checks
        if len(layer_biases) != num_neurons:
            raise ValueError(
                'Mismatch between number of neurons and biases')
        if len(layer_weights) != num_neurons:
            raise ValueError(
                'Mismatch between number of neurons and weights')

        # Attribute Initializations
        self.num_neurons = num_neurons
        self.layer_biases = layer_biases
        self.layer_weights = layer_weights
        self.activation_function = activation_function
        self.neurons = [
            Neuron(layer_type, layer_biases[i], layer_weights[i],
                   activation_function) for i in range(num_neurons)]

    def compute_output(self, x):
        # Checks
        if len(x) != self.num_neurons:
            raise ValueError(
                'Input dimensionality does not match number of neurons')

        return [self.neurons[i].compute_output(x[i])
                for i in range(self.num_neurons)]
