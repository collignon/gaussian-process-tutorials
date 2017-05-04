import numpy as np

# class MultilayerPerceptron():

class Neuron():
    '''Computational unit of a Neural Network.'''
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
    '''Layer of Neurons in a Layered Neural Network.'''
    def __init__(self, layer_type, num_neurons=None, layer_biases=None,
                 layer_weights=None, activation_function=None):
        '''
        Params:
        -------
        layer_type: <str> - 'input', 'output' or 'hidden'
        num_neurons: <int> - number of neurons in the layer.
        layer_biases: list(<float>) - bias for each neuron in the layer.
        layer_weights: list(list(<float>)) - list of weights
                        for each neuron in the layer
        activation_function: <float> -> <float>
        '''
        # Default Initializations
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
        self.layer_type = layer_type
        self.num_neurons = num_neurons
        self.layer_biases = layer_biases
        self.layer_weights = layer_weights
        self.activation_function = activation_function
        self.neurons = [
            Neuron(layer_type, layer_biases[i], layer_weights[i],
                   activation_function) for i in range(num_neurons)]

    def compute_output(self, x):
        # Checks
        if self.layer_type == 'input':
            if len(x) != self.num_neurons:
                raise ValueError(
                    'Wrong input dimensionality for input layer')

            return [self.neurons[i].compute_output(x[i])
                    for i in range(self.num_neurons)]

        else:
            return [self.neurons[i].compute_output(x)
                    for i in range(self.num_neurons)]

class MultilayerPerceptron():
    def __init__(self, num_layers, num_neurons, network_biases=None,
                 network_weights=None, activation_function=None):
        '''
        Params:
        -------
        num_layers: <int> - number of layers (>= 2)
        num_neurons: <int> - number of neurons in the layer.
        network_biases: list(list(<float>)) - layer biases for each layer
                    in the network.
        network_weights: list(list(list(<float>))) - list of weights
                        for each layer in the network.
        activation_function: <float> -> <float>
        '''
        # Default Initializations
        if network_biases is None:
            network_biases = [[None] * num_neurons[i]
                              for i in range(1, num_layers)]
        if network_weights is None:
            network_weights = [
                [[None] * num_neurons[i-1]] * num_neurons[i]
                               for i in range(1, num_layers)]

        # Checks
        if len(num_neurons) != num_layers:
            raise ValueError(
                'Mismatch between number of layers and neurons')
        if len(network_biases) != num_layers-1:
            raise ValueError(
                'Mismatch between number of layers and biases')
        if len(network_weights) != num_layers-1:
            raise ValueError(
                'Mismatch between number of layers and weights')

        # Attribute initialization
        self.num_layers = num_layers
        self.num_neurons = num_neurons
        self.network_biases = [None] + network_biases
        self.network_weights = [None] + network_weights
        self.activation_function = activation_function

        layer_types = ['input'] +\
                      ['hidden'] * (self.num_layers-2) +\
                      ['output']
        self.layers = [Layer(
            layer_types[i], self.num_neurons[i], self.network_biases[i],
            self.network_weights[i], self.activation_function)
                       for i in range(self.num_layers)]

    def compute_output(self, x):
        rolling_output = [x]
        for i in range(1, self.num_layers):
            output = self.layers[i].compute_output(
                rolling_output[-1])
            rolling_output.append(output)
        return output
