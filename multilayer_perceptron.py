import numpy as np

# class MultilayerPerceptron():

class Neuron():
    def __init__(self, neuron_type, bias=None, connection_weights=None,
                 activation_function=None):
        # Checks and default initializations
        ## Default: zero bias, identity weights, identity activation for
        ##  input and output neurons, step activation for hidden neurons.
        if neuron_type not in ['input', 'hidden', 'output']:
            raise ValueError('Unknown Neuron Type %s' %(neuron_type,))
        if bias is None:
            bias = 0
        if connection_weights is None or neuron_type == 'input':
            connection_weights = [1]
        if neuron_type != 'hidden':
            identity = lambda x: x
            activation_function = identity
        if activation_function is None and neuron_type == 'hidden':
            step_function = lambda x: -1 if x < 0 else 1
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
