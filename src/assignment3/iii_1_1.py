from __future__ import division

import random
import math
import os

class Neuron:
	"""Models a single neuron in the neural network"""
	
	def __init__(self, transfer_function):
		self.transfer_function = transfer_function
		self.inputs = []

    #Adds a node to the input list with a given weight
	def add_input(self, weight, neuron):
		self.inputs.append((weight, neuron))

	def get_inputs(self):
		return self.inputs

	def set_inputs(self, inputs):
		self.inputs = inputs

    #This function polls the neurons input for data in a recursive manner
    #To run the whole network simply call this method on one of the output neurons, the call will then propagate backwards
    #If this neuron is an input neuron, it should have an empty list of inputs so the result will just be transfer_function(0)
	def run(self):
		s = 0
		for (w, n) in self.inputs:
			s += n.run() * w
		return self.transfer_function(s)

	def __str__(self):
		return "Neuron\tInputs: %d\n" % len(self.inputs)

class NeuralNetwork:
	"""Models a multi-layer neural network"""

	#Transfer function for input nodes, a will always be 0 but we ignore it and simply return the data of the current pattern
	def input_function(self, a):
		return self.data[self.index][0]

	#Transfer function for hidden nodes - sigmoidal
	def hidden_function(self, a):
		return (1 / (1 + math.fabs(a)))

	#Transfer function for output nodes - identity function
	def output_function(self, a):
		return a

	#Transfer function for bias nodes, again a will always be 0 but we ignore it and return our bias parameter instead
	def bias_function(self, a):
		return 1.0

	#Load the given dataset
	def load(self, filename):
		self.data = [map(float, line.strip().split(' ')) for line in open(filename)]
		self.N = len(self.data)

	#Initialise the network with the given dataset
	#D is the number of input nodes (should always be 1)
	#M is the number of hidden nodes
	#K is the number of output nodes (should always be 1)
	#Bias nodes are automatically added
	def __init__(self, filename, D, M, K):
		self.load(filename)
		self.D = D
		self.M = M
		self.K = K
		self.input_neurons = [Neuron(self.input_function) for _ in xrange(D)]
		self.hidden_neurons = [Neuron(self.hidden_function) for _ in xrange(M)]
		self.output_neurons = [Neuron(self.output_function) for _ in xrange(K)]

		bias_hidden = Neuron(self.bias_function)
		bias_output = Neuron(self.bias_function)

		self.input_neurons.append(bias_hidden)
		self.hidden_neurons.append(bias_output)

		for h_n in self.hidden_neurons:
			h_n.set_inputs([(random.uniform(-1.0, 1.0), n) for n in self.input_neurons])

		for o_n in self.output_neurons:
			o_n.set_inputs([(random.uniform(-1.0, 1.0), n) for n in self.hidden_neurons])

	#Run the network on the i'th data point
	def run(self, i):
		self.index = i
		result = []
		for o_n in self.output_neurons:
			result.append(o_n.run())
		return result

	#Run the network on the i'th data point and get the error value
	# ********* Probably not correct ***********
	def get_error_n(self, i):
		result = self.run(i)
		error = 0
		for k in xrange(self.K):
			y_i = result[k]
			t_i = self.data[i][1]
			error += (1 / self.N) * (y_i - t_i) ** 2
		return error

	def __str__(self):
		output = "Input neurons: \n\t" + "\t".join(map(str, self.input_neurons))
		output += "Hidden neurons: \n\t" + "\t".join(map(str, self.hidden_neurons))
		output += "Output neurons: \n\t" + "\t".join(map(str, self.output_neurons))
		return output

if __name__ == '__main__':
	nn = NeuralNetwork(os.path.dirname(__file__) + '/../../data/sincTrain25.dt', 1, 10, 1)
	print nn.get_error_n(0)