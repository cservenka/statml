import random
import math
import os

class Neuron:
	def __init__(self, transfer_function):
		self.transfer_function = transfer_function
		self.inputs = []

	def add_input(self, weight, neuron):
		self.inputs.append((weight, neuron))

	def get_inputs(self):
		return self.inputs

	def set_inputs(self, inputs):
		self.inputs = inputs

	def run(self):
		s = 0
		for (w, n) in self.inputs:
			s += n.run() * w
		return self.transfer_function(s)

	def __str__(self):
		return "Neuron\tInputs: %d\n" % len(self.inputs)

class NeuralNetwork:
	def input_function(self, a):
		return self.data[self.index][0]

	def hidden_function(self, a):
		return (1 / (1 + math.fabs(a)))

	def output_function(self, a):
		return a

	def bias_function(self, a):
		return 1.0

	def load(self, filename):
		self.data = [map(float, line.strip().split(' ')) for line in open(filename)]

	def __init__(self, filename, D, M, K):
		self.load(filename)
		self.index = 0
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

	def run(self, i):
		self.index = i
		result = []
		for o_n in self.output_neurons:
			result.append(o_n.run())
		return result

	def __str__(self):
		output = "Input neurons: \n\t" + "\t".join(map(str, self.input_neurons))
		output += "Hidden neurons: \n\t" + "\t".join(map(str, self.hidden_neurons))
		output += "Output neurons: \n\t" + "\t".join(map(str, self.output_neurons))
		return output

if __name__ == '__main__':
	nn = NeuralNetwork(os.path.dirname(__file__) + '/../../data/sincTrain25.dt', 1, 10, 1)
	print nn
	print nn.run(2)