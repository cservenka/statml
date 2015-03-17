from __future__ import division

import numpy as np
import random
import math
import os
import collections

class Neuron:
	"""Models a single neuron in the neural network"""
	
	def __init__(self, network):
		# Neurons have access to the network (so that they can compute and
		# set the partial derivatives).
		self.network = network
		# Neurons know their input neurons (possibly empty).
		self.inputs = []
		# Neurons produce an output value.
		self.current_output_value = 0.0
	
	# Adds a neuron to the input list.
	def addInput(self, neuron):
		self.inputs.append(neuron)

class InputNeuron(Neuron):
	"""An input neuron in the neural network"""
	
	def __init__(self, network, get_data_function, i):
		Neuron.__init__(self, network)
		# Input neurons use a function that simply fetches data.
		self.get_data_function = get_data_function
		# The number of this input neuron.
		self.i = i
	
	def __str__(self):
		return "InputNeuron%d" % self.i
	
	def setInputData(self, input_data):
		# Keep a copy of the input data.
		self.input_data = input_data
	
	# Read the input data and set it as the output of this neuron.
	def computeOutputValue(self, n):
		self.current_output_value = self.get_data_function(
														self.input_data,
														self.i-1,
														n)

class HiddenNeuron(Neuron):
	"""A hidden neuron in the neural network"""
	
	def __init__(self, network, activation_function, activation_function_deriv, j):
		Neuron.__init__(self, network)
		# Activation function and derivative of the activation function.
		self.activation_function = activation_function
		self.activation_function_deriv = activation_function_deriv
		# Hidden neurons know their outputs.
		self.outputs = []
		# Store the a_j when computing the activation since it is re-used
		# when computing the delta and the derivatives.
		self.current_a = 0.0
		# The number of this hidden neuron.
		self.j = j
	
	def __str__(self):
		return "HiddenNeuron%d" % self.j
	
	# Adds a neuron to the input list.
	def addOutput(self, neuron):
		self.outputs.append(neuron)
	
	# Compute the activation of this hidden neuron.
	def computeOutputValue(self):
		# Sum up the weighted inputs.
		self.current_a = 0.0
		for i_n in self.inputs:
			weight = self.network.w[(i_n, self)]
			self.current_a += i_n.current_output_value * weight
		
		# Apply the activation function.
		self.current_output_value = self.activation_function(self.current_a)
	
	# Compute the delta (back-propagation) and use it to compute the partial
	# derivatives of the connections (weights) from the input neurons.
	def computeDerivatives(self, n):
		# Sum up the weighted deltas.
		current_delta = 0.0
		for o_n in self.outputs:
			weight = self.network.w[(self, o_n)]
			current_delta += o_n.current_delta * weight
		# Multiply by h'(a).
		current_delta *= self.activation_function_deriv(self.current_a)
		
		# Use delta to compute derivatives.
		for i_n in self.inputs:
			delta_E_n = current_delta * i_n.current_output_value
			self.network.dE[(i_n, self)] += delta_E_n

class OutputNeuron(Neuron):
	"""An output neuron in the neural network"""
	
	def __init__(self, network, activation_function, k):
		Neuron.__init__(self, network)
		# Activation function of this output neuron (identity function).
		self.activation_function = activation_function
		# The number of this input neuron.
		self.k = k
		# Store the delta so that it can be used in back-propagation.
		self.current_delta = 0
	
	def __str__(self):
		return "OutputNeuron%d" % self.k
	
	def setTargetData(self, target_data):
		# Keep a copy of the target values (to compute the delta).
		self.target_data = target_data
	
	# Compute the activation (i.e., output value) of this output neuron.
	def computeOutputValue(self):
		# Sum up the weighted inputs.
		a = 0.0
		for h_n in self.inputs:
			weight = self.network.w[(h_n, self)]
			a += h_n.current_output_value * weight
		
		# Activation function is just the identity.
		self.current_output_value = self.activation_function(a)
	
	# Compute the delta (and store it), and use it to compute the partial
	# derivatives of the connections (weights) from the hidden neurons.
	def computeDerivatives(self, n):
		t = self.target_data[n][self.k-1]
		self.current_delta = self.current_output_value - t
		
		# Use delta to compute derivatives.
		for h_n in self.inputs:
			delta_E_n = self.current_delta * h_n.current_output_value
			self.network.dE[(h_n, self)] += delta_E_n

class NeuralNetwork:
	"""Models a multi-layer neural network"""
	
	def __str__(self):
		output = "Input neurons: \n\t" + "\t".join(map(str, self.input_neurons))
		output += "\nHidden neurons: \n\t" + "\t".join(map(str, self.hidden_neurons))
		output += "\nOutput neurons: \n\t" + "\t".join(map(str, self.output_neurons))
		return output
	
	# Initialise the network with the given dataset.
	# D is the number of input nodes (should always be 1).
	# M is the number of hidden nodes.
	# K is the number of output nodes (should always be 1).
	# Bias nodes are automatically added.
	def __init__(self, D, M, K):
		# Set the dimensions.
		self.D = D
		self.M = M
		self.K = K
		
		# Create D input neurons (plus one bias).
		bias_input = InputNeuron(self, inputBiasFunction, 0)
		self.input_neurons = [bias_input] + [InputNeuron(self, inputFunction, i+1) for i in range(D)]
		
		# Create M hidden neurons (plus one bias).
		bias_hidden = HiddenNeuron(self, hiddenBiasFunction, hiddenBiasFunctionDeriv, 0)
		self.hidden_neurons = [bias_hidden] + [HiddenNeuron(self, hiddenFunction, hiddenFunctionDeriv, j+1) for j in range(M)]
		
		# Create K output neurons.
		self.output_neurons = [OutputNeuron(self, outputFunction, k+1) for k in range(K)]
		
		# The weights of the network, using a dictionary that maps pairs of
		# neurons to weights.
		self.w = collections.OrderedDict()
		
		# Set up connections from input neurons to hidden neurons.
		for i_n in self.input_neurons:
			for h_n in self.hidden_neurons[1:]: # Excluding bias since it has no incoming connections.
				# Initially, set the weight to some arbitrary (random) value.
				weight = random.uniform(-1.0, 1.0)
				# Associate the weight with this pair of neurons.
				self.w[(i_n, h_n)] = weight
				# Hidden neurons know their input neurons.
				h_n.addInput(i_n)
		
		# Set up connections from hidden neurons to output neurons.
		for h_n in self.hidden_neurons:
			for o_n in self.output_neurons:
				# Initially, set the weight to some arbitrary (random) value.
				weight = random.uniform(-1.0, 1.0)
				# Associate the weight with this pair of neurons.
				self.w[(h_n, o_n)] = weight
				# Output neurons know their input neurons (which are the
				# hidden neurons).
				o_n.addInput(h_n)
				# Hidden neurons also know their output neurons (in order to
				# back-propagate deltas).
				h_n.addOutput(o_n)
		
		# The partial derivatives of the network, using a dictionary that maps
		# pairs of neurons to partial derivatives.
		self.dE = collections.OrderedDict()
	
	# Load the given dataset.
	def load(self, dataset):
		self.data = dataset
		self.N = len(self.data)
		
		# Extract the first D columns of the data.
		input_data = transposeList(transposeList(self.data)[:self.D])
		for i_n in self.input_neurons:
			i_n.setInputData(input_data)
		
		# Extract the last K columns of the data.
		target_data = transposeList(transposeList(self.data)[self.K:])
		for o_n in self.output_neurons:
			o_n.setTargetData(target_data)
	
	# Run the network on the loaded dataset.
	def runNetwork(self):
		
		# Initialise error to zero.
		self.error = 0.0
		
		# Initialise the partial derivatives to zero. Each iteration adds to
		# the derivatives.
		for k in self.w.iterkeys():
			self.dE[k] = 0.0
		
		self.outputs = []
		
		# Iterate over the input patterns.
		for n in range(self.N):
			# Run the network on each input pattern.
			self.runSingleData(n)
			
			outputs_n = []
			
			# Compute the error and output.
			deltas = []
			for o_n in self.output_neurons:
				outputs_n.append(o_n.current_output_value)
				deltas.append(o_n.current_delta)
			self.error += np.linalg.norm(np.array(deltas)) ** 2
			self.outputs.append(outputs_n)
		
		# Since we added the squared error in each iteration, we need to divide
		# by N to get the mean-squared error.
		self.error *= 1.0 / self.N
		
		# Since each iteration added to the derivatives, divide by N to
		# compute the mean. We also multiply by 2 since each term that we take
		# derivatives over should be a squared term.
		for k in self.dE.iterkeys():
			self.dE[k] *= 2.0 / self.N

	# Run the network on the n'th input pattern.
	def runSingleData(self, n):
		# Do forward-propagation and back-propagation.
		self.forwardPropagate(n)
		self.backPropagate(n)
	
	# Forward-propagate the inputs (and activations). Each layer computes its
	# output values so that the values can be used as inputs to the next layer.
	def forwardPropagate(self, n):
		for i_n in self.input_neurons:
			i_n.computeOutputValue(n)
		
		for h_n in self.hidden_neurons:
			h_n.computeOutputValue()
		
		for o_n in self.output_neurons:
			o_n.computeOutputValue()
	
	# Back-propagate the deltas and compute the partial derivatives.
	def backPropagate(self, n):
		for o_n in self.output_neurons:
			o_n.computeDerivatives(n)
		
		for h_n in self.hidden_neurons:
			h_n.computeDerivatives(n)
	
	# Compute the numerical estimates of the partial derivatives.
	def computeNumericalEstimates(self):
		estimates = collections.OrderedDict()
		
		epsilon = 0.00000001
		
		# Run the network to produce an error value.
		self.runNetwork()
		errorWithoutEpsilon = self.error
		
		for k in self.w.iterkeys():
			# Save old value of the weight.
			old_value = self.w[k]
			# Add epsilon to the weight.
			self.w[k] += epsilon
			# Run the network to produce an error value after adding epsilon.
			self.runNetwork()
			errorWithEpsilon = self.error
			# Restore the old value of the weight.
			self.w[k] = old_value
			# Compute the numerically estimated partial derivative.
			estimates[k] = (errorWithEpsilon - errorWithoutEpsilon) / epsilon
		
		return estimates

# The bias parameter that the bias neurons output.
bias_parameter = 1.0

# The input neurons simply read data.
def inputFunction(data, i, n):
	return data[n][i]

# The bias input neuron simply returns the bias parameter.
def inputBiasFunction(data, i, n):
	return bias_parameter

# The activation function of the hidden neurons.
def hiddenFunction(a):
	return (a / (1.0 + math.fabs(a)))

# The derivative of the activation function of the hidden neurons.
def hiddenFunctionDeriv(a):
	return (1.0 / ((1.0 + math.fabs(a)) ** 2))

# The bias hidden neuron simply returns the bias parameter.
def hiddenBiasFunction(a):
	return bias_parameter

# The derivative of a constant (the bias parameter) is zero.
def hiddenBiasFunctionDeriv(a):
	return 0.0

# The activation function of the output neurons is the identity function.
def outputFunction(a):
	return a

# Transpose a list of lists.
def transposeList(lst):
	return (np.array(lst).T).tolist()

# Load the given dataset.
def readFile(filename):
	return [map(float, line.strip().split(' ')) for line in open(filename)]

def run():
	train = readFile(os.path.dirname(__file__) + '/../../data/sincTrain25.dt')
	#test = readFile(os.path.dirname(__file__) + '/../../data/sincValidate10.dt')
	
	# One-dimensional inputs.
	D = 1
	# Some number of hidden neurons
	M = 2
	# One-dimensional output.
	K = 1
	# Create neural network.
	nn = NeuralNetwork(D, M, K)
	# Load data.
	nn.load(train)
	# Run the network to produce the partial derivatives.
	nn.runNetwork()
	derivatives = nn.dE
	# Numerically estimate the partial derivatives.
	estimates = nn.computeNumericalEstimates()
	print "Partial derivatives"
	for (n1,n2) in derivatives.iterkeys():
		print "(" + str(n1) + ", " + str(n2) + "):"
		print "\t" + str(derivatives[(n1,n2)])
		print "\t" + str(estimates[(n1,n2)])

if __name__ == '__main__':
	run()