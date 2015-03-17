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
	
	def __init__(self, network, get_data_function, input_data, i):
		Neuron.__init__(self, network)
		# Input neurons use a function that simply fetches data.
		self.get_data_function = get_data_function
		# Keep a copy of the input data.
		self.input_data = input_data
		# The number of this input neuron.
		self.i = i
	
	def __str__(self):
		return "InputNeuron%d" % self.i
	
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
			self.network.E[(i_n, self)] += delta_E_n

class OutputNeuron(Neuron):
	"""An output neuron in the neural network"""
	
	def __init__(self, network, activation_function, target_data, k):
		Neuron.__init__(self, network)
		# Activation function of this output neuron (identity function).
		self.activation_function = activation_function
		# Keep a copy of the target values (to compute the delta).
		self.target_data = target_data
		# The number of this input neuron.
		self.k = k
		# Store the delta so that it can be used in back-propagation.
		self.current_delta = 0
	
	def __str__(self):
		return "OutputNeuron%d" % self.k
	
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
			self.network.E[(h_n, self)] += delta_E_n

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
	def __init__(self, filename, D, M, K):
		# Load data.
		self.load(filename)
		
		# Set the dimensions.
		self.D = D
		self.M = M
		self.K = K
		
		# Extract the first D columns of the data.
		input_data = transposeList(transposeList(self.data)[:self.D])
		# Create D input neurons (plus one bias).
		bias_input = InputNeuron(self, inputBiasFunction, [], 0)
		self.input_neurons = [bias_input] + [InputNeuron(self, inputFunction, input_data, i+1) for i in range(D)]
		
		# Create M hidden neurons (plus one bias).
		bias_hidden = HiddenNeuron(self, hiddenBiasFunction, hiddenBiasFunctionDeriv, 0)
		self.hidden_neurons = [bias_hidden] + [HiddenNeuron(self, hiddenFunction, hiddenFunctionDeriv, j+1) for j in range(M)]
		
		# Extract the last K columns of the data.
		target_data = transposeList(transposeList(self.data)[self.K:])
		# Create K output neurons.
		self.output_neurons = [OutputNeuron(self, outputFunction, target_data, k+1) for k in range(K)]
		
		# The weights of this network.
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
		
		# E is a dictionary that maps pairs of neurons to partial derivatives.
		self.E = collections.OrderedDict()
	
	# Load the given dataset.
	def load(self, filename):
		self.data = [map(float, line.strip().split(' ')) for line in open(filename)]
		self.N = len(self.data)
	
	def runNetwork(self):
		
		# Initialise error to zero.
		self.error = 0.0
		
		# Initialise the partial derivatives to zero. Each iteration adds to
		# the derivatives.
		for k in self.w.iterkeys():
			self.E[k] = 0.0
		
		for n in range(self.N):
			# Run the network on each input pattern.
			self.runSingleData(n)
			
			# Group the deltas for 1..K in a vector.
			deltas = []
			for o_n in self.output_neurons:
				deltas.append(o_n.current_delta)
			# Compute the squared error.
			self.error += np.linalg.norm(np.array(deltas)) ** 2
		
		# Since each iteration added to the derivatives, divide by N to
		# compute the mean.
		for k in self.E.iterkeys():
			self.E[k] /= self.N
		
		# Since we added a squared term in each iteration, we need to divide
		# by N to get the mean squared error.
		self.error /= 2.0 * self.N
		
# 		# Aggregate (i.e., calculate the mean) the gradients for the N patterns.
# 		for k in self.E[0].iterkeys():
# 			self.E_aggr[k] = 0
# 		for n in range(self.N):
# 			for k,v in self.E[n].iteritems():
# 				self.E_aggr[k] += v
# 		for k,v in self.E_aggr.iteritems():
# 			self.E_aggr[k] /= self.N
		
# 		print "E:"
# 		for (n1,n2),v in self.E.iteritems():
# 			print "(" + str(n1) + ", " + str(n2) + "): " + str(v)
# 		
# 		print "w:"
# 		for (n1,n2),v in self.w.iteritems():
# 			print "(" + str(n1) + ", " + str(n2) + "): " + str(v)
# 		
# 		print "error = %0.2f" % self.error
	
	# Run the network on the n'th input pattern.
	def runSingleData(self, n):
# 		# Do forward-propagation and back-propagation.
		self.forwardPropagate(n)
		self.backPropagate(n)
	
	# Forward-propagate the inputs (and activations). Each layer computes its
	# output values so that it can be used as input to the next layer.
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

def run():
	# One-dimensional inputs.
	D = 1
	# Some number of hidden neurons
	M = 2
	# One-dimensional output.
	K = 1
	# Create neural network.
	nn = NeuralNetwork(os.path.dirname(__file__) + '/../../data/sincTrain25.dt', D, M, K)
	# Run the network to produce the partial derivatives.
	nn.runNetwork()
	derivatives = nn.E
	# Numerically estimate the partial derivatives.
	estimates = nn.computeNumericalEstimates()
	print "Partial derivatives, computed with backpropagation and numerical estimation."
	for (n1,n2) in derivatives.iterkeys():
		print "(" + str(n1) + ", " + str(n2) + "):"
		print "\t" + str(derivatives[(n1,n2)])
		print "\t" + str(estimates[(n1,n2)])

if __name__ == '__main__':
	run()