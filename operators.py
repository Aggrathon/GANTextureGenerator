import tensorflow as tf
import numpy as np


# Helper Functions

def lerp_int(value_a: int, value_b: int, perc: float, epsilon :float=0.01):
	"""Returns a lerp from the minimum to the maximum (inclusive int)"""
	if perc < epsilon:
		return value_a
	elif perc >= 1-epsilon:
		return value_b
	else:
		return int(value_a + perc*(value_b-value_a))

class WithNone():
	def __enter__(self):
		return self
	def __exit__(self, type, value, traceback):
		pass

def openif_scope(name=None):
	if name is None:
		return WithNone()
	else:
		return tf.variable_scope(name)


# Variable Creation

def weight_bias(shape, stddev: float=0.02, const: float=0.01, summary: bool=True):
	"""Create Weight and Bias tensors for a layer in the nn"""
	w_var = tf.get_variable('weight', shape, tf.float32, tf.random_normal_initializer(0, stddev), trainable=True)
	b_var = tf.get_variable('bias', [shape[-1]], tf.float32, tf.constant_initializer(const,), trainable=True)
	if summary:
		tf.summary.histogram('weight', w_var)
		tf.summary.histogram('bias', b_var)
	return w_var, b_var

def filter_bias(shape, stddev: float=0.02, const: float=0.1, summary: bool=True):
	"""Create Filter and Bias tensors for a conv2d-transpose layer"""
	w_var = tf.get_variable('filter', shape, tf.float32, tf.random_normal_initializer(0, stddev), trainable=True)
	b_var = tf.get_variable('bias', [shape[-2]], tf.float32, tf.constant_initializer(const,), trainable=True)
	if summary:
		tf.summary.histogram('bias', b_var)
	return w_var, b_var


# Operator Functions

def lrelu(tensor, leak: float=0.2):
	"""Create a leaky relu node"""
	return tf.maximum(tensor, tensor*leak, name='lrelu')


# Network Layers

def conv2d(tensors, output_size: int, name: str='conv2d', stddev: float=0.02, term: float=0.01, summary: bool=True):
	"""Create a convolutional layer"""
	with tf.variable_scope(name):
		weight, bias = weight_bias([5, 5, int(tensors[0].get_shape()[-1]), output_size], stddev, term, summary)
		output = []
		for i, tensor in enumerate(tensors):
			conv = tf.nn.conv2d(tensor, weight, [1, 2, 2, 1], "SAME")
			conv = tf.contrib.layers.batch_norm(conv, decay=0.9, updates_collections=None, scale=False,
				trainable=True, reuse=(i!=0), scope="normalization", is_training=True, epsilon=0.00001)
			output.append(lrelu(tf.nn.bias_add(conv, bias)))
		return output

def relu(tensor, output_size: int, name: str='relu', stddev: float=0.02, term: float=0.01, summary: bool=True):
	"""Create a relu layer"""
	with tf.variable_scope(name):
		weight, bias = weight_bias([int(tensor.get_shape()[-1]), output_size], stddev, term, summary)
		return tf.nn.relu(tf.matmul(tensor, weight) + bias)

def relu_dropout(tensors, output_size: int, dropout: float=0.4, name: str='relu_dropout', stddev: float=0.02, term: float=0.01, summary: bool=True):
	"""Create a relu layer with dropout"""
	with tf.variable_scope(name):
		weight, bias = weight_bias([int(tensors[0].get_shape()[-1]), output_size], stddev, term, summary)
		output = []
		for tensor in tensors:
			relu_layer = tf.nn.relu(tf.matmul(tensor, weight) + bias)
			output.append(tf.nn.dropout(relu_layer, dropout))
		return output

def linear(tensors, output_size: int, name: str='linear', stddev: float=0.02, term: float=0.01, summary: bool=True):
	'''Create a fully connected layer'''
	with tf.variable_scope(name):
		weight, bias = weight_bias([tensors[0].get_shape()[-1], output_size], stddev, term, summary)
		return [tf.matmul(tensor, weight) + bias for tensor in tensors]

def conv2d_transpose(tensors, batch_size=1, conv_size=32, name: str='conv2d_transpose', stddev: float=0.02, term: float=0.01, summary: bool=True):
	"""Create a transpose convolutional layer"""
	with tf.variable_scope(name):
		tensor_shape = tensors[0].get_shape()
		filt, bias = filter_bias([5, 5, conv_size, tensor_shape[-1]], stddev, term, summary)
		conv_shape = [batch_size, int(tensor_shape[1]*2), int(tensor_shape[2]*2), conv_size]
		output = []
		for i, tensor in enumerate(tensors):
			deconv = tf.nn.conv2d_transpose(tensor, filt, conv_shape, [1, 2, 2, 1])
			deconv = tf.contrib.layers.batch_norm(deconv, decay=0.9, updates_collections=None, scale=False,
				trainable=True, reuse=(i!=0), scope="normalization", is_training=True, epsilon=0.00001)
			output.append(tf.nn.relu(tf.nn.bias_add(deconv, bias)))
		return output

def conv2d_transpose_tanh(tensors, batch_size=1, conv_size=32, name: str='conv2d_transpose_tanh', stddev: float=0.02, summary: bool=True):
	"""Create a transpose convolutional layer"""
	with tf.variable_scope(name):
		tensor_shape = tensors[0].get_shape()
		filt = tf.get_variable('filter', [5, 5, conv_size, tensor_shape[-1]], tf.float32, tf.random_normal_initializer(0, stddev), trainable=True)
		output = []
		for tensor in tensors:
			conv_shape = [batch_size, int(tensor_shape[1]), int(tensor_shape[2]), conv_size]
			deconv = tf.nn.conv2d_transpose(tensor, filt, conv_shape, [1, 1, 1, 1])
			output.append(tf.nn.tanh(deconv))
		return output

def expand_relu(tensors, out_shape, name: str='expand_relu', norm: bool=True, stddev: float=0.2, term: float=0.01, summary: bool=True):
	"""Create a layer that expands an input to a shape"""
	with tf.variable_scope(name) as scope:
		weight, bias = weight_bias([tensors[0].get_shape()[-1], np.prod(out_shape[1:])], stddev, term, summary)
		output = []
		for i, tensor in enumerate(tensors):
			lin = tf.matmul(tensor, weight) + bias
			reshape = tf.reshape(lin, out_shape)
			if norm:
				reshape = tf.contrib.layers.batch_norm(reshape, decay=0.9, updates_collections=None, scale=False,
					trainable=True, reuse=(i!=0), scope=scope, is_training=True, epsilon=0.00001)
			output.append(tf.nn.relu(reshape))
		return output


