import tensorflow as tf


def weight_bias(name, shape, stddev=0.02, const=0.1):
	"""Create Weight and Bias tensors for a layer in the nn"""
	w_var = tf.Variable(tf.truncated_normal(shape, stddev=stddev, name=name+'_w'))
	b_var = tf.Variable(tf.constant(const, shape=shape[-1], name=name+"_b"))
	return w_var, b_var

def conv2d(input, weight, bias):
	"""Create a convolutional layer"""
	conv = tf.nn.conv2d(input, weight, [1, 1, 1, 1], "SAME")
	relu = tf.nn.relu(conv + bias)
	return tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def relu_dropout(input, weight, bias, dropout):
	"""Create a relu layer with dropout"""
	relu = tf.nn.relu(tf.matmul(input, weight) + bias)
	dropout = tf.nn.dropout(relu, dropout)
	return dropout
