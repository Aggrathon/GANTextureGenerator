import tensorflow as tf


def weight_bias(name, shape, stddev=0.02, const=0.1):
	"""Create Weight and Bias tensors for a layer in the nn"""
	w_var = tf.Variable(tf.truncated_normal(shape, stddev=stddev, name=name+'_w'))
	b_var = tf.Variable(tf.constant(const, shape=shape[-1:], name=name+"_b"))
	return w_var, b_var

def conv2d(in_tensor, weight, bias):
	"""Create a convolutional layer"""
	conv = tf.nn.conv2d(in_tensor, weight, [1, 1, 1, 1], "SAME")
	relu = tf.nn.relu(tf.nn.bias_add(conv, bias))
	return tf.nn.avg_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def relu_dropout(in_tensor, weight, bias, dropout):
	"""Create a relu layer with dropout"""
	relu = tf.nn.relu(tf.matmul(in_tensor, weight) + bias)
	dropout = tf.nn.dropout(relu, dropout)
	return dropout

def conv2d_transpose(in_tensor, weight, bias, out_shape):
	"""Create a transpose convolutional layer"""
	deconv = tf.nn.conv2d_transpose(in_tensor, weight, out_shape, [1, 2, 2, 1])
	#Batch norm before relu?
	return tf.nn.relu(tf.nn.bias_add(deconv, bias))

def filter_bias(name, shape, stddev=0.02, const=0.1):
	"""Create Filter and Bias tensors for a conv2d-transpose layer"""
	w_var = tf.Variable(tf.truncated_normal(shape, stddev=stddev, name=name+'_w'))
	b_var = tf.Variable(tf.constant(const, shape=shape[-2:-1], name=name+"_b"))
	return w_var, b_var
