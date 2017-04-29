
import tensorflow as tf
import numpy as np

from operators import linear, conv2d, relu_dropout, conv2d_transpose, conv2d_transpose_tanh, expand_relu


def image_encoder(input_tensors, name='encoder', image_size=64, convolutions=5, base_width=32,
                  fully_connected=1, dropout=0.4, output_size=1):
    """Create a network for reducing an image to a 1D tensor"""
    conv_output_size = ((image_size//(2**convolutions))**2) * base_width * convolutions
    assert conv_output_size != 0, "Invalid number of convolutions compared to the image size"
    fc_output_size = max(2**int(np.log2(conv_output_size//2)), output_size)
    with tf.variable_scope(name):
        #Create Layers
        prev_layer = input_tensors
        for i in range(convolutions): #Convolutional layers
            prev_layer = conv2d(prev_layer, base_width*(i+1), name='convolution_%d'%i, norm=(i != 0))
        prev_layer = [tf.reshape(layer, [-1, conv_output_size]) for layer in prev_layer]
        for i in range(fully_connected): #Fully connected layers
            prev_layer = relu_dropout(prev_layer, fc_output_size, dropout, 'fully_connected_%d'%i)
        prev_layer = linear(prev_layer, output_size, 'output')
    return prev_layer

def image_decoder(input_tensors, name='decoder', image_size=64, convolutions=5, base_width=32,
                  input_size=64, batch_size=128, colors=3):
    """Create a network for generating an image from a 1D tensor"""
    conv_image_size = image_size // (2**convolutions)
    assert conv_image_size*(2**convolutions) == image_size, "Images must be a multiple of two (and >= 2**convolutions)"
    with tf.variable_scope(name):
        prev_layer = expand_relu(input_tensors, [-1, conv_image_size, conv_image_size, base_width*2**(convolutions-1)], 'expand')
        for i in range(convolutions-1):
            prev_layer = conv2d_transpose(prev_layer, batch_size, 2**(convolutions-i-2)*base_width, 'convolution_%d'%i)
        prev_layer = conv2d_transpose_tanh(prev_layer, batch_size, colors, 'output')
    return prev_layer

def image_output(input_tensors, name='output', image_size=64, grid_size=4):
    output = []
    with tf.name_scope(name):
        with tf.name_scope("image_single"):
            output = [tf.cast((tensor + 1) * 127.5, tf.uint8) for tensor in input_tensors]
        with tf.name_scope('image_grid'):
            for i in range(len(input_tensors)):
                wh = grid_size * (image_size + 2) + 2
                grid = tf.Variable(0, trainable=False, dtype=tf.uint8, expected_shape=[wh, wh, image_size])
                for x in range(grid_size):
                    for y in range(grid_size):
                        bound = tf.to_int32(tf.image.pad_to_bounding_box(output[i][x+y*grid_size], 2 + x*(image_size + 2), 2 + y*(image_size + 2), wh, wh))
                        if x == 0 and y == 0:
                            grid = bound
                        else:
                            grid = tf.add(grid, bound)
                output.append(tf.cast(grid, tf.uint8))
    return output
