import os
import tensorflow as tf
import numpy as np
from image import save_image
from operators import conv2d_transpose, lerp_int, conv2d_transpose_tanh, linear
from config import GeneratorConfig


class Generator():

    @classmethod
    def from_config(cls, config=None):
        """Create a generator from a config object"""
        if config is None:
            return Generator.from_config(GeneratorConfig())
        return Generator(
            config.image_size,
            config.colors,
            config.conv_layers,
            config.conv_size,
            config.input_size,
            config.batch_size,
            config.learning_rate
        )

    def __init__(self, image_size=32, colors=1, conv_layers=3, conv_size=32,
                 input_size=128, batch_size=64, learning_rate=0.001):
        self.image_size = image_size
        self.input_size = input_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.output = None
        self.gen_output = None
        self.loss = None
        self.solver = None
        self.trainable_variables = None

        #Create the network
        with tf.variable_scope('generator') as scope:
            self.scope = scope
            #Network layer variables
            conv_image_size = image_size // (2**conv_layers)
            assert conv_image_size*(2**conv_layers) == image_size, "Images must be a multiple of two (or at least divisible by 2**num_of_conv_layers_plus_one)"
            #Input Layers
            self.input = tf.placeholder(tf.float32, [None, input_size])
            prev_layer = linear(self.input, conv_image_size**2 * conv_size*conv_layers, 'expand', 0.6, 0)
            prev_layer = tf.reshape(prev_layer, [-1, conv_image_size, conv_image_size, conv_size*(conv_layers)])
            #Conv layers
            for i in range(conv_layers-1):
                prev_layer = conv2d_transpose(prev_layer, batch_size, (conv_layers-i-1)*conv_size, 'convolution_%d'%i)
            self.output = conv2d_transpose_tanh(prev_layer, batch_size, colors, 'output', factor=255.0)

    def setup_loss(self, classification_logits):
        """Creates the loss functions and the optimizers when fed a classification tensor"""
        with tf.variable_scope(self.scope, reuse=True):
            self.batch_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=classification_logits, labels=tf.constant(0.9, shape=[self.batch_size, 1]))
            self.loss = tf.reduce_mean(self.batch_loss, name='loss')
            self.trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope.name)
        self.solver = tf.train.AdadeltaOptimizer(self.learning_rate).minimize(self.loss, var_list=self.trainable_variables)


    def random_input(self, n=1):
        """Creates a random input for the generator"""
        return np.random.uniform(0.0, 1.0, size=[n, self.input_size])


    def generate(self, session, name='generated'):
        losses, images = session.run([self.batch_loss, self.output], feed_dict={self.input: self.random_input(self.batch_size)})
        image = images[max(range(self.batch_size), key=lambda i: losses[i])]
        save_image(image, self.image_size, name)

