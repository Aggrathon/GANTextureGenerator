import os
import tensorflow as tf
import numpy as np
from image import save_image
from operators import conv2d_transpose, lerp_int, relu, conv2d_transpose_tanh
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
            config.expand_layers,
            config.conv_layers,
            config.conv_size,
            config.input_size,
            config.batch_size,
            config.learning_rate
        )

    def __init__(self, image_size=32, colors=1, expand_layers=2, conv_layers=3, conv_size=32,
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

        #Network layers
        with tf.variable_scope('generator') as scope:
            self.scope = scope
            #Network layer variables
            self.input = tf.placeholder(tf.float32, [None, input_size])
            prev_layer = self.input
            conv_image_size = image_size // (2**conv_layers)
            assert conv_image_size*(2**conv_layers) == image_size, "Images must be a multiple of two (or at least divisible by 2**num_of_conv_layers_plus_one)"
            expand_layer_in_size = input_size
            expand_layer_out_size = conv_image_size**2 * conv_size*(conv_layers)
            #Pre conv layers
            for i in range(expand_layers):
                prev_layer = relu(prev_layer, lerp_int(expand_layer_in_size, expand_layer_out_size, (i+1)/expand_layers), 'expand_%d'%i)
            prev_layer = tf.reshape(prev_layer, [-1, conv_image_size, conv_image_size, conv_size*(conv_layers)])
            #Conv layers
            genr_layer = prev_layer
            for i in range(conv_layers-1):
                prev_layer = conv2d_transpose(prev_layer, batch_size, (conv_layers-i-1)*conv_size, 'convolution_%d'%i)
            self.output = conv2d_transpose_tanh(prev_layer, batch_size, colors, 'output')
            #Image generating layers
            scope.reuse_variables()
            for i in range(conv_layers-1):
                genr_layer = conv2d_transpose(genr_layer, 1, (conv_layers-i-1)*conv_size, 'convolution_%d'%i)
            self.gen_output = conv2d_transpose_tanh(genr_layer, batch_size, colors, 'output')

    def setup_loss(self, classification_logits):
        with tf.variable_scope(self.scope, reuse=True):
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=classification_logits, labels=tf.constant(0.9, shape=[self.batch_size, 1])), name='loss')
            self.trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope.name)
        self.solver = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, var_list=self.trainable_variables)


    def random_input(self, n=1):
        return np.random.uniform(-1., 1., size=[n, self.input_size])


    def generate(self, session, amount=1, name='generated'):
        #Prepare Images
        images = session.run(self.gen_output, feed_dict={self.input: self.random_input(amount)})
        #Save
        if amount == 1:
            save_image(images, self.image_size, name)
        else:
            for i in range(amount):
                save_image(images[i], self.image_size, name+"-"+str(i))

