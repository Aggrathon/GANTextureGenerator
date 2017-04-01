import os
import tensorflow as tf
import numpy as np
from image import save_image
from operators import weight_bias, conv2d_transpose, relu_dropout, filter_bias
from config import GeneratorConfig


class Generator():

    @classmethod
    def from_config(cls, name, config=None):
        """Create a generator from a config object"""
        if config is None:
            return Generator.from_config(name, GeneratorConfig())
        return Generator(
            name,
            config.image_size,
            config.colors,
            config.expand_layers,
            config.conv_layers,
            config.conv_size,
            config.input_size,
            config.batch_size
        )

    def __init__(self, name, image_size=32, colors=1,
                 expand_layers=2, conv_layers=3, conv_size=32, input_size=128, batch_size=64):
        self.name = name
        self.image_size = image_size
        self.input_size = input_size
        self.input = tf.placeholder(tf.float32, [None, input_size])
        self.theta = [] #Trainable variables

        #Network layer variables
        prev_layer = self.input
        expand_layer_size = input_size
        conv_image_size = image_size // (2**conv_layers)
        assert conv_image_size*(2**conv_layers) == image_size, \
                "Images must be a multiple of two (or at least divisible by 2**num_of_conv_layers_plus_one)"
        #Pre conv layers
        with tf.name_scope('Generator') as scope:
            for i in range(expand_layers):
                if i != expand_layers -1:
                    next_layer = int(input_size+(i+1)/expand_layers*(conv_image_size*conv_image_size*colors - input_size))
                else:
                    next_layer = conv_image_size*conv_image_size*colors
                w, b = weight_bias('expand_layer%d'%i, [expand_layer_size, next_layer], 0.1, 0.1)
                expand_layer_size = next_layer
                prev_layer = tf.nn.relu(tf.matmul(prev_layer, w) + b)
                self.theta.extend((w, b))
            #Conv layers
            prev_layer = tf.reshape(prev_layer, [-1, conv_image_size, conv_image_size, colors])
            generate_layer = prev_layer
            for i in range(conv_layers):
                size = conv_size*(conv_layers-i) if i < conv_layers-1 else colors
                prev_size = conv_size*(conv_layers-i+1) if i != 0 else colors
                w, b = filter_bias('convtr_layer%d'%i, [5, 5, size, prev_size], 0.1, 0.1)
                conv_image_size *= 2
                prev_layer = conv2d_transpose(prev_layer, w, b, [batch_size, conv_image_size, conv_image_size, size])
                generate_layer = conv2d_transpose(generate_layer, w, b, [1, conv_image_size, conv_image_size, size])
                self.theta.extend((w, b))
            self.output = tf.nn.tanh(prev_layer)*255
            self.generate_output = tf.nn.tanh(generate_layer)*255


    def random_input(self, n=1):
        return np.random.uniform(-1., 1., size=[n, self.input_size])


    def generate(self, session, amount=1, name=None):
        #Prepare Images
        fullname = self.name
        if not name is None:
            fullname += '-'+name
        images = session.run(self.generate_output, feed_dict={self.input: self.random_input(amount)})
        #Save
        if amount == 1:
            save_image(images, self.image_size, fullname)
        else:
            for i in range(amount):
                save_image(images[i], self.image_size, fullname+"-"+str(i))

