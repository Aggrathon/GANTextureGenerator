import os
import tensorflow as tf
import numpy as np
from image import save_image
from config import INPUT_SIZE, NETWORK_FOLDER, GEN_DROPOUT, GEN_HIDDEN_LAYERS, IMAGE_SIZE, COLORED
from operators import weight_bias, conv2d_transpose, relu_dropout, filter_bias


class Generator():

    @classmethod
    def from_config(cls, config):
        """Create a generator from a config object"""
        return Generator(
            config.name,
            config.image_size,
            config.colors,
            config.expand_layers,
            config.conv_layers,
            config.conv_size,
            config.input_size,
            config.dropout,
            config.batch_size
        )

    def __init__(self, name, image_size=32, colors=1,
                 expand_layers=2, conv_layers=2, conv_size=32, input_size=128, dropout=0.4, batch_size=64):
        self.name = name+"_generator"
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
        for i in range(expand_layers):
            if i != expand_layers -1:
                next_layer = int(input_size+(i+1)/expand_layers*(conv_image_size*conv_image_size*colors - input_size))
            else:
                next_layer = conv_image_size*conv_image_size*colors
            w, b = weight_bias(self.name+str(i), [expand_layer_size, next_layer], 0.1, 0.1)
            expand_layer_size = next_layer
            prev_layer = relu_dropout(prev_layer, w, b, dropout)
            self.theta.extend((w, b))
        #Conv layers
        prev_layer = tf.reshape(prev_layer, [-1, conv_image_size, conv_image_size, colors])
        for i in range(conv_layers):
            size = conv_size*(conv_layers-i) if i < conv_layers-1 else colors
            prev_size = conv_size*(conv_layers-i+1) if i != 0 else colors
            w, b = filter_bias(self.name+str(expand_layers+i), [5, 5, size, prev_size], 0.1, 0.1)
            conv_image_size *= 2
            prev_layer = conv2d_transpose(prev_layer, w, b, [batch_size, conv_image_size, conv_image_size, size])
            self.theta.extend((w, b))
        self.output = tf.nn.tanh(prev_layer)


    def random_input(self, n=1):
        return np.random.uniform(-1., 1., size=[n, self.input_size])


    def generate(self, session, amount=1, name=None, print_array=False):
        #Prepare Images
        fullname = self.name
        if not name is None:
            fullname += '-'+name
        images = session.run(self.output, feed_dict={self.input: self.random_input(amount)})
        #Save
        if amount == 1:
            if print_array:
                print(np.asarray(images))
            save_image(images, self.image_size, fullname)
        else:
            for i in range(amount):
                save_image(images[i], self.image_size, fullname+"-"+str(i))

