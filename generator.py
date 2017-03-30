import os
import tensorflow as tf
import numpy as np
from image import save_image
from config import INPUT_SIZE, NETWORK_FOLDER, GEN_DROPOUT, GEN_HIDDEN_LAYERS, IMAGE_SIZE, COLORED
from operators import weight_bias


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
            config.input_size,
            config.dropout
        )

    def __init__(self, name, image_size=32, colors=1, 
                 expand_layers=2, conv_layers=3, input_size=128, dropout=0.4):
        self.name = name+"_generator"
        self.image_size = image_size
        preconv = ((image_size-1)//(2**len(conv_layers))+1)**2 * colors
        self.input = tf.placeholder(tf.float32, [None, input_size])
        self.theta = [] #Trainable variables

        #Network layer variables
        prev_layer = self.input
        for i in range(expand_layers):
            w, b = weight_bias(self.name+str(i), [self.layer_data[i], self.layer_data[i+1]], 0.1, 0.1)
            pass
        for i in range(hidden_layers+1):
            w, b = weight_bias(self.name+str(expand_layers+i), [self.layer_data[i], self.layer_data[i+1]], 0.1, 0.1)
            if i != hidden_layers:
                prev_layer = tf.nn.relu(tf.matmul(prev_layer, w) + b)
                prev_layer = tf.nn.dropout(prev_layer, dropout)
            else:
                self.output = tf.matmul(prev_layer, w) + b
            self.theta.extend((w, b))

    def random_input(self, n=1):
        return np.random.uniform(-1., 1., size=[n, self.layer_data[0]])


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

