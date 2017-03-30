import math
import tensorflow as tf
from operators import weight_bias, conv2d, relu_dropout
from config import DiscriminatorConfig


class Discriminator():

    @classmethod
    def from_config(cls, name, config=None):
        """Create a Discriminator from a config object"""
        if config is None:
            return Discriminator.from_config(name, DiscriminatorConfig())
        return Discriminator(
            name,
            config.image_size,
            config.colors,
            config.conv_layers,
            config.class_layers,
            config.conv_size,
            config.dropout,
            config.batch_size
            )

    def __init__(self, name, image_size=32, colors=1, conv_layers=3, conv_size=32, class_layers=2, dropout=0.4, batch_size=64):
        self.name = name+"_discriminator"
        self.input_dimensions = [batch_size, image_size, image_size, colors]
        self.conv_layers = conv_layers
        self.conv_size = conv_size
        self.class_layers = class_layers
        self.input = tf.placeholder(tf.float32, shape=self.input_dimensions)
        self.theta = [] #Trainable variables
        self.dropout = dropout
        self.batch_size = batch_size

        #Network Layer Variables
        for i in range(conv_layers):
            wb_vars = weight_bias(self.name+str(i), [5, 5, 1 if i == 0 else conv_size*(i-1), conv_size*i])
            self.theta.extend(wb_vars)
        class_layer_size = (conv_layers-1)*conv_size*((image_size//(2**conv_layers))**2)
        for i in range(class_layers):
            next_size = 2**int(math.log(class_layer_size//2, 2))
            wb_vars = weight_bias(self.name+str(conv_layers+i), [class_layer_size, next_size])
            class_layer_size = next_size
            self.theta.extend(wb_vars)
        wb_vars = weight_bias(self.name+str(conv_layers+i), [class_layer_size, 1])
        self.theta.extend(wb_vars)


    def get_network(self, input=None):
        """Constructs a network for the given input (or placeholder self.input if None)"""
        prev_layer = self.input if input is None else tf.reshape(input, self.input_dimensions)
        for i in range(0, self.conv_layers*2, 2):
            prev_layer = conv2d(prev_layer, self.theta[i], self.theta[i+1])
        prev_layer = tf.reshape(prev_layer, [self.batch_size, -1])
        for i in range(self.conv_layers*2, len(self.theta)-2, 2):
            prev_layer = relu_dropout(prev_layer, self.theta[i], self.theta[i+1], self.dropout)
        return tf.matmul(prev_layer, self.theta[-2]) + self.theta[-1]


