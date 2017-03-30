import tensorflow as tf
from operators import weight_bias, conv2d, relu_dropout


class Discriminator():

    @classmethod
    def from_config(cls, config):
        """Create a Discriminator from a config object"""
        return Discriminator(
            config.name,
            config.image_size,
            config.colors,
            config.conv_layers,
            config.class_layers,
            config.dropout
            )

    def __init__(self, name, image_size=32, colors=1, conv_layers=[32, 64], class_layers=2, dropout=0.4):
        self.name = name+"_discriminator"
        self.input_dimensions = [None, image_size, image_size, colors]
        self.conv_layers = [1]+conv_layers
        self.class_layers = class_layers
        self.input = tf.placeholder(tf.float32, shape=self.input_dimensions)
        self.theta = [] #Trainable variables
        self.dropout = dropout

        #Network Layer Variables
        for i in range(len(conv_layers)):
            wb_vars = weight_bias(self.name+str(i), self.conv_layers[i:i+2])
            self.theta.extend(wb_vars)
        prev_layer_size = conv_layers[-1]
        for i in range(class_layers+1):
            next_layer_size = int(conv_layers[-1]*(1-i/class_layers)+1)
            wb_vars = weight_bias(self.name+str(len(conv_layers)+i), [prev_layer_size, next_layer_size])
            prev_layer_size = next_layer_size
            self.theta.extend(wb_vars)


    def get_network(self, input=None):
        """Constructs a network for the given input (or placeholder self.input if None)"""
        prev_layer = self.input if input is None else tf.reshape(input, self.input_dimensions)
        for i in range(0, len(self.conv_layers)*2-2, 2):
            prev_layer = conv2d(prev_layer, self.theta[i], self.theta[i+1])
        for i in range(len(self.conv_layers)*2, (len(self.conv_layers)+len(self.class_layers))*2, 2):
            prev_layer = relu_dropout(prev_layer, self.theta[i], self.theta[i-1], self.dropout)
        return tf.matmul(prev_layer, self.theta[-2]) + self.theta[-1]


