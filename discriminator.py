import math
import tensorflow as tf
from operators import weight_bias, conv2d, relu_dropout
from config import DiscriminatorConfig


class Discriminator():

    @classmethod
    def from_config(cls, name, fake_input, config=None):
        """Create a Discriminator from a config object"""
        if config is None:
            return Discriminator.from_config(name, fake_input, DiscriminatorConfig())
        return Discriminator(
            name,
            fake_input,
            config.image_size,
            config.colors,
            config.conv_layers,
            config.conv_size,
            config.class_layers,
            config.dropout,
            config.batch_size
            )

    def __init__(self, name, fake_input, image_size=32, colors=1, conv_layers=3, conv_size=32, class_layers=2, dropout=0.4, batch_size=64):
        self.name = name
        self.input_dimensions = [batch_size, image_size, image_size, colors]
        self.conv_layers = conv_layers
        self.conv_size = conv_size
        self.class_layers = class_layers
        self.fake_input = fake_input
        self.real_input = tf.placeholder(tf.float32, shape=self.input_dimensions)
        self.theta = [] #Trainable variables
        self.dropout = dropout
        self.batch_size = batch_size

        #Network Layers
        real_layer = self.real_input
        fake_layer = self.fake_input
        with tf.name_scope('Discriminator') as scope:
            conv_sizes = [1] + [conv_size*i for i in range(1, conv_layers+1)]
            for i in range(conv_layers):
                wb_vars = weight_bias('conv2d_layer%d'%i, [5, 5, conv_sizes[i], conv_sizes[i+1]])
                self.theta.extend(wb_vars)
                real_layer = conv2d(real_layer, *wb_vars)
                fake_layer = conv2d(fake_layer, *wb_vars)
            real_layer = tf.reshape(real_layer, [self.batch_size, -1])
            fake_layer = tf.reshape(fake_layer, [self.batch_size, -1])
            class_layer_size = ((image_size//(2**conv_layers))**2) * conv_sizes[-1]
            for i in range(class_layers):
                next_layer_size = 2**int(math.log(class_layer_size//2, 2)) if i == 0 else class_layer_size
                wb_vars = weight_bias('class_layer%d'%i, [class_layer_size, next_layer_size])
                class_layer_size = next_layer_size
                real_layer = relu_dropout(real_layer, *wb_vars, self.dropout)
                fake_layer = relu_dropout(fake_layer, *wb_vars, self.dropout)
                self.theta.extend(wb_vars)
            wb_vars = weight_bias('output_layer', [class_layer_size, 1])
            self.theta.extend(wb_vars)
            self.real_output = tf.matmul(real_layer, self.theta[-2]) + self.theta[-1]
            self.fake_output = tf.matmul(fake_layer, self.theta[-2]) + self.theta[-1]


if __name__ == "__main__":
    import numpy as np
    disc = Discriminator('asd', tf.placeholder(tf.float32, [1, 32, 32, 1]), batch_size=1, class_layers=0, conv_layers=1)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    print("sess")
    res = sess.run(disc.real_output, feed_dict={disc.real_input: np.random.uniform(0, 1, [1, 32, 32, 1])})
    print(res)