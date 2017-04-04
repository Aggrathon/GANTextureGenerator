import math
import tensorflow as tf
from operators import conv2d, relu_dropout, linear
from config import DiscriminatorConfig


class Discriminator():

    @classmethod
    def from_config(cls, fake_input, config=None):
        """Create a Discriminator from a config object"""
        if config is None:
            return Discriminator.from_config(fake_input, DiscriminatorConfig())
        return Discriminator(
            fake_input,
            config.image_size,
            config.colors,
            config.conv_layers,
            config.conv_size,
            config.class_layers,
            config.dropout,
            config.batch_size,
            config.learning_rate
            )

    def __init__(self, fake_input, image_size=32, colors=1, conv_layers=3, conv_size=32, class_layers=2, dropout=0.4, batch_size=64, learning_rate=0.001):
        self.image_size = image_size
        self.real_input = None
        self.fake_input = fake_input
        self.real_output = None
        self.fake_output = None
        self.trainable_variables = None
        self.loss = None
        self.solver = None

        #Network Layers
        with tf.variable_scope('discriminator') as scope:
            self.real_input = tf.placeholder(tf.uint8, shape=[batch_size, image_size, image_size, colors], name='real_input')
            conv_output_size = ((image_size//(2**conv_layers))**2) * conv_size * conv_layers
            class_output_size = 2**int(math.log(conv_output_size//2, 2))
            #Create Layers
            def create_network(layer):
                #Convolutional layers
                for i in range(conv_layers):
                    layer = conv2d(layer, conv_size*(i+1), name='convolution_%d'%i, norm=(i != 0))
                layer = tf.reshape(layer, [batch_size, conv_output_size])
                #Classification layers
                for i in range(class_layers):
                    layer = relu_dropout(layer, class_output_size, dropout, 'classification_%d'%i)
                return linear(layer, 1, 'output')
            self.fake_output = create_network(self.fake_input)
            scope.reuse_variables()
            self.real_output = create_network(tf.to_float(self.real_input)/127.5 - 1)
            #Loss and solver functions
            with tf.name_scope('loss'):
                self.real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.real_output, labels=tf.constant(0.9, shape=[batch_size, 1])))
                self.fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake_output, labels=tf.constant(0.1, shape=[batch_size, 1])))
                self.loss = self.real_loss + self.fake_loss
            self.trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)
        self.solver = tf.train.AdadeltaOptimizer(learning_rate).minimize(self.loss, var_list=self.trainable_variables)

        