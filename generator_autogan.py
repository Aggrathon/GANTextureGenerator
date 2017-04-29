
import math

import tensorflow as tf

from generator_gan import GANetwork
from operators import conv2d, relu_dropout, linear

class AutoGanGenerator(GANetwork):

    def __init__(self, **kwargs):
        super().__init__(setup=False, **kwargs)
        self.autoencoder_solver = None
        #TODO setup autencoder
        self.setup_network()

    def setup_network(self):
        """Initialize the network if it is not done in the constructor"""
        auto_code = self.__autoencoder_encoder__()
        # pylint: disable=unbalanced-tuple-unpacking
        self.generator_output, auto_out = self.__generator__([self.generator_input, auto_code])
        self.__output__()
        gen_logit, image_logit = self.__discriminator__([self.generator_output, self.image_input_scaled])
        g_loss, d_loss, d_loss_real, d_loss_fake = self.loss_functions(image_logit, gen_logit, self._y_offset)
        self.generator_solver, self.discriminator_solver = self.solver_functions(g_loss, d_loss, *self.learning_rate)
        self.autoencoder_solver = self.__autoencoder_solver__(auto_out)

    def __autoencoder_encoder__(self):
        with tf.variable_scope('autoencoder'):
            #Encoder
            conv_layers, conv_size, class_layers, image_size = self._dis_conv, self._dis_width, self._class_depth, self.image_size
            conv_output_size = ((image_size//(2**conv_layers))**2) * conv_size * conv_layers
            class_output_size = 2**int(math.log(conv_output_size//2, 2))
            prev_layer = [self.image_input_scaled]
            for i in range(conv_layers): #Convolutional layers
                prev_layer = conv2d(prev_layer, conv_size*(i+1), name='convolution_%d'%i, norm=(i != 0))
            prev_layer = [tf.reshape(layer, [-1, conv_output_size]) for layer in prev_layer]
            for i in range(class_layers): #Coding layers
                prev_layer = relu_dropout(prev_layer, class_output_size, self._dropout, 'coding_%d'%i)
            return linear(prev_layer, self.input_size, 'output')[0]

    def __autoencoder_solver__(self, decoder_result):
        #Decoder
        with tf.variable_scope('decoder_loss'):
            #Optimizer
            loss = tf.reduce_mean(tf.pow(decoder_result-self.image_input_scaled, 2))
            optimizer = tf.train.AdamOptimizer(*self.learning_rate)
            variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='autoencoder')
            auto_solver = optimizer.minimize(loss, var_list=variables)
        return auto_solver

    def get_calculations(self):
        return [self.generator_solver, self.discriminator_solver, self.autoencoder_solver]
