
import math

import tensorflow as tf
import numpy as np

from generator_gan import GANetwork
from network import image_decoder, image_encoder, image_output, image_optimizer, batch_optimizer

class AutoGanGenerator(GANetwork):

    def __init__(self, **kwargs):
        super().__init__(setup=False, **kwargs)
        self.autoencoder_solver = None
        self.setup_network()

    def setup_network(self):
        """Initialize the network if it is not done in the constructor"""
        auto_code = image_encoder([self.image_input_scaled], 'autoencoder', self.image_size, self._dis_conv, self._dis_width, self._class_depth, self._dropout, self.input_size)[0]
        self.generator_output, auto_out = image_decoder([self.generator_input, auto_code], 'generator', self.image_size, self._gen_conv, self._gen_width, self.input_size, self.batch_size, self.colors)
        self.image_output, self.image_grid_output = image_output([self.generator_output], 'output', self.image_size, self.grid_size)
        gen_logit, image_logit = image_encoder([self.generator_output, self.image_input_scaled], 'discriminator', self.image_size, self._dis_conv, self._dis_width, self._class_depth, self._dropout, 1)
        with tf.variable_scope('train'):
            auto_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='autoencoder')
            gen_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
            dis_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
            self.generator_solver = batch_optimizer('generator', gen_var, [gen_logit], 1-self._y_offset, '', None, 0, '', *self.learning_rate, global_step=self.iterations, summary=self.log)
            self.discriminator_solver = batch_optimizer('discriminator', dis_var, [image_logit], 1-self._y_offset, 'real_', [gen_logit], self._y_offset, 'fake_', *self.learning_rate, summary=self.log)
            self.autoencoder_solver = image_optimizer('autoencoder', auto_var+gen_var, [self.image_input_scaled], [auto_out], *self.learning_rate, summary=self.log)
        if self.log:
            self.__variation_summary__()

    def __autoencoder_solver__(self, decoder_result):
        #Decoder
        with tf.variable_scope('decoder_loss'):
            #Optimizer
            loss = tf.reduce_mean(tf.pow(decoder_result-self.image_input_scaled, 2))
            optimizer = tf.train.AdamOptimizer(*self.learning_rate)
            variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='autoencoder')
            auto_solver = optimizer.minimize(loss, var_list=variables)
        return auto_solver

    def random_input(self, n=1):
        """Creates a random input for the generator"""
        return np.random.uniform(-1.0, 1.0, size=[n, self.input_size])

    def get_calculations(self):
        return [self.generator_solver, self.discriminator_solver, self.autoencoder_solver]
