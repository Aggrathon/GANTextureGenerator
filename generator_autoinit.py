
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
        auto_code = image_encoder([self.image_input_scaled], 'autoencoder', self.image_size, self._dis_conv, self._dis_width, self._class_depth, self._dropout, self.input_size, True)[0]
        self.generator_output, auto_out = image_decoder([self.generator_input, auto_code], 'generator', self.image_size, self._gen_conv, self._gen_width, self.input_size, self.batch_size, self.colors)
        self.image_output, self.image_grid_output = image_output([self.generator_output], 'output', self.image_size, self.grid_size)
        gen_logit, image_logit = image_encoder([self.generator_output, self.image_input_scaled], 'discriminator', self.image_size, self._dis_conv, self._dis_width, self._class_depth, self._dropout, 1)
        with tf.variable_scope('train'):
            auto_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='autoencoder')
            gen_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
            dis_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
            self.generator_solver = batch_optimizer('generator', gen_var, [gen_logit], 1-self._y_offset, '', None, 0, '', *self.learning_rate, global_step=self.iterations, summary=self.log)
            self.discriminator_solver = batch_optimizer('discriminator', dis_var, [image_logit], 1-self._y_offset, 'real_', [gen_logit], self._y_offset, 'fake_', *self.learning_rate, summary=self.log)
            self.autoencoder_solver = image_optimizer('autoencoder', auto_var+gen_var, [self.image_input_scaled], [auto_out], self.learning_rate[0]*2, self.learning_rate[1], self.learning_rate[2], summary=self.log)


    def __training_iteration__(self, session, i):
        feed_dict = {
            self.image_input: self.image_manager.get_batch(),
            self.generator_input: self.random_input()
        }
        if i <= 200:
            session.run([self.discriminator_solver, self.autoencoder_solver], feed_dict=feed_dict)
        else:
            session.run([self.generator_solver, self.discriminator_solver], feed_dict=feed_dict)

