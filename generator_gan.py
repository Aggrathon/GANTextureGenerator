
import math
import os
from timeit import default_timer as timer

import numpy as np
import tensorflow as tf

from image import ImageVariations
from network import image_decoder, image_encoder, image_output

LOG_DIR = 'logs'

class GANetwork():

    def __init__(self, name, setup=True, image_size=64, colors=3, batch_size=64, directory='network', image_manager=None,
                 input_size=64, learning_rate=0.0002, dropout=0.4, generator_convolutions=5, generator_base_width=32,
                 discriminator_convolutions=4, discriminator_base_width=32, classification_depth=1, grid_size=4,
                 log=True, y_offset=0.1, learning_momentum=0.6, learning_momentum2=0.9):
        """
        Create a GAN for generating images
        Args:
          name: The name of the network
          setup: Initialize the network in the constructor
          image_size: The size of the generated images
          colors: number of color layers (3 is rgb, 1 is grayscale)
          batch_size: images per training batch
          directory: where to save the trained network
          image_manager: a class generating real images for training
          input_size: the number of images fed to the generator when generating an image
          learning_rate: the initial rate of learning
          dropout: improve the discriminator with some dropout
          generator_convolutions: the number of convolutional layers in the generator
          generator_base_width: the base number of convolution kernels per layer in the generator
          discriminator_convolutions: the number of convolutional layers in the discriminator
          discriminator_base_width: the base number of convolution kernels per layer in the discriminator
          classification_depth: the number of fully connected layers in the discriminator
          grid_size: the size of the grid when generating an image grid
          log: should tensorboard logs be created
          y_offset: how much should the "right" answers vary from 1s and 0s
          learning_momentum: the beta1 momentum for ADAM
          learning_momentum2: the beta2 momentum for ADAM
        """
        self.name = name
        self.image_size = image_size
        self.colors = colors
        self.batch_size = batch_size
        self.grid_size = min(grid_size, int(math.sqrt(batch_size)))
        self.log = log
        self.directory = directory
        os.makedirs(directory, exist_ok=True)
        #Network variables
        self.input_size = input_size
        self._gen_conv = generator_convolutions
        self._gen_width = generator_base_width
        self._dis_conv = discriminator_convolutions
        self._dis_width = discriminator_base_width
        self._class_depth = classification_depth
        self._dropout = dropout
        #Training variables
        self.learning_rate = (learning_rate, learning_momentum, learning_momentum2)
        self._y_offset = y_offset
        #Setup Images
        if image_manager is None:
            self.image_manager = ImageVariations(image_size=image_size, batch_size=batch_size, colored=(colors == 3))
        else:
            self.image_manager = image_manager
            self.image_manager.batch_size = batch_size
            self.image_manager.image_size = image_size
            self.image_manager.colored = (colors == 3)
        self.image_manager.start_threads()
        #Setup Networks
        self.iterations = tf.Variable(0, name="training_iterations", trainable=False)
        with tf.variable_scope('input'):
            self.generator_input = tf.placeholder(tf.float32, [None, self.input_size], name='generator_input')
            self.image_input = tf.placeholder(tf.uint8, shape=[None, image_size, image_size, self.colors], name='image_input')
            self.image_input_scaled = tf.subtract(tf.to_float(self.image_input)/127.5, 1, name='image_scaling')
        self.generator_output = None
        self.image_output = self.image_grid_output = None
        self.variation_updater = self.image_variation = None
        self.generator_solver = self.discriminator_solver = None
        if setup:
            self.setup_network()

    def setup_network(self):
        """Initialize the network if it is not done in the constructor"""
        self.generator_output = image_decoder([self.generator_input], 'generator', self.image_size,
            self._gen_conv, self._gen_width, self.input_size, self.batch_size, self.colors)[0]
        self.image_output, self.image_grid_output = image_output([self.generator_output], 'output', self.image_size, self.grid_size)
        gen_logit, image_logit = image_encoder([self.generator_output, self.image_input_scaled], 'discriminator',
            self.image_size, self._dis_conv, self._dis_width, self._class_depth, self._dropout, 1)
        g_loss, d_loss, d_loss_real, d_loss_fake = self.loss_functions(image_logit, gen_logit, self._y_offset)
        self.generator_solver, self.discriminator_solver = self.solver_functions(g_loss, d_loss, *self.learning_rate)
        if self.log:
            self.__variation_summary__()

    def __output__(self):
        if self.log:
            self.__variation_summary__()
        with tf.name_scope('output'):
            with tf.name_scope("image_list") as scope:
                self.image_output = tf.cast((self.generator_output + 1) * 127.5, tf.uint8, name=scope)
            with tf.name_scope('image_grid') as scope:
                wh = self.grid_size * (self.image_size + 2) + 2
                grid = tf.Variable(0, trainable=False, dtype=tf.uint8, expected_shape=[wh, wh, self.image_size])
                for x in range(self.grid_size):
                    for y in range(self.grid_size):
                        bound = tf.to_int32(tf.image.pad_to_bounding_box(
                            self.image_output[x+y*self.grid_size],
                            2 + x*(self.image_size + 2),
                            2 + y*(self.image_size + 2),
                            wh, wh
                        ))
                        if x == 0 and y == 0:
                            grid = bound
                        else:
                            grid = tf.add(grid, bound)
                self.image_grid_output = tf.cast(grid, tf.uint8, name=scope)


    def __variation_summary__(self):
        """Create summaries for pixel variation"""
        with tf.variable_scope('pixel_variation'):
            #Pixel Variations
            img_tot_var = tf.image.total_variation(self.image_input_scaled)
            gen_tot_var = tf.image.total_variation(self.generator_output)
            image_variation = tf.reduce_sum(img_tot_var)
            gener_variation = tf.reduce_sum(gen_tot_var)
            tf.summary.histogram('images', img_tot_var)
            tf.summary.histogram('generated', gen_tot_var)
            tf.summary.scalar('images', image_variation)
            tf.summary.scalar('generated', gener_variation)
            ema = tf.train.ExponentialMovingAverage(decay=0.95, num_updates=self.iterations)
            ema_apply = ema.apply([image_variation])
            self.variation_updater = tf.group(ema_apply)
            self.image_variation = ema.average(image_variation)
            tf.summary.scalar('images_averaged', self.image_variation)


    def loss_functions(self, real_logit, fake_logit, y_offset=0):
        """Create loss calculations for the networks"""
        with tf.variable_scope('loss'):
            with tf.name_scope('discriminator'):
                if y_offset < 0.001:
                    d_r_labels = tf.ones_like(real_logit)
                    d_f_labels = tf.zeros_like(fake_logit)
                    g_r_labels = tf.ones_like(fake_logit)
                else:
                    d_r_labels = tf.fill(tf.shape(real_logit), (1 - y_offset))
                    d_f_labels = tf.fill(tf.shape(fake_logit), y_offset)
                    g_r_labels = tf.fill(tf.shape(fake_logit), (1 - y_offset))
                real_loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logit, labels=d_r_labels),
                    name='real_loss')
                fake_loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logit, labels=d_f_labels),
                    name='fake_loss')
                d_loss = tf.add(real_loss, fake_loss, 'loss')
                tf.summary.scalar('discriminator_loss', d_loss)
                tf.summary.scalar('fake_loss', fake_loss)
                tf.summary.scalar('real_loss', real_loss)
            with tf.name_scope('generator'):
                batch_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logit, labels=g_r_labels)
                g_loss = tf.reduce_mean(batch_loss, name='loss')
                tf.summary.scalar('generator_loss', g_loss)
        return g_loss, d_loss, real_loss, fake_loss

    def solver_functions(self, g_loss, d_loss, learning_rate, learning_momentum, learning_momentum2=0.7):
        """Create solvers for the networks"""
        with tf.variable_scope('train'):
            with tf.variable_scope('generator'):
                g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
                g_solver = tf.train.AdamOptimizer(learning_rate*2, learning_momentum, learning_momentum2).minimize(g_loss, var_list=g_vars, global_step=self.iterations)
            with tf.variable_scope('discriminator'):
                d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
                d_solver = tf.train.AdamOptimizer(learning_rate, learning_momentum, learning_momentum2).minimize(d_loss, var_list=d_vars)
        return g_solver, d_solver


    def random_input(self, n=1):
        """Creates a random input for the generator"""
        return np.random.uniform(0.0, 1.0, size=[n, self.input_size])


    def generate(self, session, name, amount=1):
        """Generate a image and save it"""
        def get_arr():
            arr = np.asarray(session.run(
                self.image_output,
                feed_dict={self.generator_input: self.random_input(self.batch_size)}
            ), np.uint8)
            arr.shape = self.batch_size, self.image_size, self.image_size, self.colors
            return arr
        if amount == 1:
            self.image_manager.save_image(get_arr()[0], name)
        else:
            images = []
            counter = amount
            while counter > 0:
                images.extend(get_arr())
                counter -= self.batch_size
            for i in range(amount):
                self.image_manager.save_image(images[i], "%s_%02d"%(name, i))

    def generate_grid(self, session, name):
        """Generate a image and save it"""
        grid = session.run(
            self.image_grid_output,
            feed_dict={self.generator_input: self.random_input(self.batch_size)}
        )
        self.image_manager.image_size = self.image_grid_output.get_shape()[1]
        self.image_manager.save_image(grid, name)
        self.image_manager.image_size = self.image_size


    def get_session(self):
        saver = tf.train.Saver()
        session = tf.Session()
        session.run(tf.global_variables_initializer())
        try:
            saver.restore(session, os.path.join(self.directory, self.name))
            start_iteration = session.run(self.iterations)
            print("\nLoaded an existing network\n")
        except Exception as e:
            start_iteration = 0
            if self.log:
                tf.summary.FileWriter(os.path.join(LOG_DIR, self.name), session.graph)
                print("\nCreated a new network (%s)\n"%repr(e))
        return session, saver, start_iteration

    def get_calculations(self):
        return [self.generator_solver, self.discriminator_solver]

    def train(self, batches=100000, print_interval=1):
        """Train the network for a number of batches (continuing if there is an existing model)"""
        start_time = last_time = last_save = timer()
        session, saver, start_iteration = self.get_session()
        calculations = self.get_calculations()
        if self.log:
            logger = SummaryLogger(self, session)
            calculations += logger.get_calculations()
        try:
            print("Training the GAN on images in the '%s' folder"%self.image_manager.in_directory)
            print("To stop the training early press Ctrl+C (progress will be saved)")
            print('To continue training just run the training again')
            if self.log:
                print("To view the progress run 'python -m tensorflow.tensorboard --logdir %s'"%LOG_DIR)
            print("To generate images using the trained network run 'python generate.py %s'"%self.name)
            print()
            for i in range(start_iteration, start_iteration+batches+1):
                feed_dict = {
                    self.image_input: self.image_manager.get_batch(),
                    self.generator_input: self.random_input(self.batch_size)
                }
                session.run(calculations, feed_dict=feed_dict)
                #Print progress
                if i%print_interval == 0:
                    curr_time = timer()
                    time_per = (curr_time-last_time)/print_interval
                    time = curr_time - start_time
                    print("Iteration: %04d    Time: %02d:%02d:%02d  (%02.1fs / iteration)" % \
                        (i, time//3600, time%3600//60, time%60, time_per), end='\r')
                    last_time = curr_time
                if self.log:
                    logger(i, feed_dict)
                #Save network
                if timer() - last_save > 1800:
                    saver.save(session, os.path.join(self.directory, self.name))
                    last_save = timer()
        except KeyboardInterrupt:
            print()
            print("Stopping the training", end='')
        finally:
            saver.save(session, os.path.join(self.directory, self.name))
            if self.log:
                logger.close()
            session.close()


class SummaryLogger():
    """Log the progress of training to tensorboard (and some progress output to the console)"""
    def __init__(self, network, session, summary_interval=20, image_interval=500):
        self.session = session
        self.gan = network
        self.image_interval = image_interval
        self.summary_interval = summary_interval
        os.makedirs(LOG_DIR, exist_ok=True)
        self.writer = tf.summary.FileWriter(os.path.join(LOG_DIR, network.name))
        self.summary = tf.summary.merge_all()
        self.batch_input = network.random_input(network.batch_size)

    def get_calculations(self):
        return [self.gan.variation_updater]

    def __call__(self, iteration, dict=None):
        #Save image
        if iteration%self.image_interval == 0:
            #Hack to make tensorboard show multiple images, not just the latest one
            dict[self.gan.generator_input] = self.batch_input
            image, summary = self.session.run(
                [tf.summary.image(
                    'training/iteration/%d'%iteration,
                    tf.stack([self.gan.image_grid_output]),
                    max_outputs=1,
                    collections=['generated_images']
                ), self.summary],
                feed_dict=dict
            )
            self.writer.add_summary(image, iteration)
            self.writer.add_summary(summary, iteration)
        elif iteration%self.summary_interval == 0:
            #Save summary
            summary = self.session.run(self.summary, feed_dict=dict)
            self.writer.add_summary(summary, iteration)

    def close(self):
        self.writer.close()
        print()
