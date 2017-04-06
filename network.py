
import math
import os
from timeit import default_timer as timer

import numpy as np
import tensorflow as tf

from image import ImageVariations
from operators import *

LOG_FOLDER = 'logs'

class GANetwork():

    def __init__(self, name, image_size=64, colors=3, batch_size=64, directory='network', image_manager=None, 
                 input_size=128, learning_rate=0.1, dropout=0.4, generator_convolutions=5, generator_base_width=32,
                 discriminator_convolutions=4, discriminator_base_width=32, classification_depth=1):
        """
        Create a GAN for generating images
        Args:
          name: The name of the network
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
        """
        self.name = name
        self.image_size = image_size
        self.colors = colors
        self.batch_size = batch_size
        self.directory = directory
        self.input_size = input_size
        self.dropout = dropout
        #Setup Folders
        os.makedirs(directory, exist_ok=True)
        os.makedirs(LOG_FOLDER, exist_ok=True)
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
        self.generator_input, self.generator_output, self.image_output = \
            self.generator(generator_convolutions, generator_base_width)
        self.image_grid_output = self.image_grid()
        self.image_input, self.image_logit, self.generated_logit = \
            self.discriminator(self.generator_output, discriminator_convolutions, discriminator_base_width, classification_depth)
        self.generator_loss, self.discriminator_loss, self.d_loss_real, self.d_loss_fake = \
            self.loss_functions(self.image_logit, self.generated_logit)
        self.generator_solver, self.discriminator_solver = \
            self.solver_functions(self.generator_loss, self.discriminator_loss, learning_rate)


    def generator(self, conv_layers, conv_size):
        """Create a Generator Network"""
        with tf.variable_scope('generator'):
            #Network layer variables
            conv_image_size = self.image_size // (2**conv_layers)
            assert conv_image_size*(2**conv_layers) == self.image_size, "Images must be a multiple of two (or at least divisible by 2**num_of_conv_layers_plus_one)"
            #Input Layers
            input = tf.placeholder(tf.float32, [None, self.input_size], name='input')
            prev_layer = expand_relu(input, [-1, conv_image_size, conv_image_size, conv_size*2**(conv_layers-1)], 'expand')
            #Conv layers
            for i in range(conv_layers-1):
                prev_layer = conv2d_transpose(prev_layer, self.batch_size, 2**(conv_layers-i-2)*conv_size, 'convolution_%d'%i)
            output = conv2d_transpose_tanh(prev_layer, self.batch_size, self.colors, 'output')
            with tf.name_scope("image_output"):
                image_output = tf.cast((output + 1) * 127.5, tf.uint8)
        return input, output, image_output


    def discriminator(self, generator_output, conv_layers, conv_size, class_layers):
        """Create a Discriminator Network"""
        image_size = self.image_size
        with tf.variable_scope('discriminator') as scope:
            with tf.variable_scope('real_input'):
                real_input = tf.placeholder(tf.uint8, shape=[None, image_size, image_size, self.colors], name='image_input')
                real_input_scaled = tf.subtract(tf.to_float(real_input)/127.5, 1, name='scaling')
            conv_output_size = ((image_size//(2**conv_layers))**2) * conv_size * conv_layers
            class_output_size = 2**int(math.log(conv_output_size//2, 2))
            #Create Layers
            def create_network(layer):
                #Convolutional layers
                for i in range(conv_layers):
                    layer = conv2d(layer, conv_size*(i+1), name='convolution_%d'%i, norm=(i != 0))
                layer = tf.reshape(layer, [-1, conv_output_size])
                #Classification layers
                for i in range(class_layers):
                    layer = relu_dropout(layer, class_output_size, self.dropout, 'classification_%d'%i)
                return linear(layer, 1, 'output')
            fake_output = create_network(generator_output)
            scope.reuse_variables()
            real_output = create_network(real_input_scaled)
        return real_input, real_output, fake_output


    def loss_functions(self, real_logit, fake_logit):
        """Create loss calculations for the networks"""
        with tf.variable_scope('loss'):
            with tf.name_scope('discriminator'):
                real_loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logit, labels=tf.ones_like(real_logit)),
                    name='real_loss')
                fake_loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logit, labels=tf.zeros_like(fake_logit)),
                    name='fake_loss')
                d_loss = tf.add(real_loss, fake_loss, 'loss')
            with tf.name_scope('generator'):
                batch_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logit, labels=tf.ones_like(fake_logit))
                g_loss = tf.reduce_mean(batch_loss, name='loss')
        return g_loss, d_loss, real_loss, fake_loss

    def solver_functions(self, g_loss, d_loss, learning_rate):
        """Create solvers for the networks"""
        with tf.variable_scope('train'):
            g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
            g_solver = tf.train.AdadeltaOptimizer(learning_rate).minimize(g_loss, var_list=g_vars, global_step=self.iterations)
            d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
            d_solver = tf.train.AdadeltaOptimizer(learning_rate).minimize(d_loss, var_list=d_vars)
        return g_solver, d_solver

    def image_grid(self, size=5):
        ms = int(math.sqrt(self.batch_size))
        if ms < size:
            size = ms
        with tf.variable_scope('image_grid'):
            rows = []
            for w in range(size):
                columns = []
                for h in range(size):
                    columns.append(self.image_output[w+h*size])
                rows.append(tf.concat(columns, 0))
            grid = tf.concat(rows, 1, 'grid')
            """
            wh = self.image_size + (size+1)*2
            grid = tf.Variable(0, trainable=False, expected_shape=[wh, wh, self.colors], name='grid')
            grid = tf.batch_to_space_nd(
                self.image_output,
                [self.image_size, self.image_size],
                [self.image_size, self.image_size],
                name='grid'
            )
            images = tf.unpack(self.image_output, self.batch_size, 0)
            rows = []
            for i in range(size):
                imgs = []
                for j in range(size):
                    img = tf.slice(self.image_output, (j,0,0,0), (1,-1,-1,-1))
                    img = tf.reshape(img, (self.image_size, self.image_size, self.colors))
                    imgs.append(img)
                rows.append(tf.concat(0, imgs))
            grid = tf.concat(1, rows, name='grid')
            """
        return grid


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


    def train(self, batches=10000):
        """Train the network for a number of batches (continuing if there is an existing model)"""
        last_save = time = timer()
        saver = tf.train.Saver()
        with tf.Session() as session:
            start_iteration = self.__setup_session__(session, saver) + 1
            #Train
            try:
                for i in range(start_iteration, start_iteration+batches):
                    d_r_l, d_f_l, d_loss, g_loss, _, _ = session.run(
                        [self.d_loss_real, self.d_loss_fake, self.discriminator_loss,
                        self.generator_loss, self.discriminator_solver, self.generator_solver],
                        feed_dict={
                            self.image_input: self.image_manager.get_batch(),
                            self.generator_input: self.random_input(self.batch_size)
                        }
                    )
                    #Track progress
                    if i%10 == 0:
                        t = timer() - time
                        print("Iteration: %04d   Time: %02d:%02d:%02d    \tD loss: %.2f (%.2f | %.2f) \tG loss: %.2f" % \
                                (i, t//3600, t%3600//60, t%60, d_loss, d_r_l, d_f_l, g_loss))
                        if timer() - last_save > 600:
                            saver.save(session, os.path.join(self.directory, self.name))
                            last_save = timer()
                        if i%500 == 0:
                            self.generate_grid(session, "%s_%05d"%(self.name, i))
            finally:
                saver.save(session, os.path.join(self.directory, self.name))

    def __setup_session__(self, session, saver):
        session.run(tf.global_variables_initializer())
        try:
            saver.restore(session, os.path.join(self.directory, self.name))
            start_iteration = session.run(self.iterations)
            print("\nTraining an existing network\n")
        except:
            start_iteration = 0
            tf.summary.FileWriter(os.path.join(LOG_FOLDER, self.name), session.graph)
            print("\nTraining a new network\n")
        return start_iteration
