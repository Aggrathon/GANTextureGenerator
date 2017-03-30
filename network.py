import os
from timeit import default_timer as timer
import tensorflow as tf
from image import ImageVariations
from generator import Generator
from discriminator import Discriminator
from config import NETWORK_FOLDER, BATCH_SIZE, IMAGE_SIZE


class GANetwork():

    def __init__(self, name, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, network_folder=NETWORK_FOLDER, image_source=None, generator=None, discriminator=None):
        self.name = name
        #Setup Folders
        os.makedirs(NETWORK_FOLDER, exist_ok=True)
        #Setup Objects
        self.image_manager = ImageVariations(image_size, batch_size) if image_source is None else image_source
        self.generator = Generator(name, image_size) if generator is None else generator
        self.discriminator = Discriminator(name, image_size) if discriminator is None else discriminator
        #Setup Networks
        self.real_input = self.discriminator.input
        self.fake_input = self.generator.input
        #Setup Training
        d_logit_real = self.discriminator.get_network()
        d_logit_fake = self.discriminator.get_network(self.generator.output)
        self.d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logit_real, labels=tf.ones_like(d_logit_real))) + \
                        tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logit_fake, labels=tf.zeros_like(d_logit_fake)))
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logit_fake, labels=tf.ones_like(d_logit_fake)))
        self.d_solver = tf.train.AdamOptimizer(0.1).minimize(self.d_loss, var_list=self.discriminator.theta)
        self.g_solver = tf.train.AdamOptimizer(0.1).minimize(self.g_loss, var_list=self.generator.theta)

    def train(self, batches=10000):
        time = timer()
        saver = tf.train.Saver()
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            try:
                saver.restore(session, os.path.join(NETWORK_FOLDER, self.name))
                print("Training an old network")
            except:
                print("Training a new network")
            #Train
            for i in range(1, batches+1):
                feed_dict = {
                    self.real_input: self.image_manager.get_batch(),
                    self.fake_input: self.generator.random_input(BATCH_SIZE)
                }
                session.run([self.d_solver, self.g_solver], feed_dict=feed_dict)
                #Track progress
                if i%50 == 0:
                    d_loss, g_loss = session.run([self.d_loss, self.g_loss], feed_dict=feed_dict)
                    t = timer() - time
                    print("Iteration: %04d \t D loss: %.1f \t G loss: %.1f \t Time: %02d:%02d:%02d" % \
                            (i, d_loss, g_loss, t//3600, t%3600//60, t%60))
                    if i%200 == 0:
                        saver.save(session, os.path.join(NETWORK_FOLDER, self.name))
                        self.generator.generate(session, 1, "%05d"%i, True)
