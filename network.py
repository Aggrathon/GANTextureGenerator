import os
from timeit import default_timer as timer
import tensorflow as tf
from image import ImageVariations
from generator import Generator
from discriminator import Discriminator
from config import NETWORK_FOLDER, BATCH_SIZE, IMAGE_SIZE, LEARNING_RATE, GeneratorConfig, DiscriminatorConfig


class GANetwork():

    def __init__(self, name, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,
                 network_folder=NETWORK_FOLDER, image_source=None, 
                 generator_config=GeneratorConfig(), discriminator_config=DiscriminatorConfig()):
        self.name = name
        #Setup Folders
        os.makedirs(NETWORK_FOLDER, exist_ok=True)
        #Setup Objects
        generator_config.batch_size = batch_size
        discriminator_config.batch_size = batch_size
        generator_config.image_size = image_size
        discriminator_config.image_size = image_size
        self.image_manager = ImageVariations(image_size, batch_size) if image_source is None else image_source
        self.generator = Generator.from_config(name, generator_config)
        self.discriminator = Discriminator.from_config(name, self.generator.output, discriminator_config)
        #Setup Networks
        self.real_input = self.discriminator.real_input
        self.fake_input = self.generator.input
        #Setup Training
        d_logit_real = self.discriminator.real_output
        d_logit_fake = self.discriminator.fake_output
        self.d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logit_real, labels=tf.ones_like(d_logit_real))) + \
                        tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logit_fake, labels=tf.zeros_like(d_logit_fake)))
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logit_fake, labels=tf.ones_like(d_logit_fake)))
        self.d_solver = tf.train.AdamOptimizer(learning_rate).minimize(self.d_loss, var_list=self.discriminator.theta)
        self.g_solver = tf.train.AdamOptimizer(learning_rate).minimize(self.g_loss, var_list=self.generator.theta)

    def train(self, batches=10000):
        time = timer()
        last_save = time
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
                d_loss, g_loss, _, _ = session.run([self.d_loss, self.g_loss, self.d_solver, self.g_solver], feed_dict=feed_dict)
                #Track progress
                if i%10 == 0:
                    t = timer() - time
                    print("Iteration: %04d \t D loss: %.1f \t G loss: %.1f \t Time: %02d:%02d:%02d" % \
                            (i, d_loss, g_loss, t//3600, t%3600//60, t%60))
                if i%100 == 0 or timer() - last_save > 600:
                    saver.save(session, os.path.join(NETWORK_FOLDER, self.name))
                    self.generator.generate(session, 1, "%05d"%i)
                    last_save = timer()
