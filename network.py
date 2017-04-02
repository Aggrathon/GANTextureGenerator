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
        self.batch_size = batch_size
        self.iteration = tf.Variable(0, name=name+"_iterations")
        #Setup Folders
        os.makedirs(NETWORK_FOLDER, exist_ok=True)
        #Setup Objects
        generator_config.batch_size = batch_size
        discriminator_config.batch_size = batch_size
        generator_config.image_size = image_size
        discriminator_config.image_size = image_size
        self.image_manager = ImageVariations(image_size, batch_size) if image_source is None else image_source
        #Setup Networks
        self.generator = Generator.from_config(generator_config)
        self.discriminator = Discriminator.from_config(self.generator.output, discriminator_config)
        self.real_input = self.discriminator.real_input
        self.fake_input = self.generator.input
        self.generator.setup_loss(self.discriminator.fake_output)

    def train(self, batches=10000):
        time = timer()
        last_save = time
        saver = tf.train.Saver()
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            try:
                saver.restore(session, os.path.join(NETWORK_FOLDER, self.name))
                start_iteration = session.run(self.iteration)
                print("\nTraining an old network\n")
            except:
                start_iteration = 0
                print("\nTraining a new network\n")
            #Train
            for i in range(start_iteration+1, start_iteration+batches+1):
                d_loss, g_loss, _, _ = session.run(
                    [self.discriminator.loss, self.generator.loss, self.discriminator.solver, self.generator.solver], 
                    feed_dict={
                        self.real_input: self.image_manager.get_batch(),
                        self.fake_input: self.generator.random_input(self.batch_size)
                    }
                )
                #Track progress
                if i%10 == 0:
                    t = timer() - time
                    print("Iteration: %04d   Time: %02d:%02d:%02d    \tD loss: %.1f \tG loss: %.1f" % \
                            (i, t//3600, t%3600//60, t%60, d_loss, g_loss))
                    if i%100 == 0 or timer() - last_save > 600:
                        session.run(self.iteration.assign(i))
                        saver.save(session, os.path.join(NETWORK_FOLDER, self.name))
                        last_save = timer()
                        if i%200 == 0:
                            self.generator.generate(session, "%s%05d"%(self.name, i))
