import os
from timeit import default_timer as timer
import tensorflow as tf
from image import ImageVariations
from generator import Generator
from discriminator import Discriminator
from config import NETWORK_FOLDER, BATCH_SIZE, IMAGE_SIZE, LOG_FOLDER, GeneratorConfig, DiscriminatorConfig


class GANetwork():

    def __init__(self, name, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, network_folder=NETWORK_FOLDER, image_source=None,
                 generator_config=GeneratorConfig(), discriminator_config=DiscriminatorConfig()):
        self.name = name
        self.batch_size = batch_size
        self.iteration = tf.Variable(0, name="training_iterations")
        #Setup Folders
        os.makedirs(network_folder, exist_ok=True)
        os.makedirs(LOG_FOLDER, exist_ok=True)
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
        """Train the network for a number of batches (continuing if there is an existing model)"""
        last_save = time = timer()
        saver = tf.train.Saver()
        with tf.Session() as session:
            start_iteration = self.__setup_session__(session, saver) + 1
            #Train
            for i in range(start_iteration, start_iteration+batches):
                try:
                    d_r_l, d_f_l, d_loss, g_loss, _, _ = session.run(
                        [self.discriminator.real_loss, self.discriminator.fake_loss, self.discriminator.loss,
                         self.generator.loss, self.discriminator.solver, self.generator.solver],
                        feed_dict={
                            self.real_input: self.image_manager.get_batch(),
                            self.fake_input: self.generator.random_input(self.batch_size)
                        }
                    )
                    #Track progress
                    if i%10 == 0:
                        t = timer() - time
                        print("Iteration: %04d   Time: %02d:%02d:%02d    \tD loss: %.2f (%.2f | %.2f) \tG loss: %.2f" % \
                                (i, t//3600, t%3600//60, t%60, d_loss, d_r_l, d_f_l, g_loss))
                        if i%100 == 0 or timer() - last_save > 600:
                            last_save = self.__save_network__(session, saver, i)
                            if i%200 == 0:
                                self.generator.generate(session, "%s_%05d"%(self.name, i))
                except (KeyboardInterrupt, SystemExit):
                    self.__save_network__(session, saver, i)
                    raise

    def __setup_session__(self, session, saver):
        session.run(tf.global_variables_initializer())
        try:
            saver.restore(session, os.path.join(NETWORK_FOLDER, self.name))
            start_iteration = session.run(self.iteration)
            print("\nTraining an old network\n")
        except:
            start_iteration = 0
            tf.summary.FileWriter(os.path.join(LOG_FOLDER, self.name), session.graph)
            print("\nTraining a new network\n")
        return start_iteration

    def __save_network__(self, session, saver, iteration):
        session.run(self.iteration.assign(iteration))
        saver.save(session, os.path.join(NETWORK_FOLDER, self.name))
        return timer()
