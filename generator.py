import os
from timeit import default_timer as timer
import tensorflow as tf
import numpy as np
from image import ImageManager
from constants import *


class Generator():

    def __init__(self, name, image_size=64, hidden_layers=2, input_size=128, dropout=0.4, batch_size=64):
        setup_folders()
        self.name = name
        self.session = None
        self.saver = None
        self.image_size = image_size
        output_size = image_size*image_size*3
        self.layer_data = \
            [int(input_size+i/(hidden_layers+1)*(output_size-input_size)) for i in range(hidden_layers+1)] \
            + [output_size]
        self.batch_size = batch_size
        #TODO Move this:
        self.image_manager = ImageManager(INPUT_FOLDER, OUTPUT_FOLDER, image_size)

        #Network
        self.input = tf.placeholder(tf.float32, [None, input_size])
        #Hidden layers
        prev_layer = self.input
        self.theta = []
        drop_prob = tf.Variable(tf.constant(dropout, tf.float32))
        for i in range(hidden_layers+1):
            w = tf.Variable(tf.truncated_normal([self.layer_data[i], self.layer_data[i+1]], stddev=0.1, name=self.name+"_w"+str(i)))
            b = tf.Variable(tf.constant(0.1, shape=[self.layer_data[i+1]], name=self.name+"_b"+str(i)))
            prev_layer = tf.nn.relu(tf.matmul(prev_layer, w) + b)
            if i != hidden_layers:
                prev_layer = tf.nn.dropout(prev_layer, drop_prob)
            self.theta.append(w)
            self.theta.append(b)
        #Output
        self.output = prev_layer


    def random_input(self, n=1):
        return np.random.uniform(-1., 1., size=[n, self.layer_data[0]])


    def pretrain(self, epochs = 200):
        time = timer()
        y = tf.placeholder(tf.float32, [None,self.layer_data[-1]])
        loss = tf.reduce_mean(tf.nn.l2_loss(tf.pow(self.output-y, 2)))
        train_step = tf.train.AdamOptimizer().minimize(loss, var_list=self.theta)
        self.__create_session__()
        for i in range(epochs):
            self.session.run(train_step, feed_dict={
                self.input: self.random_input(self.batch_size),
                y: self.image_manager.get_batch(self.batch_size)})
            if i%20 == 0:
                t = timer() - time
                print("epoch %d at %02d:%02d:%02d"%(i, t//3600, t%3600//60, t%60))
                if i%100:
                    self.generate(True, "e%d"%i)
        self.saver.save(self.session, os.path.join(NETWORK_FOLDER, self.name))
        self.__close_session__()


    def generate(self, save=False, name=None):
        self.__create_session__()
        image = self.session.run(self.output, feed_dict={self.input: self.random_input()})
        if save:
            if self.image_manager is None:
                self.image_manager = ImageManager(INPUT_FOLDER, OUTPUT_FOLDER, self.image_size, False)
            fullname = self.name
            if not name is None:
                fullname += '-'+name
            self.image_manager.write(image, fullname)
        return image


    def __create_session__(self):
        if self.session is None:
            self.session = tf.Session()
            self.session.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()
            try:
                self.saver.restore(self.session, NETWORK_FOLDER)
            except:
                print("No already trained network found, creating new")

    def __close_session__(self):
        if not self.session is None:
            self.session.close()
            self.session = None


if __name__ == "__main__":
    gen = Generator("test")
    gen.pretrain()