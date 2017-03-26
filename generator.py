import os
import tensorflow as tf
import numpy as np
from image import ImageManager


INPUT_FOLDER = 'input'
OUTPUT_FOLDER = 'output'
NETWORK_FOLDER = 'network'


def setup_folders():
    os.makedirs(INPUT_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(NETWORK_FOLDER, exist_ok=True)


class Generator():

    def __init__(self, name, image_size=64, hidden_layers=2, input_size=128, dropout=0.4, batch_size=64):
        setup_folders()
        self.name = name
        self.session = None
        output_size = image_size*image_size*3
        self.layer_data = \
            [int(input_size+i/(hidden_layers+1)*(output_size-input_size)) for i in range(hidden_layers+1)] \
            + [output_size]
        self.image_manager = ImageManager(INPUT_FOLDER, image_size)
        self.batch_size = batch_size

        #Handle input
        self.input = tf.placeholder(tf.float32, [None, input_size])

        #Create hidden layers
        prev_layer = self.input
        self.theta = []
        drop_prob = tf.Variable(tf.constant(dropout, tf.float32))
        for i in range(hidden_layers+1):
            w = tf.Variable(tf.truncated_normal([self.layer_data[i], self.layer_data[i+1]], stddev=0.1))
            b = tf.Variable(tf.constant(0.1, shape=[self.layer_data[i+1]]))
            prev_layer = tf.nn.relu(tf.matmul(prev_layer, w) + b)
            prev_layer = tf.nn.dropout(prev_layer, drop_prob)
            self.theta.append(w)
            self.theta.append(b)

        #Handle output
        self.output = prev_layer# tf.placeholder(tf.uint8, [None, output_size])


    def random_input(self, n=1):
        return np.random.uniform(-1., 1., size=[n, self.layer_data[0]])


    def pretrain(self, epochs = 10):
        y = tf.placeholder(tf.float32, [None,self.layer_data[-1]])
        loss = tf.reduce_mean(tf.nn.l2_loss(tf.pow(self.output-y, 2)))
        train_step = tf.train.AdamOptimizer().minimize(loss, var_list=self.theta)
        self.__create_session__()
        self.image_manager.read(self.session)
        for i in range(epochs):
            self.session.run(train_step, feed_dict={
                self.input: self.random_input(self.batch_size),
                y: tf.to_float(
                    tf.reshape(
                        tf.train.shuffle_batch(
                            [self.image_manager.images], self.batch_size, 256, 128),
                        [self.batch_size, self.layer_data[-1]]))
                })
            print("epoch %d"%i)


    def generate(self, save=False):
        self.__create_session__()
        image = self.session.run(self.output, feed_dict={self.input: self.random_input()})
        if save:
            path = os.path.join(OUTPUT_FOLDER, "generated.png")
            self.image_manager.write(self.session, tf.cast(image, tf.uint8), path)
        return image


    def __create_session__(self):
        if self.session is None:
            self.session = tf.Session()
            self.session.run(tf.global_variables_initializer())

if __name__ == "__main__":
    gen = Generator("test")
    gen.pretrain()
    gen.generate(True)