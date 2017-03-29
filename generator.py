import os
import tensorflow as tf
import numpy as np
from image import save_image
from config import INPUT_SIZE, NETWORK_FOLDER, GEN_DROPOUT, GEN_HIDDEN_LAYERS, IMAGE_SIZE


class Generator():

    def __init__(self, name, image_size=IMAGE_SIZE, hidden_layers=GEN_HIDDEN_LAYERS, input_size=INPUT_SIZE, dropout=GEN_DROPOUT):
        self.name = name
        self.image_size = image_size
        output_size = image_size*image_size*3
        self.layer_data = \
            [int(input_size+i/(hidden_layers+1)*(output_size-input_size)) for i in range(hidden_layers+1)] \
            + [output_size]

        #Network
        self.input = tf.placeholder(tf.float32, [None, input_size])
        #Hidden layers
        prev_layer = self.input
        self.theta = []
        drop_prob = tf.Variable(tf.constant(dropout, tf.float32))
        for i in range(hidden_layers+1):
            w = tf.Variable(tf.truncated_normal([self.layer_data[i], self.layer_data[i+1]], stddev=0.1, name=self.name+"_gw"+str(i)))
            b = tf.Variable(tf.constant(0.1, shape=[self.layer_data[i+1]], name=self.name+"_gb"+str(i)))
            prev_layer = tf.nn.relu(tf.matmul(prev_layer, w) + b)
            if i != hidden_layers:
                prev_layer = tf.nn.dropout(prev_layer, drop_prob)
            self.theta.append(w)
            self.theta.append(b)
        #Output
        self.output = prev_layer

    def random_input(self, n=1):
        return np.random.uniform(-1., 1., size=[n, self.layer_data[0]])


    def generate(self, session, amount=1, name=None):
        #Prepare Images
        fullname = self.name
        if not name is None:
            fullname += '-'+name
        images = session.run(self.output, feed_dict={self.input: self.random_input(amount)})
        #Save
        if amount == 1:
            save_image(images, self.image_size, fullname)
        else:
            for i in range(amount):
                save_image(images[i], self.image_size, fullname+"-"+str(i))

