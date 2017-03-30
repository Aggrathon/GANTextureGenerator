import tensorflow as tf
from config import IMAGE_SIZE, DIS_HIDDEN_LAYERS, DIS_DROPOUT, COLORED


class Discriminator():

    def __init__(self, name, image_size=IMAGE_SIZE, hidden_layers=DIS_HIDDEN_LAYERS, dropout=DIS_DROPOUT):
        self.name = name
        input_size = image_size*image_size*(3 if COLORED else 1)
        self.layer_data = [int((1-i/(hidden_layers+1))*input_size) for i in range(hidden_layers+1)] + [1]
        self.hidden_layers = hidden_layers

        #Network
        self.input = tf.placeholder(tf.float32, shape=[None, input_size])
        self.theta = []
        self.dropout = dropout
        #Layer Variables
        for i in range(hidden_layers+1):
            w = tf.Variable(tf.truncated_normal([self.layer_data[i], self.layer_data[i+1]], stddev=0.1, name=self.name+"_gw"+str(i)))
            b = tf.Variable(tf.constant(0.1, shape=[self.layer_data[i+1]], name=self.name+"_gb"+str(i)))
            self.theta.append(w)
            self.theta.append(b)


    def get_network(self, input=None):
        if input is None:
            prev_layer = self.input
        else:
            prev_layer = input
        for i in range(self.hidden_layers+1):
            w = self.theta[i*2]
            b = self.theta[i*2+1]
            if i != self.hidden_layers:
                prev_layer = tf.nn.relu(tf.matmul(prev_layer, w) + b)
                prev_layer = tf.nn.dropout(prev_layer, self.dropout)
            else:
                prev_layer = tf.matmul(prev_layer, w) + b
        return prev_layer


