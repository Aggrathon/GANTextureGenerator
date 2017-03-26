import os
import tensorflow as tf

class ImageManager():

    def __init__(self, input_folder, output_size):
        #File reading
        filename_queue = tf.train.string_input_producer(
            tf.train.match_filenames_once(os.path.join(input_folder, "*.png")))
        image_reader = tf.WholeFileReader()
        _, value = image_reader.read(filename_queue)
        self.images = tf.image.decode_png(value)
        self.images.set_shape((output_size, output_size, 3))
        #File writing
        self.output_file = tf.placeholder(tf.uint8, [output_size, output_size, 3])
        self.writer = tf.image.encode_png(self.output_file, 5)
        self.image_size = output_size

    def read(self, session):
        tf.train.start_queue_runners(sess=session)
        return self.images

    def write(self, session, image, path):
        f = open(path, "wb+")
        image = tf.reshape(image,(self.image_size, self.image_size, 3))
        f.write(session.run(self.writer, feed_dict={self.output_file: image}))
        f.close()
