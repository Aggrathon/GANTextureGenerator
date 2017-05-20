
import tensorflow as tf
import numpy as np

from operators import linear, conv2d, relu_dropout, conv2d_transpose, conv2d_transpose_tanh, expand_relu, openif_scope


def image_encoder(input_tensors, name='encoder', image_size=64, convolutions=5, base_width=32,
                  fully_connected=1, dropout=0.4, output_size=1, logit=True):
    """Create a network for reducing an image to a 1D tensor"""
    conv_output_size = ((image_size//(2**convolutions))**2) * base_width * convolutions
    assert conv_output_size != 0, "Invalid number of convolutions compared to the image size"
    fc_output_size = max(2**int(np.log2(conv_output_size//2)), output_size)
    with tf.variable_scope(name):
        #Create Layers
        prev_layer = input_tensors
        for i in range(convolutions): #Convolutional layers
            prev_layer = conv2d(prev_layer, base_width*(i+1), name='convolution_%d'%i)
        with tf.name_scope('reshape'):
            prev_layer = [tf.reshape(layer, [-1, conv_output_size]) for layer in prev_layer]
        for i in range(fully_connected): #Fully connected layers
            prev_layer = relu_dropout(prev_layer, fc_output_size, dropout, 'fully_connected_%d'%i)
        prev_layer = linear(prev_layer, output_size, 'logit')
        if not logit:
            with tf.name_scope('output'):
                prev_layer = [tf.multiply(tf.nn.tanh(layer)+1, 0.5, name='output') for layer in prev_layer]
    return prev_layer

def image_decoder(input_tensors, name='decoder', image_size=64, convolutions=5, base_width=32,
                  input_size=64, batch_size=128, colors=3):
    """Create a network for generating an image from a 1D tensor"""
    conv_image_size = image_size // (2**convolutions)
    assert conv_image_size*(2**convolutions) == image_size, "Images must be a multiple of two (and >= 2**convolutions)"
    with tf.variable_scope(name):
        prev_layer = expand_relu(input_tensors, [-1, conv_image_size, conv_image_size, base_width*2**(convolutions-1)], 'expand')
        for i in range(convolutions):
            prev_layer = conv2d_transpose(prev_layer, batch_size, 2**(convolutions-i-1)*base_width, 'convolution_%d'%i)
        prev_layer = conv2d_transpose_tanh(prev_layer, batch_size, colors, 'output')
    return prev_layer

def image_output(input_tensors, name='output', image_size=64, grid_size=4):
    """Create operations for converting tensors into images"""
    output = []
    with tf.name_scope(name):
        with tf.name_scope("image_single"):
            output = [tf.cast((tensor + 1) * 127.5, tf.uint8) for tensor in input_tensors]
        with tf.name_scope('image_grid'):
            for i in range(len(input_tensors)):
                wh = grid_size * (image_size + 2) + 2
                grid = tf.Variable(0, trainable=False, dtype=tf.uint8, expected_shape=[wh, wh, image_size])
                for x in range(grid_size):
                    for y in range(grid_size):
                        bound = tf.to_int32(tf.image.pad_to_bounding_box(output[i][x+y*grid_size], 2 + x*(image_size + 2), 2 + y*(image_size + 2), wh, wh))
                        if x == 0 and y == 0:
                            grid = bound
                        else:
                            grid = tf.add(grid, bound)
                output.append(tf.cast(grid, tf.uint8))
    return output

def batch_optimizer(name, variables, positive_tensors=None, positive_value=1, positive_prefix='',
                    negative_tensors=None, negative_value=0, negative_prefix='',
                    learning_rate=0.001, learning_momentum=0.9, learning_momentum2=0.99, global_step=None, summary=True):
    """Create optimizer for batches with negative and positive influences"""
    with tf.variable_scope(name):
        losses = []
        if positive_tensors is not None:
            labels = tf.fill(tf.shape(positive_tensors[0]), positive_value)
            pos_los = [tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=logit, labels=labels),
                name=positive_prefix+'loss') for logit in positive_tensors]
            losses.extend(pos_los)
            if summary and negative_tensors is not None:
                tf.summary.scalar(positive_prefix+'loss', tf.add_n(pos_los))
        if negative_tensors is not None:
            labels = tf.fill(tf.shape(negative_tensors[0]), negative_value)
            neg_los = [tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=logit, labels=labels),
                name=negative_prefix+'loss') for logit in negative_tensors]
            losses.extend(neg_los)
            if summary and positive_tensors is not None:
                tf.summary.scalar(negative_prefix+'loss', tf.add_n(neg_los))
        loss = tf.add_n(losses)
        if summary:
            tf.summary.scalar('loss', loss)
        adam = tf.train.AdamOptimizer(learning_rate, learning_momentum, learning_momentum2)
        if global_step is None:
            solver = adam.minimize(loss, var_list=variables)
        else:
            solver = adam.minimize(loss, var_list=variables, global_step=global_step)
        return solver

def gan_optimizer(name, gen_vars, dis_vars, fake_tensor, real_tensor, false_val=0, real_val=1,
                  learning_rate=0.001, learning_momentum=0.9, learning_momentum2=0.99,
                  learning_rate_pivot=0, global_step=None, dicriminator_scaling_favor=4, summary=True):
    """Create an optimizer for a GAN"""
    with openif_scope(name):
        #learning rate scaling
        if learning_rate_pivot > 0 and global_step is not None:
            scaler = tf.sqrt(tf.div(tf.to_float(global_step), float(learning_rate_pivot))+1)
            learning_rate = tf.div(learning_rate, scaler)
        #generator
        with tf.variable_scope('generator'):
            gen_labels = tf.fill(tf.shape(fake_tensor), float(real_val))
            gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_tensor, labels=gen_labels), name='loss')
            if summary:
                tf.summary.scalar('loss', gen_loss)
            gen_opt = tf.train.AdamOptimizer(learning_rate, learning_momentum, learning_momentum2)
        #discriminator
        with tf.variable_scope('discriminator'):
            dis_real_labels = tf.fill(tf.shape(real_tensor), float(real_val))
            dis_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_tensor, labels=dis_real_labels), name='real_loss')
            dis_fake_labels = tf.fill(tf.shape(fake_tensor), float(false_val))
            dis_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_tensor, labels=dis_fake_labels), name='fake_loss')
            dis_loss = tf.add(dis_fake_loss, dis_real_loss*2, name="loss")
            if summary:
                tf.summary.scalar('loss', dis_loss)
                tf.summary.scalar('real_loss', dis_real_loss)
                tf.summary.scalar('fake_loss', dis_fake_loss)
            dis_opt = tf.train.AdamOptimizer(learning_rate, learning_momentum, learning_momentum2)
        scale = tf.divide(gen_loss, dis_loss*dicriminator_scaling_favor, name='scale')
        if summary:
            less = (tf.sign(scale)-1.)*0.5/scale        # -1/scale if scale < 1.0
            more = (tf.sign(scale)+1.)*0.5*(scale-1.0)  # 1*scale-1 if scale > 1.0
            tf.summary.scalar('relative_loss_comparison', more+less)
        #optimizers
        with tf.variable_scope('optimizers'):
            gen_solver = gen_opt.minimize(gen_loss, var_list=gen_vars)
            dis_solver = dis_opt.minimize(dis_loss, var_list=dis_vars)
        return gen_solver, dis_solver, scale

def image_optimizer(name, variables, real_images, fake_images,
                    learning_rate=0.001, learning_momentum=0.9, learning_momentum2=0.99, summary=True):
    """Create optimizer for images that should be the same"""
    with tf.variable_scope(name):
        loss = tf.add_n([tf.reduce_mean(tf.square(fake-real)) for fake, real in zip(fake_images, real_images)], name='loss')
        if summary:
            tf.summary.scalar('loss', loss)
        optimizer = tf.train.AdamOptimizer(learning_rate, learning_momentum, learning_momentum2)
        solver = optimizer.minimize(loss, var_list=variables)
    return solver
