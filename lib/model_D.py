import tensorflow as tf
import tensorflow.contrib as tc
import tensorflow.contrib.layers as tcl

def leaky_relu(x, alpha=0.2):
    return tf.maximum(tf.minimum(0.0, alpha * x), x)

class Discriminator(object):
    def __init__(self, cfg):
        self.has_use = False
        self.name = 'd_net'
        self.x_dim = [cfg.resolution, cfg.resolution, 1]
        self.dim = cfg.d_dim
        self.ksize = cfg.d_ksize
        self.use_bn = cfg.d_use_bn
        self.use_ln = cfg.d_use_ln
        self.input_norm=cfg.d_input_norm # whether apply batch norm in the input layer

    def __call__(self, x, is_training=False, reuse=None):
        with tf.variable_scope(self.name) as vs:
            if (reuse is None and self.has_use) or reuse:
                vs.reuse_variables()

            # Dimension sanity check
            print("Discrimnator dimensions:%s"%self.x_dim)
            print("Input dimensions:%s"%x.get_shape())
            x = tf.reshape(x, tf.stack([
                x.get_shape()[0], self.x_dim[0], self.x_dim[1], self.x_dim[2]]))

            conv1 = tf.layers.conv2d(
                x, self.dim, [self.ksize, self.ksize], [2, 2],
                padding='same', activation=None
            )
            if self.input_norm:
                if self.use_ln:
                    conv1 = tcl.layer_norm(conv1)
                elif self.use_bn:
                    conv1 = tf.layers.batch_normalization(conv1, training=is_training)
            conv1 = leaky_relu(conv1)

            conv2 = tf.layers.conv2d(
                conv1, self.dim*2, [self.ksize, self.ksize], [2, 2],
                padding='same', activation=None
            )
            if self.use_ln:
                conv2 = tcl.layer_norm(conv2)
            elif self.use_bn:
                conv2 = tf.layers.batch_normalization(conv2, training=is_training)
            conv2 = leaky_relu(conv2)

            conv3 = tf.layers.conv2d(
                conv2, self.dim*4, [self.ksize, self.ksize], [2, 2],
                padding='same', activation=None
            )
            if self.use_ln:
                conv3 = tcl.layer_norm(conv3)
            elif self.use_bn:
                conv3 = tf.layers.batch_normalization(conv3, training=is_training)
            conv3 = leaky_relu(conv3)
            flat = tf.reshape(conv3, tf.stack([conv3.get_shape()[0], -1]))

            out = tc.layers.fully_connected(
                flat, 1,
                activation_fn=tf.identity)

        self.has_use = True
        return out

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

