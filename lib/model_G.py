import tensorflow as tf
import tensorflow.contrib as tc
from lib.projector import TFProjector

class Generator(object):
    def __init__(self, cfg):
        self.name = 'g_net'
        self.has_use = False
        self.rotation_axis = cfg.rotation_axis
        assert self.rotation_axis in ['X', 'Y', 'Z', 'XYZ', "PTN"]
        self.vp_dim = 3 if self.rotation_axis in ['XYZ', 'PTN'] else 1
        self.z_dim = cfg.z_dim + self.vp_dim
        self.dim = cfg.g_dim
        self.ksize = cfg.g_ksize
        self.use_batch_norm = cfg.g_bn
        self.projector = TFProjector(cfg, verbose=0)

    def __call__(self, z, is_training, reuse=None):
        """
        Args:
            z:              (bs, self.z_dim)    noise vector
            is_training:    boolean             tensor, indicating whether it's training
        NOTE: conv3d_T doesnt support undeterministic batch size. References:
              https://github.com/tensorflow/tensorflow/issues/10520
        """
        with tf.variable_scope(self.name) as vs:
            if (reuse is None and self.has_use) or reuse:
                vs.reuse_variables()

            print("Building voxel grids generator...")
            fc1 = tc.layers.fully_connected(
                z[:,:-self.vp_dim], 256*4*4*4,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
            )

            if self.use_batch_norm:
                fc1 = tf.layers.batch_normalization(fc1, training=is_training)
            fc1 = tf.nn.relu(fc1)
            vg = tf.reshape(fc1, tf.stack([-1, 4, 4, 4, 256]))
            conv_vg1 = tf.layers.conv3d_transpose(
                vg, self.dim*2, [self.ksize, self.ksize, self.ksize],
                [2, 2, 2], padding='same', activation=None
            )
            if self.use_batch_norm:
                conv_vg1 = tf.layers.batch_normalization(conv_vg1, training=is_training)
            conv_vg1 = tf.nn.relu(conv_vg1)

            conv_vg2 = tf.layers.conv3d_transpose(
                conv_vg1, self.dim, [self.ksize, self.ksize, self.ksize],
                [2, 2, 2], padding='same', activation=None
            )
            if self.use_batch_norm:
                conv_vg2 = tf.layers.batch_normalization(conv_vg2, training=is_training)
            conv_vg2 = tf.nn.relu(conv_vg2)

            conv_vg3 = tf.layers.conv3d_transpose(
                conv_vg2, 1, [self.ksize, self.ksize, self.ksize],
                [2, 2, 2], padding='same', activation=None
            )
            conv_vg3 = tf.sigmoid(conv_vg3)
            conv_vg3 = conv_vg3[:,:,:,:,0]

            p = self.projector(conv_vg3, z[:,-self.vp_dim:])

        self.has_use = True
        return p, conv_vg3

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

