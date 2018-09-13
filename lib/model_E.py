import tensorflow as tf
import tensorflow.contrib as tc
class Encoder(object):
    """Model structure for the decoder, using the structure in PTN
    """
    def __init__(self, cfg):
        self.x_dim = [cfg.resolution, cfg.resolution, 1]
        self.name = 'encoder_net'
        self.has_use = False

        self.dim = cfg.e_dim
        self.ksize = cfg.e_ksize
        self.out_dim = cfg.z_dim
        self.viewpoints = cfg.viewpoints
        self.use_tanh = cfg.use_tanh
        self.use_bn = cfg.encoder_use_bn
        self.use_mlp = cfg.encoder_use_mlp
        self.mlp_hidden = cfg.encoder_hidden_layers
        self.weight_decay = cfg.encoder_weight_decay
        self.use_avg_pooling = cfg.encoder_use_avg_pooling

    def conv2d(self, x, filters, ksizes, strides, padding='same',  activation=None):
        if self.weight_decay > 0:
            return tf.layers.conv2d(
                    x, filters, ksizes, strides, padding=padding, activation=activation,
                    kernel_regularizer=tc.layers.l2_regularizer(scale=self.weight_decay))
        else:
            return tf.layers.conv2d(
                    x, filters, ksizes, strides, padding=padding, activation=activation)

    def __call__(self, x, is_training, reuse=None):
        with tf.variable_scope(self.name) as vs:
            if (reuse is None and self.has_use) or reuse:
                vs.reuse_variables()
            print("Building encoders")

            # Dimension sanity check
            x = tf.reshape(x, tf.stack([
                x.get_shape()[0], self.x_dim[0], self.x_dim[1], self.x_dim[2]]))
            print("Encoder input:%s"%x.get_shape()) # 32x32

            conv1 = self.conv2d(
                x, self.dim, [self.ksize, self.ksize], [2, 2],
                padding='same',  activation=None
            )
            if self.use_bn:
                conv1 = tf.layers.batch_normalization(conv1, training=is_training)
            conv1 = tf.nn.relu(conv1)
            print("Conv1:%s"%conv1.get_shape()) # 16x16

            conv2 = self.conv2d(
                conv1, self.dim*2, [self.ksize, self.ksize], [2, 2],
                padding='same', activation=None
            )
            if self.use_bn:
                conv2 = tf.layers.batch_normalization(conv2, training=is_training)
            conv2 = tf.nn.relu(conv2)
            print("Conv2:%s"%conv2.get_shape()) # 8x8

            conv3 = self.conv2d(
                conv2, self.dim*4, [self.ksize, self.ksize], [2, 2],
                padding='same', activation=None
            )
            if self.use_bn:
                conv3 = tf.layers.batch_normalization(conv3, training=is_training)
            conv3 = tf.nn.relu(conv3)
            print("Conv3:%s"%conv3.get_shape()) # 4x4

            if self.use_avg_pooling:
                conv3 = tf.layers.average_pooling2d(conv3, [4, 4], [1, 1])
                print("\tPooled:%s"%conv3.get_shape())

            flat = tf.reshape(conv3, tf.stack([conv3.get_shape()[0], -1]))
            print(flat.get_shape())

            # Regress toward the noise
            if self.use_mlp:
                fc_1_z = flat
                for h_dim in self.mlp_hidden:
                    fc_1_z = tc.layers.fully_connected(
                        fc_1_z, h_dim, activation_fn=tf.identity)
                    if self.use_bn:
                        fc_1_z = tf.layers.batch_normalization(fc_1_z, training=is_training)
                    fc_1_z = tf.nn.relu(fc_1_z)
                    print("\tHidden:%s"%fc_1_z.get_shape())
                flat_noise = fc_1_z
            else:
                flat_noise = flat

            print("Noise output:%s"%flat_noise.get_shape())
            pred_noise = tc.layers.fully_connected(
                flat_noise, self.out_dim,
                activation_fn=tf.identity)
            if self.use_tanh:
                pred_noise = tf.tanh(pred_noise)

            # Output he logits
            if self.use_mlp:
                fc_1_p = flat
                for h_dim in self.mlp_hidden:
                    fc_1_p = tc.layers.fully_connected(
                        fc_1_p, h_dim, activation_fn=tf.identity)
                    if self.use_bn:
                        fc_1_p = tf.layers.batch_normalization(fc_1_p, training=is_training)
                    fc_1_p = tf.nn.relu(fc_1_p)
                    print("\tHidden:%s"%fc_1_p.get_shape())
                flat_logits = fc_1_p
            else:
                flat_logits = flat
            print("Pose output:%s"%flat_logits.get_shape())
            pred_logits = tc.layers.fully_connected(
                flat_logits, self.viewpoints,
                activation_fn=tf.identity)

            print("Finished building encoders")
        self.has_use = True
        return pred_noise, pred_logits

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

