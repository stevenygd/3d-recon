import os
import time
import numpy as np
import tensorflow as tf
import progressbar
from utils import average_precision, iou_t

class DCGANCycEncDecRandomVP(object):
    name = "gan_cyc_encdec_randomvp"

    def __init__(self, g_net, e_net, d_net, x_sampler, z_sampler, val_sampler,
                 prefix, config, verbose=1):

        self.g_net = g_net
        self.e_net = e_net
        self.d_net = d_net
        self.x_sampler = x_sampler
        self.z_sampler = z_sampler
        self.v_sampler = val_sampler
        self.prefix = prefix
        self.config = config
        self.verbose = verbose

        # Input variables
        encdec_inputs, gan_inputs = self.x_sampler(self.config.batch_size)
        self.x_1 = encdec_inputs['img_1']
        self.x_2 = encdec_inputs['img_2']
        self.x_3 = gan_inputs['img']
        self.y_1 = encdec_inputs['pos_1']
        self.y_2 = encdec_inputs['pos_2']
        self.z = self.z_sampler(self.config.batch_size, self.g_net.z_dim)
        self.is_training = tf.placeholder(tf.bool, name='is_training')

        # Summary list
        self.auto_summary_lst = []
        self.gan_summary_lst = []

        self.x_, _ = self.g_net(self.z, self.is_training)
        self.d = self.d_net(self.x_3, is_training=self.is_training)    # discriminate real
        self.d_ = self.d_net(self.x_, is_training=self.is_training)    # discriminate fake
        self.gan_summary_lst.append(tf.summary.image("gan_fake", self.x_, max_outputs=12))
        self.gan_summary_lst.append(tf.summary.image("gan_real", self.x_3, max_outputs=12))

        # Adding Encoder pass on top of he GAN training
        self.gan_pred_noise, self.gan_pred_logits = self.e_net(self.x_, self.is_training)

        self.gan_summary_lst.append(tf.summary.histogram("sample_z", self.z))

        # build GAN losses
        if self.config.gan_loss_type == 'DCGAN':
            gan_losses = self._build_nonsaturating_gan_objective_(
                    self.d, self.d_, self.x_3, self.x_)
        elif self.config.gan_loss_type == 'LSGAN':
            gan_losses = self._build_least_square_gan_objective_(
                    self.d, self.d_, self.x_3, self.x_)
        else:
            raise Exception("Invalid gan type:%s"%self.config.gan_loss_type)

        self.g_loss = gan_losses['g_loss']
        self.d_loss = gan_losses['d_loss']
        self.d_acc  = gan_losses['d_acc']

        ########################################
        # Building graph Cycle Encoder-Decoder #
        ########################################
        self.pred_noise_1, self.pred_logits_1 = self.e_net(self.x_1, self.is_training)
        self.pred_noise_2, self.pred_logits_2 = self.e_net(self.x_2, self.is_training)
        self.val_z_1 = self.make_generator_noise(self.pred_noise_1, self.y_2)
        self.val_z_2 = self.make_generator_noise(self.pred_noise_2, self.y_1)
        self.val_imgs_1, self.val_vox_1 = self.g_net(self.val_z_1, self.is_training)
        self.val_imgs_2, self.val_vox_2 = self.g_net(self.val_z_2, self.is_training)

        if self.config.use_auto_d_update:
            self.d_val = self.d_net(self.val_imgs, is_training=self.is_training)

        self.auto_summary_lst.append(tf.summary.histogram("val_z_1", self.val_z_1))
        self.auto_summary_lst.append(tf.summary.histogram("val_z_2", self.val_z_2))
        self.auto_summary_lst.append(
                tf.summary.image("val_fake_1", self.val_imgs_1, max_outputs=12))
        self.auto_summary_lst.append(
                tf.summary.image("val_fake_2", self.val_imgs_2, max_outputs=12))
        self.auto_summary_lst.append(tf.summary.image("real_1", self.x_1, max_outputs=12))
        self.auto_summary_lst.append(tf.summary.image("real_2", self.x_2, max_outputs=12))

        #######################################
        # Building loss Cycle Encoder-Decoder #
        #######################################
        def build_content_loss(x, y):
            ret = 0.
            if 'l1' in self.config.img_loss_types:
                ret += tf.reduce_mean(tf.abs(x - y))
            if 'l2' in self.config.img_loss_types:
                ret += tf.reduce_mean(tf.losses.mean_squared_error(x, y))
            return ret

        self.loss_content = build_content_loss(self.x_1, self.val_imgs_2) \
                          + build_content_loss(self.x_2, self.val_imgs_1)

        #  TODO: add quaternion loss
        # def build_clf_loss(labels, logits):
        #     return tf.reduce_mean(
        #         tf.nn.softmax_cross_entropy_with_logits(
        #             logits=logits, labels=labels))
        # self.loss_clf = build_clf_loss(self.y_1, self.pred_logits_1) \
        #               + build_clf_loss(self.y_2, self.pred_logits_2)

        self.pose_inv_loss = tf.reduce_mean(
                tf.losses.mean_squared_error(self.pred_noise_1, self.pred_noise_2))

        self.vox_inv_loss = 0
        print("Building voxel invariant loss:%s"%self.config.vox_inv_loss_types)
        if 'l2' in self.config.vox_inv_loss_types:
            print("Added L2 loss")
            self.vox_inv_loss += tf.reduce_mean(
                    tf.losses.mean_squared_error(self.val_vox_1, self.val_vox_2)
            ) * self.config.vox_inv_loss_types['l2']
        if 'l1' in self.config.vox_inv_loss_types:
            print("Added L1 loss")
            self.vox_inv_loss += tf.reduce_mean(
                tf.losses.absolute_difference(self.val_vox_1, self.val_vox_2)
            ) * self.config.vox_inv_loss_types['l1']

        self.loss = 0
        if self.config.content_loss_weight > 0:
            self.loss += self.loss_content * self.config.content_loss_weight
        if self.config.pose_inv_loss_weight > 0:
            self.loss += self.pose_inv_loss * self.config.pose_inv_loss_weight
        if self.config.vox_inv_loss_weight > 0:
            self.loss += self.vox_inv_loss * self.config.vox_inv_loss_weight

        self.auto_summary_lst.append(tf.summary.scalar('loss', self.loss))
        self.auto_summary_lst.append(tf.summary.scalar('content_loss', self.loss_content))

        if self.config.pose_inv_loss_weight > 0:
            self.auto_summary_lst.append(tf.summary.scalar('pose_inv_loss', self.pose_inv_loss))
        if self.config.vox_inv_loss_weight > 0:
            self.auto_summary_lst.append(tf.summary.scalar('vox_inv_loss', self.vox_inv_loss))

        if self.config.use_auto_d_update:
            self.loss_adv = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.d_val, labels=tf.ones_like(self.d_val)
            ))

            # Build adversarial loss for the GAN
            self.auto_d_loss_fake = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.d_val, labels=tf.zeros_like(self.d_val)
            ))
            self.auto_d_loss_real = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.d,  labels=tf.ones_like(self.d)
            )) # use exactly the same batch as the GAN pass.
            self.auto_d_loss = 0.5 * (self.auto_d_loss_fake + self.auto_d_loss_real)

            self.auto_gp = self._build_gp(self.x_3, self.x_)
            self.auto_d_loss += self.config.scale*self.auto_gp
            self.auto_summary_lst.append(tf.summary.scalar("auto_gp", self.auto_gp))

            self.auto_d_acc = 0.5 * tf.reduce_mean(tf.cast(self.d_val < 0, tf.float32)) \
                            + 0.5 * tf.reduce_mean(tf.cast(self.d >= 0, tf.float32))

            self.loss += self.loss_adv * self.config.adv_loss_weight
            self.auto_summary_lst.append(tf.summary.scalar('adv_loss',      self.loss_adv))
            self.auto_summary_lst.append(tf.summary.scalar('auto_d_acc',    self.auto_d_acc))
            self.auto_summary_lst.append(tf.summary.scalar('auto_d_loss',   self.auto_d_loss))

        if self.verbose > 1:
            for v in self.g_net.vars:
                s = tf.summary.histogram(v.name, v)
                self.gan_summary_lst.append(s)
                self.auto_summary_lst.append(s)
            for v in self.e_net.vars:
                self.auto_summary_lst.append(tf.summary.histogram(v.name, v))
            for v in self.d_net.vars:
                self.gan_summary_lst.append(tf.summary.histogram(v.name, v))

        self._build_optimizer()
        self._build_validation_pass_()

        # Other options
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.gan_summary = tf.summary.merge(self.gan_summary_lst)
        self.auto_summary = tf.summary.merge(self.auto_summary_lst)

        self.log_dir = "log/train_%s/%s"%(self.name, self.prefix)

        if os.path.isdir(self.log_dir):
            raise Exception("Log path :%s already exists"%self.log_dir)
        else:
            os.makedirs(self.log_dir)

        self.log_writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
        self.saver = tf.train.Saver()
        self.best_max_iou_saver = tf.train.Saver(max_to_keep=1)
        self.best_avg_prc_saver = tf.train.Saver(max_to_keep=1)
        self.best_iou_t04_saver = tf.train.Saver(max_to_keep=1)
        self.best_iou_t05_saver = tf.train.Saver(max_to_keep=1)

    def _build_optimizer(self):
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            print("Building optimizers...")
            print("\tE lr:%.10f"%self.config.e_lr)
            print("\tD lr:%.10f"%self.config.d_lr)
            print("\tG lr:%.10f"%self.config.g_lr)
            print("\tOptimizer type:%s"%self.config.optimizer_type)

            self.e_global_steps = tf.Variable(0, trainable=False)
            self.e_decayed_lr = tf.train.exponential_decay(
                self.config.e_lr, self.e_global_steps,
                self.config.decay_steps, self.config.decay_rate, staircase=True)
            if self.config.optimizer_type.lower() == 'adam':
                self.e_opt = tf.train.AdamOptimizer(learning_rate=self.e_decayed_lr)
            elif self.config.optimizer_type.lower() == 'rmsprop':
                self.e_opt = tf.train.RMSPropOptimizer(learning_rate=self.e_decayed_lr)

            self.auto_e_adam = self.e_opt.minimize(
                self.loss, var_list=self.e_net.vars, global_step=self.e_global_steps)
            if self.config.enc_content_loss_weight > 0 or self.config.enc_clf_loss_weight > 0:
                self.gan_e_adam = self.e_opt.minimize(
                        self.gan_enc_loss, var_list=self.e_net.vars,
                        global_step=self.e_global_steps)

            self.d_global_steps = tf.Variable(0, trainable=False)
            self.d_decayed_lr = tf.train.exponential_decay(
                self.config.d_lr, self.d_global_steps,
                self.config.decay_steps, self.config.decay_rate, staircase=True)

            if self.config.optimizer_type.lower() == 'adam':
                self.d_opt = tf.train.AdamOptimizer(
                    learning_rate=self.d_decayed_lr, beta1=0.5, beta2=0.9
                )
            elif self.config.optimizer_type.lower() == 'rmsprop':
                self.d_opt = tf.train.RMSPropOptimizer(learning_rate=self.d_decayed_lr)

            self.gan_d_adam = self.d_opt.minimize(
                self.d_loss, var_list=self.d_net.vars, global_step=self.d_global_steps)
            if self.config.use_auto_d_update:
                self.auto_d_adam = self.d_opt.minimize(
                        self.auto_d_loss, var_list=self.d_net.vars,
                        global_step=self.d_global_steps)

            self.g_global_steps = tf.Variable(0, trainable=False)
            self.g_decayed_lr = tf.train.exponential_decay(
                self.config.g_lr, self.g_global_steps, self.config.decay_steps,
                self.config.decay_rate, staircase=True)
            if self.config.optimizer_type.lower() == 'adam':
                self.g_opt = tf.train.AdamOptimizer(
                    learning_rate=self.g_decayed_lr, beta1=0.5, beta2=0.9
                )
            elif self.config.optimizer_type.lower() == 'rmsprop':
                self.g_opt = tf.train.RMSPropOptimizer(learning_rate=self.g_decayed_lr)
            self.gan_g_adam = self.g_opt.minimize(
                self.g_loss, var_list=self.g_net.vars, global_step=self.g_global_steps)
            self.auto_g_adam = self.g_opt.minimize(
                self.loss, var_list=self.g_net.vars, global_step=self.g_global_steps)

            self.auto_summary_lst.append(
                    tf.summary.scalar('e_global_steps', self.e_global_steps))
            self.auto_summary_lst.append(
                    tf.summary.scalar('d_global_steps', self.d_global_steps))
            self.auto_summary_lst.append(
                    tf.summary.scalar('g_global_steps', self.g_global_steps))
            self.auto_summary_lst.append(tf.summary.scalar('e_decayed_lr', self.e_decayed_lr))
            self.auto_summary_lst.append(tf.summary.scalar('d_decayed_lr', self.d_decayed_lr))
            self.auto_summary_lst.append(tf.summary.scalar('g_decayed_lr', self.g_decayed_lr))


    def _build_least_square_gan_objective_(self,
            d_real, d_fake, x_real, x_fake, a=0, b=1, c=1):
        d_fake = tf.sigmoid(d_fake)
        d_real = tf.sigmoid(d_real)
        d_loss_fake = tf.reduce_mean(tf.square(d_fake - a))
        d_loss_real = tf.reduce_mean(tf.square(d_real - b))
        d_loss      = 0.5 * (d_loss_fake + d_loss_real)
        g_loss      = 0.5 * tf.reduce_mean(tf.square(d_fake - c))

        d_acc = 0.5 * tf.reduce_mean(tf.cast(d_fake < (a + b)/2., tf.float32)) \
              + 0.5 * tf.reduce_mean(tf.cast(d_real >= (a + b)/2., tf.float32))

        self.gan_summary_lst.append(tf.summary.scalar("d_loss_real", d_loss_real))
        self.gan_summary_lst.append(tf.summary.scalar("d_loss_fake", d_loss_fake))
        self.gan_summary_lst.append(tf.summary.scalar("d_loss",      d_loss))
        self.gan_summary_lst.append(tf.summary.scalar("d_acc",       d_acc))
        self.gan_summary_lst.append(tf.summary.scalar("g_loss",      g_loss))

        return {
            "d_loss" : d_loss,
            "g_loss" : g_loss,
            "d_acc"  : d_acc,
        }



    def _build_nonsaturating_gan_objective_(self, d, d_, x, x_):
        ######################################
        # Normal nonsatruating GAN objective #
        ######################################
        g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_, labels=tf.ones_like(d_)
        ))
        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_, labels=tf.zeros_like(d_)
        ))
        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d,  labels=tf.ones_like(d)
        ))
        d_loss = 0.5 * (d_loss_fake + d_loss_real)

        self.gan_gp = self._build_gp(x, x_)
        d_loss += self.config.scale*self.gan_gp
        self.gan_summary_lst.append(tf.summary.scalar("gan_gp", self.gan_gp))

        d_acc = 0.5 * tf.reduce_mean(tf.cast(d_ < 0, tf.float32)) \
              + 0.5 * tf.reduce_mean(tf.cast(d >= 0, tf.float32))

        self.gan_summary_lst.append(tf.summary.scalar("d_loss_real", d_loss_real))
        self.gan_summary_lst.append(tf.summary.scalar("d_loss_fake", d_loss_fake))
        self.gan_summary_lst.append(tf.summary.scalar("d_loss",      d_loss))
        self.gan_summary_lst.append(tf.summary.scalar("d_acc",       d_acc))
        self.gan_summary_lst.append(tf.summary.scalar("g_loss",      g_loss))

        return {
            "d_loss" : d_loss,
            "g_loss" : g_loss,
            "d_acc"  : d_acc,
        }

    def _build_validation_pass_(self):
        ###################
        # validation pass #
        ###################
        # self.val_pos_acc = tf.placeholder(tf.float32)
        self.val_max_iou = tf.placeholder(tf.float32)
        self.val_t04_iou = tf.placeholder(tf.float32)
        self.val_t05_iou = tf.placeholder(tf.float32)
        self.val_avg_prc = tf.placeholder(tf.float32)
        self.val_iou_thr = tf.placeholder(tf.float32)

        self.val_max_iou_best = tf.placeholder(tf.float32)
        self.val_t04_iou_best = tf.placeholder(tf.float32)
        self.val_t05_iou_best = tf.placeholder(tf.float32)
        self.val_avg_prc_best = tf.placeholder(tf.float32)
        self.val_iou_thr_best = tf.placeholder(tf.float32)

        # Validation pass
        self.val_init, val_inputs = self.v_sampler(self.config.batch_size)
        self.x_val_1 = val_inputs["image"]
        self.y_val_1 = val_inputs["pose"]
        self.val_vox = val_inputs["vox"]
        self.val_pass_noise, self.val_pass_pose_logits = self.e_net(
                self.x_val_1, self.is_training)
        self.val_pass_z = self.make_generator_noise(self.val_pass_noise, self.y_val_1)
        _, self.val_pass_vox = self.g_net(self.val_pass_z, self.is_training)

        val_lst = []
        val_lst.append(tf.summary.scalar("val_max_iou", self.val_max_iou))
        val_lst.append(tf.summary.scalar("val_t04_iou", self.val_t04_iou))
        val_lst.append(tf.summary.scalar("val_t05_iou", self.val_t05_iou))
        val_lst.append(tf.summary.scalar("val_avg_prc", self.val_avg_prc))
        val_lst.append(tf.summary.scalar("val_iou_thr", self.val_iou_thr))

        val_lst.append(tf.summary.scalar("val_max_iou_best", self.val_max_iou_best))
        val_lst.append(tf.summary.scalar("val_t04_iou_best", self.val_t04_iou_best))
        val_lst.append(tf.summary.scalar("val_t05_iou_best", self.val_t05_iou_best))
        val_lst.append(tf.summary.scalar("val_avg_prc_best", self.val_avg_prc_best))
        val_lst.append(tf.summary.scalar("val_iou_thr_best", self.val_iou_thr_best))

        self.val_summary = tf.summary.merge(val_lst)


    def make_generator_noise(self, noise_t, pose_t):
        """
        Args:
            [noise_t] the noise tensor encoder generated.
            [pose_t]  the ground truth pose (in rotation degrees)
        """
        return tf.concat([noise_t, pose_t], axis=1)

    def _build_gp(self, x_real, x_fake):
        # For GAN part: x_fake = self.x_, x_real=self.x_3
        # Compute gradient penalety
        if self.config.gp == 'wgan':
            alpha = tf.random_uniform(
                shape=[self.config.batch_size,1],
                minval=0.,
                maxval=1.
            )
            fake_data = tf.reshape(x_fake, [self.config.batch_size, -1])
            real_data = tf.reshape(x_real, [self.config.batch_size, -1])
            differences = fake_data - real_data
            interpolates = real_data + (alpha*differences)
            gradients = tf.gradients(self.d_net(interpolates), [interpolates])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
            gradient_penalty = tf.reduce_mean((slopes-1.)**2)
            return gradient_penalty

        elif self.config.gp == 'dragan':
            real_data = tf.reshape(x_real,  [self.config.batch_size, -1])
            # TODO: stddev here should be hyperparam
            noise = tf.random_normal(shape=real_data.get_shape(),
                                     mean=0., stddev=10/255.)
            x_noise = real_data + noise
            gradients = tf.gradients(self.d_net(x_noise), [x_noise])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
            gradient_penalty = tf.reduce_mean((slopes-1.)**2)
            return gradient_penalty
        else:
            print("No gradient panelty")
            gradient_penalty = 0
            return gradient_penalty


    def _sample_data(self, is_training=True):
        return {self.is_training : is_training}

    def _gan_step(self, update_g=True, update_d=True, update_e=True):
        if self.gan_t % 100 == 0 or self.gan_t < 250:
            feed_dict = self._sample_data()
            d_acc, d_loss, g_loss, summary = self.sess.run(
                [self.d_acc, self.d_loss, self.g_loss, self.gan_summary],
                feed_dict
            )
            self.log_writer.add_summary(summary, self.gan_t)
            print('Iter [%8d] Time [%5.4f] d_loss [%.4f] g_loss [%.4f] d_acc [%.4f]' %
                    (self.gan_t, time.time() - self.start_time, d_loss, g_loss, d_acc))

        if self.gan_t > 0 and update_g:
            for _ in range((self.config.g_iters)(self.gan_t)):
                self.sess.run([self.gan_g_adam], feed_dict=self._sample_data())

        if update_e:
            if self.config.enc_content_loss_weight > 0 or self.config.enc_clf_loss_weight > 0:
                self.sess.run([self.gan_e_adam], feed_dict=self._sample_data())

        if update_d:
            for _ in range((self.config.d_iters)(self.gan_t)):
                feed_dict = self._sample_data()
                d_acc = self.sess.run(self.d_acc, feed_dict)
                if d_acc < self.config.max_d_acc:
                    self.sess.run([self.gan_d_adam], feed_dict)

        self.gan_t += 1

    def _autoencoder_step(self, update_g=True, update_e=True, update_d=True):
        # Train AutoEncoder
        feed_dict = self._sample_data()
        if self.encdec_t < 250 or self.encdec_t % 100 == 0:
            loss, content_loss, summary = self.sess.run(
                [self.loss, self.loss_content, self.auto_summary],
                feed_dict
            )
            self.log_writer.add_summary(summary, self.encdec_t)
            print('Iter [%8d] Time [%5.4f] loss [%.4f] content [%.4f]' \
                % (self.encdec_t, time.time() - self.start_time, loss, content_loss))

        # Update AutoEncoder
        if update_e:
            self.sess.run(self.auto_e_adam, feed_dict)
        if update_g:
            self.sess.run(self.auto_g_adam, feed_dict)

        if self.config.use_auto_d_update and update_d:
            self.sess.run(self.auto_d_adam, feed_dict)

        self.encdec_t += 1

    def _save_model(self, t, saver=None, prefix="model"):
        save_path = os.path.join(self.log_dir, "%s-%d.ckpt"%(prefix, t))
        if saver == None:
            saver = self.saver
        saver.save(self.sess, save_path)

    def _initialize_training(self):
        self.t = 0
        self.gan_t = 0
        self.encdec_t = 0

        self.max_ap, self.max_iou_max, self.max_iou_t4, self.max_iou_t5 = 0., 0., 0., 0.
        self.max_iou_thr = -1.
        self.sess.run(tf.global_variables_initializer())
        self.start_time = time.time()

    def train(self, num_batches=1000000, num_autoencoder_iters=1, num_gan_iters=1,
              max_validation_batches=None, resume=None,
              update_g=True, update_d=True, update_e=True):
        self._initialize_training()
        print("Start training")
        self._save_model(self.t) #  first one for sanity check and fail early

        print("Resume=%s"%resume)
        if resume != None:
            print("Resume from %s"%resume)
            self.saver.restore(self.sess, resume)

        self.validation(max_validation_batches, self.t)

        while self.t < num_batches:
            for _ in range(num_autoencoder_iters):
                self._autoencoder_step(update_g=update_g, update_d=update_d, update_e=update_e)
                self.t += 1
                # Save model
                if self.t % 1000 == 0: # just to remind me :)
                    self._save_model(self.t)

                if self.t % self.config.validation_interval == 0:
                    print("Model:%s"%self.log_dir)
                    self.validation(max_validation_batches, self.t)

            for _ in range(num_gan_iters):
                self._gan_step(update_g=update_g, update_d=update_d, update_e=update_e)
                self.t += 1

                # Save model
                if self.t % 1000 == 0: # just to remind me :)
                    print("Model:%s"%self.log_dir)

                # if (self.t % 2000 == 0 and self.t < 10000) or self.t % 5000 == 0:
                if self.t % self.config.validation_interval == 0:
                    self._save_model(self.t)
                    self.validation(max_validation_batches, self.t)


    def validation(self, max_validation_batches, t):
        print("="*80)
        print("Validation:")

        num_batches = 0
        ap      = 0
        iou_t4  = 0
        iou_t5  = 0
        ts      = np.arange(0., 1., 1e-1)
        iou_max = np.zeros(ts.shape)

        pbar = progressbar.ProgressBar(maxval=progressbar.UnknownLength)
        self.sess.run(self.val_init)
        try:
            while True:
                vox_p, vox_1 = self.sess.run(
                    [self.val_pass_vox, self.val_vox],
                    feed_dict = {self.is_training : False}
                )

                ious = []
                for t in ts:
                    iou = iou_t(vox_1, vox_p, threshold=t).mean()
                    ious.append(iou)
                iou_max += np.array(ious)
                ap      += average_precision(vox_1, vox_p)
                iou_t4  += iou_t(vox_1, vox_p, threshold=0.4).mean()
                iou_t5  += iou_t(vox_1, vox_p, threshold=0.5).mean()

                num_batches += 1
                pbar.update(num_batches)
                if num_batches == max_validation_batches:
                    break

        except tf.errors.OutOfRangeError:
            print("End of validation dataset")

        # # pos_acc  /= float(num_batches)
        iou_t4   /= float(num_batches)
        iou_t5   /= float(num_batches)
        ap       /= float(num_batches)

        iou_max  /= float(num_batches)
        iou_thr  =  iou_max.argmax() * (1/float(len(iou_max)))
        iou_max  =  iou_max.max()

        print("Performance (this pass)")
        print("\tAP:%.5f\tMaxIoU:%.5f\tIoU(t>0.4):%.5f\tIoU(t>0.5):%.5f"\
              %(ap, iou_max, iou_t4, iou_t5))

        if ap > self.max_ap:
            self.max_ap = ap
            print("New best avg prc:%.5f"%self.max_ap)
            self._save_model(self.t, saver=self.best_avg_prc_saver, prefix="model-best-avgprc")

        if iou_max > self.max_iou_max:
            self.max_iou_max = iou_max
            self.max_iou_thr = iou_thr
            print("New best max iou:%.5f, %.5f"%(self.max_iou_max, self.max_iou_thr))
            self._save_model(self.t, saver=self.best_max_iou_saver, prefix="model-best-maxiou")

        if iou_t4 > self.max_iou_t4:
            self.max_iou_t4 = iou_t4
            print("New best IoU(t04):%.5f"%self.max_iou_t4)
            self._save_model(self.t, saver=self.best_iou_t04_saver, prefix="model-best-iout04")

        if iou_t5 > self.max_iou_t5:
            self.max_iou_t5 = iou_t5
            print("New best IoU(t05):%.5f"%self.max_iou_t5)
            self._save_model(self.t, saver=self.best_iou_t05_saver, prefix="model-best-iout05")

        print("Performance (best pass)")
        print("\tAP:%.5f\tMaxIoU:%.5f\tIoU(t>0.4):%.5f\tIoU(t>0.5):%.5f"\
              %(self.max_ap, self.max_iou_max, self.max_iou_t4, self.max_iou_t5))

        val_summary = self.sess.run(self.val_summary, feed_dict = {
                self.val_max_iou : iou_max,
                self.val_t04_iou : iou_t4,
                self.val_t05_iou : iou_t5,
                self.val_avg_prc : ap,
                self.val_iou_thr : iou_thr,
                self.val_max_iou_best : self.max_iou_max,
                self.val_t04_iou_best : self.max_iou_t4,
                self.val_t05_iou_best : self.max_iou_t5,
                self.val_avg_prc_best : self.max_ap,
                self.val_iou_thr_best : self.max_iou_thr
            })
        self.log_writer.add_summary(val_summary, self.t)

        print("="*80)
        return ap, iou_max, iou_t4, iou_t5

