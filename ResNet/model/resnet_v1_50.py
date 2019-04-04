
"""ResNet model.
Related papers:
https://arxiv.org/pdf/1512.03385v1.pdf
"""

import tensorflow as tf
import utils_mg as utils
import network_base
import numpy as np


class ResNet(network_base.Network):
    """ResNet model."""

    def __init__(self, num_classes, lrn_rate_placeholder, wd_rate_placeholder, wd_rate_placeholder2,
                 mode='train', initializer='he', fix_blocks=0,
                 RV=False, fine_tune_filename=None,
                 bn_ema=0.9, bn_epsilon=1e-5, norm_only=False,
                 wd_mode=0, optimizer='mom', momentum=0.9, fisher_filename=None,
                 gpu_num=1, fisher_epsilon=0, data_format='NHWC',
                 resnet='resnet_v1_50', strides=None, filters=None, num_residual_units=None, rate=None,
                 float_type=tf.float32, separate_regularization=False):
        """ResNet constructor.
        Args:
          mode: One of 'train' and 'test'.
        """
        super(ResNet, self).__init__(num_classes, lrn_rate_placeholder, wd_rate_placeholder, wd_rate_placeholder2,
                                     mode, initializer, fix_blocks,
                                     RV, fine_tune_filename,
                                     wd_mode, optimizer, momentum, fisher_filename,
                                     gpu_num, fisher_epsilon, data_format, float_type, separate_regularization)

        # ============== for bn layers ================
        self.bn_ema = bn_ema
        self.bn_epsilon = bn_epsilon
        self.bn_use_gamma = True
        self.bn_use_beta = True
        if norm_only is True:
            self.bn_use_gamma = False
            self.bn_use_beta = False

        # ============ network structure ==============
        self.strides = [2, 2, 2, 1]
        self.filters = [256, 512, 1024, 2048]
        self.num_residual_units = [3, 4, 23, 3]
        self.rate = [1, 1, 1, 1]
        if resnet is None:
            self.strides = strides
            self.filters = filters
            self.num_residual_units = num_residual_units
            self.rate = rate
        elif resnet == 'resnet_v1_50':
            self.num_residual_units = [3, 4, 6, 3]
        elif resnet == 'resnet_v1_101':
            self.num_residual_units = [3, 4, 23, 3]
        elif resnet == 'resnet_v1_152':
            self.num_residual_units = [3, 8, 36, 3]
        else:
            print '... ERROR from resnet.py ... '

    def inference(self, images):
        print '================== Resnet structure ======================='
        print 'num_residual_units: ', self.num_residual_units
        print 'channels in each block: ', self.filters
        print 'stride in each block: ', self.strides
        print '================== constructing network ===================='

        self.image_shape = images[0].get_shape().as_list()
        self.image_shape_tensor = tf.shape(images[0])
        # images = tf.cast(images, self.float_type)
        # print self.image_shape
        x = utils.input_data(images, self.data_format)
        # x = tf.cast(x, self.float_type)

        print 'shape input: ', x[0].get_shape()
        with tf.variable_scope('conv1'):
            trainable_ = False if self.fix_blocks > 0 else True
            self.fix_blocks -= 1
            x = utils.conv2d_same(x, 64, 7, 2,
                                  trainable=trainable_, data_format=self.data_format, initializer=self.initializer,
                                  float_type=self.float_type)
            x = utils.batch_norm('BatchNorm', x, trainable_, self.data_format, self.mode,
                                 use_gamma=self.bn_use_gamma, use_beta=self.bn_use_beta,
                                 bn_epsilon=self.bn_epsilon, bn_ema=self.bn_ema,
                                 float_type=self.float_type)
            x = utils.relu(x)
            x = utils.max_pool(x, 3, 2, self.data_format)
        print 'shape after pool1: ', x[0].get_shape()

        for block_index in range(len(self.num_residual_units)):
            for unit_index in range(self.num_residual_units[block_index]):
                with tf.variable_scope('block%d' % (block_index+1)):
                    with tf.variable_scope('unit_%d' % (unit_index+1)):
                        stride = 1
                        if unit_index == self.num_residual_units[block_index] - 1:
                            stride = self.strides[block_index]

                        trainable_ = False if self.fix_blocks > 0 else True
                        self.fix_blocks -= 1
                        x = utils.bottleneck_residual(x, self.filters[block_index], stride,
                                                      data_format=self.data_format, initializer=self.initializer,
                                                      rate=self.rate[block_index],
                                                      trainable=trainable_,
                                                      bn_mode=self.mode,
                                                      bn_use_gamma=self.bn_use_gamma, bn_use_beta=self.bn_use_beta,
                                                      bn_epsilon=self.bn_epsilon, bn_ema=self.bn_ema,
                                                      float_type=self.float_type)
            print 'shape after block %d: ' % (block_index+1), x[0].get_shape()

        with tf.variable_scope('logits'):
            x = utils.global_avg_pool(x, self.data_format)
            logits = utils.fully_connected(x, self.num_classes,
                                                trainable=True,
                                                data_format=self.data_format,
                                                initializer=self.initializer,
                                                float_type=self.float_type)
        with tf.variable_scope('up_sample'):
            self.logits = utils.resize_images(logits,
                                              self.image_shape[1:3] if self.data_format == 'NHWC'
                                              else self.image_shape[2:4],
                                              self.data_format)

        print 'logits: ', self.logits[0].get_shape()
        self.probabilities = tf.nn.softmax(self.logits[0], dim=1 if self.data_format == 'NCHW' else 3)
        self.predictions = tf.argmax(self.logits[0], axis=1 if self.data_format == 'NCHW' else 3)

        print '================== network constructed ===================='
        return self.logits

    def build_train_op(self, labels, logits=None):
        self.cost = self.compute_loss(labels, logits)

        optimizer = tf.train.MomentumOptimizer(self.lrn_rate_placeholder, self.momentum)

        grads_vars = optimizer.compute_gradients(self.cost, colocate_gradients_with_ops=True)
        apply_op = optimizer.apply_gradients(grads_vars,global_step=self.global_step, name='train_step')
        train_ops = [apply_op] + self._extra_train_ops
        self.train_op = tf.group(*train_ops)
        return self.train_op

    def _normal_loss(self, logits, labels):
        print 'normal cross entropy with softmax ... '
        xent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        return tf.reduce_sum(xent)

    def compute_loss(self, list_labels, logits=None):
        self.labels = list_labels
        if logits is None:
            logits = self.logits

        total_loss = tf.convert_to_tensor(0.0, dtype=self.float_type)
        total_auxiliary_loss = tf.convert_to_tensor(0.0, dtype=self.float_type)
        num_valide_pixel = 0
        for i in range(len(list_labels)):
            with tf.device('/gpu:%d' % i):
                print 'logit size:', logits[i].get_shape()
                print 'label size:', list_labels[i].get_shape()

                logit = tf.reshape(logits[i], [-1, self.num_classes])
                label = tf.reshape(list_labels[i], [-1, ])
                indice = tf.squeeze(tf.where(tf.less_equal(label, self.num_classes - 1)), 1)
                logit = tf.gather(logit, indice)
                label = tf.cast(tf.gather(label, indice), tf.int32)
                num_valide_pixel += tf.shape(label)[0]


                loss = self._normal_loss(logit, label)

                total_loss += loss

        num_valide_pixel = tf.cast(num_valide_pixel, tf.float32)
        self.loss = tf.divide(total_loss, num_valide_pixel)

        self.wd = 0
        if self.mode == 'train':
            self.wd = self._decay(self.wd_mode)

        return self.loss + self.wd
    def _decay(self, mode):
        """L2 weight decay loss."""
        print '================== weight decay info   ===================='
        list_conv2dt = tf.get_collection('init_conv2dt_weights')

        if mode == 0:
            print 'Applying L2 regularization...'
            l2_losses_existing_layers = 0.0
            l2_losses_new_layers = 0.0
            for v in tf.trainable_variables():
                if 'weights' in v.name:
                    if any(elem in v.name for elem in self.new_layers_names):
                        print 'except ', v.name
                        l2_losses_new_layers += tf.nn.l2_loss(v)
                        continue
                    l2_losses_existing_layers += tf.nn.l2_loss(v)
            return tf.multiply(self.wd_rate_placeholder, l2_losses_existing_layers) \
                   + tf.multiply(self.wd_rate_placeholder2, l2_losses_new_layers)
        elif mode == 1:
            print 'Applying L2-SP regularization...'
            reader = tf.train.NewCheckpointReader(self.fine_tune_filename)
            l2_losses_existing_layers = 0.0
            l2_losses_new_layers = 0.0
            for v in tf.trainable_variables():
                if 'weights' in v.name:
                    if any(elem in v.name for elem in self.new_layers_names):
                        print 'except ', v.name
                        l2_losses_new_layers += tf.nn.l2_loss(v)
                        continue

                    print v.name

                    name = v.name.split(':')[0]
                    if reader.has_tensor(name):
                        pre_trained_weights = reader.get_tensor(name)
                    else:
                        name = name.split('/weights')[0]
                        for elem in list_conv2dt:
                            if elem.name.split('/Const')[0] == name:
                                pre_trained_weights = elem
                                break
                        # print v, pre_trained_weights

                    l2_losses_existing_layers += tf.nn.l2_loss(v - pre_trained_weights)
            return self.wd_rate_placeholder * l2_losses_existing_layers \
                   + self.wd_rate_placeholder2 * l2_losses_new_layers
