#coding:utf-8
import sys
sys.path.append('../')

import datetime
import os

import tensorflow as tf
from experiment_manager.utils import LogDir, sorted_str_dict
from model import pspnet_mg
import math
import numpy as np
import cv2
from database.helper_segmentation import *
from database.Image_reader import ImageReader

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--network', type=str, default='pspnet',
                    help='pspnet')
parser.add_argument('--server', type=int, default=0, help='local machine 0 or server 1 or 2')
parser.add_argument('--epsilon', type=float, default=0.00001, help='epsilon in bn layers')
parser.add_argument('--norm_only', type=int, default=0,
                    help='no beta nor gamma in fused_bn (1). Or with beta and gamma(0).')
parser.add_argument('--data_type', type=int, default=32, help='float32 or float16')
parser.add_argument('--database', type=str, default='New_sonardata', help='SBD or Cityscapes.')
parser.add_argument('--resize_images_method', type=str, default='bilinear', help='resize images method: bilinear or nn')
parser.add_argument('--color_switch', type=int, default=0, help='color switch or not')
parser.add_argument('--eval_only', type=int, default=0, help='only do the evaluation (1) or do train and eval (0).')
parser.add_argument('--log_dir', type=str, default='pspmg-0', help='according to gpu index and wd method')

parser.add_argument('--train_conv2dt', type=int, default=0, help='train conv2dt instead of using resize images.')
parser.add_argument('--loss_type', type=str, default='normal', help='normal, focal_1, etc.')
parser.add_argument('--structure_in_paper', type=int, default=0, help='first conv layers')
parser.add_argument('--train_like_in_paper', type=int, default=0,
                    help='new layers receive 10 times learning rate; biases * 2')
parser.add_argument('--has_aux_loss', type=int, default=1, help='with(1) or without(0) auxiliary loss')
parser.add_argument('--new_layer_names', type=str, default=None, help='with(1) or without(0) auxiliary loss')

parser.add_argument('--subsets_for_training', type=str, default='train,val', help='whether use val set for training')
parser.add_argument('--scale_min', type=float, default=0.5, help='random scale rate min')
parser.add_argument('--scale_max', type=float, default=2.0, help='random scale rate max')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--optimizer', type=str, default='mom', help='mom, sgd, adam, more to be added')
parser.add_argument('--poly_lr', type=int, default=1, help='poly learning rate policy')
parser.add_argument('--lrn_rate', type=float, default=0.01, help='initial learning rate')
parser.add_argument('--weight_decay_mode', type=int, default=1, help='weight decay mode')
parser.add_argument('--weight_decay_rate', type=float, default=0.01, help='weight decay rate for existing layers')
parser.add_argument('--weight_decay_rate2', type=float, default=0.01, help='weight decay rate for new layers')
parser.add_argument('--train_max_iter', type=int, default=60000, help='Maximum training iteration')
parser.add_argument('--snapshot', type=int, default=10000, help='snapshot every ')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum for mom optimizer')
parser.add_argument('--fine_tune_filename', type=str,
                    default='../z_pretrained_weights/resnet_v1_50.ckpt',
                    help='fine_tune_filename')
parser.add_argument('--fisher_filename', type=str, default='./fisher_exp.npy', help='filename of fisher matrix')
parser.add_argument('--resume_step', type=int, default=None, help='resume step')
parser.add_argument('--lr_step', type=str, default=None, help='list of lr rate decreasing step. Default None.')
parser.add_argument('--step_size', type=float, default=0.1,
                    help='Each lr_step, learning rate decreases . Default to 0.1')
parser.add_argument('--gpu_num', type=int, default=4, help='gpu num')
parser.add_argument('--ema_decay', type=float, default=0.9, help='ema decay of moving average in bn layers')
parser.add_argument('--blur', type=int, default=1, help='random blur: brightness/saturation/constrast')
parser.add_argument('--random_rotate', type=int, default=1, help='random rotate')
parser.add_argument('--random_scale', type=int, default=1, help='random scale')
parser.add_argument('--initializer', type=str, default='xavier', help='he or xavier')
parser.add_argument('--fix_blocks', type=int, default=0,
                    help='number of blocks whose weights will be fixed when training')
parser.add_argument('--save_first_iteration', type=int, default=0, help='whether saving the initial model')
parser.add_argument('--fisher_epsilon', type=float, default=0, help='clip value for fisher regularization')
parser.add_argument('--train_image_size', type=int, default=480, help='spatial size of inputs')
parser.add_argument('--bn_frozen', type=int, default=0, help='freezing the statistics in existing bn layers')

parser.add_argument('--test_max_iter', type=int, default=None, help='maximum test iteration')
parser.add_argument('--test_batch_size', type=int, default=1, help='batch size used for test or validation')
parser.add_argument('--test_image_size', type=int, default=480,
                    help='spatial size of inputs for test. not used any longer')
parser.add_argument('--mirror', type=int, default=1, help='whether adding the results from mirroring.')
FLAGS = parser.parse_args()

num_classes=2
IMG_MEAN = np.array((103.939, 116.779, 123.68), dtype=np.float32)

def train(resume_step=None):
    global_step = tf.get_variable('global_step', [], dtype=tf.int64,
                                  initializer=tf.constant_initializer(0), trainable=False)
    image_size = FLAGS.train_image_size

    print '================',
    if FLAGS.data_type == 16:
        print 'using tf.float16 ====================='
        data_type = tf.float16
        print 'can not use float16 at this moment, because of tf.nn.bn, if using fused_bn, the learning will be nan',
        print ', no idea what happened.'
    else:
        print 'using tf.float32 ====================='
        data_type = tf.float32

    data_list = FLAGS.subsets_for_training.split(',')
    if len(data_list) < 1:
        data_list = ['train']
    print data_list

    images = []
    labels = []

    with tf.device('/cpu:0'):
        IMG_MEAN = np.array((103.939, 116.779, 123.68), dtype=np.float32)
        coord = tf.train.Coordinator()
        reader = ImageReader(
            './traindata/train',
            'train.txt',
            '480,480', 'true', 'true', 255, IMG_MEAN, coord)

    print '================ Database Info ================'
    for i in range(FLAGS.gpu_num):
        with tf.device('/cpu:0'):
            image_batch, label_batch = reader.dequeue(FLAGS.batch_size)
            images.append(image_batch)
            labels.append(label_batch)

    wd_rate_ph = tf.placeholder(data_type, shape=())
    wd_rate2_ph = tf.placeholder(data_type, shape=())
    lrn_rate_ph = tf.placeholder(data_type, shape=())

    new_layer_names = FLAGS.new_layer_names
    if FLAGS.new_layer_names is not None:
        new_layer_names = new_layer_names.split(',')
    assert 'pspnet' in FLAGS.network

    resnet = 'resnet_v1_50'
    PSPModel = pspnet_mg.PSPNetMG

    with tf.variable_scope(resnet):
        model = PSPModel(num_classes, lrn_rate_ph, wd_rate_ph, wd_rate2_ph,
                         mode='train', bn_epsilon=FLAGS.epsilon, resnet=resnet,
                         norm_only=FLAGS.norm_only,
                         initializer=FLAGS.initializer,
                         fix_blocks=FLAGS.fix_blocks,
                         fine_tune_filename=FLAGS.fine_tune_filename,
                         bn_ema=FLAGS.ema_decay,
                         bn_frozen=FLAGS.bn_frozen,
                         wd_mode=FLAGS.weight_decay_mode,
                         fisher_filename=FLAGS.fisher_filename,
                         gpu_num=FLAGS.gpu_num,
                         float_type=data_type,
                         fisher_epsilon=FLAGS.fisher_epsilon,
                         has_aux_loss=FLAGS.has_aux_loss,
                         train_like_in_paper=FLAGS.train_like_in_paper,
                         structure_in_paper=FLAGS.structure_in_paper,
                         new_layer_names=new_layer_names,
                         loss_type=FLAGS.loss_type,
                         train_conv2dt=FLAGS.train_conv2dt,
                         resize_images_method=FLAGS.resize_images_method)
        model.inference(images)
        model.build_train_op(labels)

    names = []
    num_params = 0
    for v in tf.trainable_variables():
        # print v.name
        names.append(v.name)
        num = 1
        for i in v.get_shape().as_list():
            num *= i
        num_params += num
    print "Trainable parameters' num: %d" % num_params

    print 'iou precision shape: ', model.predictions.get_shape(), labels[0].get_shape()
    pred = tf.reshape(model.predictions, [-1, ])
    gt = tf.reshape(labels[0], [-1, ])
    indices = tf.squeeze(tf.where(tf.less_equal(gt, num_classes - 1)), 1)
    gt = tf.cast(tf.gather(gt, indices), tf.int32)
    pred = tf.gather(pred, indices)
    precision_op, update_op = tf.contrib.metrics.streaming_mean_iou(pred, gt, num_classes=num_classes)
    # ========================= end of building model ================================

    step = 0
    logdir = LogDir(FLAGS.database, FLAGS.log_dir, FLAGS.weight_decay_mode)
    logdir.print_all_info()
    if not os.path.exists(logdir.log_dir):
        print 'creating ', logdir.log_dir, '...'
        os.mkdir(logdir.log_dir)
    if not os.path.exists(logdir.database_dir):
        print 'creating ', logdir.database_dir, '...'
        os.mkdir(logdir.database_dir)
    if not os.path.exists(logdir.exp_dir):
        print 'creating ', logdir.exp_dir, '...'
        os.mkdir(logdir.exp_dir)
    if not os.path.exists(logdir.snapshot_dir):
        print 'creating ', logdir.snapshot_dir, '...'
        os.mkdir(logdir.snapshot_dir)

    init = [tf.global_variables_initializer(), tf.local_variables_initializer()]

    gpu_options = tf.GPUOptions(allow_growth=False)
    config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options, allow_soft_placement=True)
    sess = tf.Session(config=config)
    sess.run(init)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    if '.npy' in FLAGS.fine_tune_filename:
        # This can transform .npy weights with variables names being the same to the tf ckpt model.
        fine_tune_variables = []
        npy_dict = np.load(FLAGS.fine_tune_filename).item()
        new_layers_names = ['Momentum']
        for v in tf.global_variables():
            print '=====Saving initial snapshot process:',
            if any(elem in v.name for elem in new_layers_names):
                print 'not import', v.name
                continue

            name = v.name.split(':0')[0]
            if name not in npy_dict:
                print 'not find', v.name
                continue

            v.load(npy_dict[name], sess)
            print 'saving', v.name
            fine_tune_variables.append(v)

        saver = tf.train.Saver(var_list=fine_tune_variables)
        saver.save(sess, logdir.snapshot_dir + '/model.ckpt', global_step=0)

        return

    import_variables = tf.trainable_variables()
    if FLAGS.fix_blocks > 0 or FLAGS.bn_frozen > 0:
        import_variables = tf.global_variables()

    if FLAGS.fine_tune_filename is not None and resume_step is None:
        fine_tune_variables = []
        new_layers_names = model.new_layers_names
        new_layers_names.append('Momentum')
        new_layers_names.append('up_sample')
        for v in import_variables:
            if any(elem in v.name for elem in new_layers_names):
                print '=====Finetuning Process: not import %s' % v.name
                continue
            fine_tune_variables.append(v)
##########载入预训练参数##################
        loader = tf.train.Saver(var_list=fine_tune_variables, allow_empty=True)
        loader.restore(sess, FLAGS.fine_tune_filename)
        print('=====Succesfully loaded fine-tune model from %s.' % FLAGS.fine_tune_filename)
    elif resume_step is not None:
        # ./snapshot/model.ckpt-3000
        i_ckpt = logdir.snapshot_dir + '/model.ckpt-%d' % resume_step

        loader = tf.train.Saver(max_to_keep=0)
        loader.restore(sess, i_ckpt)

        step = resume_step
        print('=====Succesfully loaded model from %s at step=%s.' % (i_ckpt, resume_step))
    else:
        print '=====Not import any model.'

    print '=========================== training process begins ================================='
    f_log = open(logdir.exp_dir + '/' + str(datetime.datetime.now()) + '.txt', 'w')
    f_log.write('step,loss,precision,wd\n')
    f_log.write(sorted_str_dict(FLAGS.__dict__) + '\n')

    average_loss = 0.0
    show_period = 20
    snapshot = FLAGS.snapshot
    max_iter = FLAGS.train_max_iter
    lrn_rate = FLAGS.lrn_rate

    lr_step = []
    if FLAGS.lr_step is not None:
        temps = FLAGS.lr_step.split(',')
        for t in temps:
            lr_step.append(int(t))

    # fine_tune_variables = []
    # for v in tf.global_variables():
    #     if 'Momentum' in v.name:
    #         continue
    #     print '=====Saving initial snapshot process: saving %s' % v.name
    #     fine_tune_variables.append(v)
    #
    # saver = tf.train.Saver(var_list=fine_tune_variables)
    # saver.save(sess, logdir.snapshot_dir + '/model.ckpt', global_step=0)

    saver = tf.train.Saver(max_to_keep=2)
    t0 = None
    wd_rate = FLAGS.weight_decay_rate
    wd_rate2 = FLAGS.weight_decay_rate2

    if FLAGS.save_first_iteration == 1:
        saver.save(sess, logdir.snapshot_dir + '/model.ckpt', global_step=step)

    has_nan = False
    while step < max_iter + 1:
        if FLAGS.poly_lr == 1:
            lrn_rate = ((1-1.0*step/max_iter)**0.9) * FLAGS.lrn_rate

        step += 1
        if len(lr_step) > 0 and step == lr_step[0]:
            lrn_rate *= FLAGS.step_size
            lr_step.remove(step)

        _, loss, wd, update, precision = sess.run([
            model.train_op, model.loss, model.wd, update_op, precision_op
        ],
            feed_dict={
                lrn_rate_ph: lrn_rate,
                wd_rate_ph: wd_rate,
                wd_rate2_ph: wd_rate2
            }
        )

        if math.isnan(loss) or math.isnan(wd):
            print 'loss or weight norm is nan. Training Stopped!'
            has_nan = True
            break

        average_loss += loss

        if step % snapshot == 0:
            saver.save(sess, logdir.snapshot_dir + '/model.ckpt', global_step=step)
            sess.run([tf.local_variables_initializer()])

        if step % show_period == 0:
            left_hours = 0

            if t0 is not None:
                delta_t = (datetime.datetime.now() - t0).seconds
                left_time = (max_iter - step) / show_period * delta_t
                left_hours = left_time/3600.0

            t0 = datetime.datetime.now()

            average_loss /= show_period

            if step == 0:
                average_loss *= show_period

            f_log.write('%d,%f,%f,%f\n' % (step, average_loss, precision, wd))
            f_log.flush()

            print '%s %s] Step %s, lr = %f, wd_rate = %f, wd_rate_2 = %f ' \
                  % (str(datetime.datetime.now()), str(os.getpid()), step, lrn_rate, wd_rate, wd_rate2)
            print '\t loss = %.4f, precision = %.4f, wd = %.4f' % (average_loss, precision, wd)
            print '\t estimated time left: %.1f hours. %d/%d' % (left_hours, step, max_iter)

            average_loss = 0.0

    coord.request_stop()
    coord.join(threads)

    return f_log, logdir, has_nan  # f_log and logdir returned for eval.


def eval(i_ckpt):
    # does not perform multi-scale test. ms-test is in predict.py
    tf.reset_default_graph()

    print '================',
    if FLAGS.data_type == 16:
        print 'using tf.float16 ====================='
        data_type = tf.float16
    else:
        print 'using tf.float32 ====================='
        data_type = tf.float32

    if 'pspnet' in FLAGS.network:
        input_size = FLAGS.test_image_size
        print '=====because using pspnet, the inputs have a fixed size and should be divided by 48:', input_size
        assert FLAGS.test_image_size % 48 == 0
    else:
        input_size = None
        return

    with tf.device('/cpu:0'):

        coord = tf.train.Coordinator()
        reader = ImageReader(
            './traindata/test',
            'val.txt',
            '480,480', 'False', 'False', 255, IMG_MEAN, coord)

    images_pl = [tf.placeholder(tf.float32, [None, input_size, input_size, 3])]
    labels_pl = [tf.placeholder(tf.int32, [None, input_size, input_size, 1])]

    resnet = 'resnet_v1_50'
    PSPModel = pspnet_mg.PSPNetMG

    with tf.variable_scope(resnet):
        model = PSPModel(num_classes, None, None, None,
                         mode='val', bn_epsilon=FLAGS.epsilon, resnet=resnet,
                         norm_only=FLAGS.norm_only,
                         float_type=data_type,
                         has_aux_loss=False,
                         structure_in_paper=FLAGS.structure_in_paper,
                         resize_images_method=FLAGS.resize_images_method,
                         train_conv2dt=FLAGS.train_conv2dt
                         )
        logits = model.inference(images_pl)
        model.compute_loss(labels_pl, logits)
    # ========================= end of building model ================================

    gpu_options = tf.GPUOptions(allow_growth=False)
    config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options, allow_soft_placement=True)
    sess = tf.Session(config=config)
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    if i_ckpt is not None:
        loader = tf.train.Saver(max_to_keep=0)
        loader.restore(sess, i_ckpt)
        eval_step = i_ckpt.split('-')[-1]
        print('Succesfully loaded model from %s at step=%s.' % (i_ckpt, eval_step))

    print '======================= eval process begins ========================='
    average_loss = 0.0
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    images_filenames = reader.image_list
    labels_filenames = reader.label_list
    img_mean = IMG_MEAN

    if FLAGS.test_max_iter is None:
        max_iter = len(images_filenames)
    else:
        max_iter = FLAGS.test_max_iter

    step = 0
    while step < max_iter:
        image, label = cv2.imread(images_filenames[step], 1), cv2.imread(labels_filenames[step], 0)
        label = np.reshape(label, [1, label.shape[0], label.shape[1], 1])

        imgsplitter = ImageSplitter(image, 1.0, FLAGS.color_switch, input_size, img_mean)
        feed_dict = {images_pl[0]: imgsplitter.get_split_crops()}
        [logits] = sess.run([
            model.probabilities
        ],
            feed_dict=feed_dict
        )
        total_logits = imgsplitter.reassemble_crops(logits)
        if FLAGS.mirror == 1:
            image_mirror = image[:, ::-1]
            imgsplitter_mirror = ImageSplitter(image_mirror, 1.0, FLAGS.color_switch, input_size, img_mean)
            feed_dict = {images_pl[0]: imgsplitter_mirror.get_split_crops()}
            [logits_m] = sess.run([
                model.probabilities
            ],
                feed_dict=feed_dict
            )
            logits_m = imgsplitter_mirror.reassemble_crops(logits_m)
            total_logits += logits_m[:, ::-1]

        prediction = np.argmax(total_logits, axis=-1)
        step += 1
        compute_confusion_matrix(label, prediction, confusion_matrix)
        if step % 200 == 0:
            print '%s %s] %d / %d. iou updating' \
                  % (str(datetime.datetime.now()), str(os.getpid()), step, max_iter)
            compute_iou(confusion_matrix)
            #print 'imprecise loss', average_loss / step

    precision = compute_iou(confusion_matrix)
    coord.request_stop()
    coord.join(threads)

    return precision


def main(_):
    # ============================================================================
    # ============================= TRAIN ========================================
    # ============================================================================
    print(sorted_str_dict(FLAGS.__dict__))
    if FLAGS.resume_step is not None:
        print 'Ready to resume from step %d.' % FLAGS.resume_step

    assert FLAGS.gpu_num is not None, 'should specify the number of gpu.'
    assert FLAGS.gpu_num > 0, 'the number of gpu should be bigger than 0.'
    if FLAGS.eval_only:
        logdir = LogDir(FLAGS.database, FLAGS.log_dir, FLAGS.weight_decay_mode)
        logdir.print_all_info()
        f_log = open(logdir.exp_dir + '/' + str(datetime.datetime.now()) + '.txt', 'w')
        f_log.write('step,loss,precision,wd\n')
        f_log.write(sorted_str_dict(FLAGS.__dict__) + '\n')
    else:
        f_log, logdir, has_nan = train(FLAGS.resume_step)

        if has_nan:
            f_log.write('TEST:0,nan,nan\n')
            f_log.flush()
            return

    # ============================================================================
    # ============================= EVAL =========================================
    # ============================================================================
    f_log.write('TEST:step,loss,precision\n')

    import glob
    i_ckpts = sorted(glob.glob(logdir.snapshot_dir + '/model.ckpt-*.index'), key=os.path.getmtime)

    # ============================================================================
    # ======================== Eval for the last model ===========================
    # ============================================================================
    i_ckpt = i_ckpts[-1].split('.index')[0]
    precision = eval(i_ckpt)
    step = i_ckpt.split('-')[-1]
    print '%s %s] Step %s Test' % (str(datetime.datetime.now()), str(os.getpid()), step)
    print '\t precision = %.4f' % (precision)
    f_log.write('TEST:%s,%f\n' % (step, precision))
    f_log.flush()

    f_log.close()


if __name__ == '__main__':
    tf.app.run()
