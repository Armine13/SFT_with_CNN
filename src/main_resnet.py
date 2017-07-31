# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""ResNet Train/Eval module.
"""
import time
import six
import sys

#import cifar_input
import numpy as np
import resnet_model
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('dataset', 'cifar10', 'cifar10 or cifar100.')
tf.app.flags.DEFINE_string('mode', 'train', 'train or eval.')
tf.app.flags.DEFINE_string('train_data_path', '/home/isit/armine/Dropbox/tube_test/train',
                           'Filepattern for training data.')
tf.app.flags.DEFINE_string('eval_data_path', '/home/isit/armine/Dropbox/tube_test/test',
                           'Filepattern for eval data')
tf.app.flags.DEFINE_integer('image_size', 224, 'Image side length.')
tf.app.flags.DEFINE_string('train_dir', 'checkpoints/train',
                           'Directory to keep training outputs.')
tf.app.flags.DEFINE_string('eval_dir', 'checkpoints/eval',
                           'Directory to keep eval outputs.')
tf.app.flags.DEFINE_integer('eval_batch_count', 4,
                            'Number of batches to eval.')
tf.app.flags.DEFINE_bool('eval_once', False,
                         'Whether evaluate the model only once.')
tf.app.flags.DEFINE_string('log_root', 'checkpoints',
                           'Directory to keep the checkpoints. Should be a '
                           'parent directory of FLAGS.train_dir/eval_dir.')
tf.app.flags.DEFINE_integer('num_gpus', 0,
                            'Number of gpus used for training. (0 or 1)')



def read_files(path, filename_queue):
    textReader = tf.TextLineReader()
    _, csv_content = textReader.read(filename_queue)
    record_defaults = [tf.constant([], dtype=tf.string)] + [tf.constant([], dtype=tf.float32)]*3008
    all_data = tf.decode_csv(csv_content, record_defaults=record_defaults, field_delim=",")


    im_name = tf.string_join([path, all_data[0]],"/")
#    im_name = all_data[0]
    fl = all_data[1]
    presence = all_data[2]
    coords = tf.pack(all_data[3:])

#    coords = tf.divide(coords, 227)
    
    im_cont = tf.read_file(im_name)
    example = tf.image.decode_png(im_cont, channels=3)
    return example, fl, presence, coords

def input_pipeline(path, filenames, batch_size, num_epochs=1, read_threads=1, shuffle=True):
    filenames_tensor = ops.convert_to_tensor(filenames, dtype=dtypes.string)
    filename_queue = tf.train.string_input_producer(filenames_tensor,num_epochs=num_epochs, shuffle=shuffle)
#    filename_queue = tf.train.string_input_producer(filenames_tensor, shuffle=shuffle)


    min_after_dequeue = 1000
    capacity = min_after_dequeue + 3 * batch_size

    data_list=[(im, fl, presence, pts) for im, fl, presence, pts in [read_files(path,filename_queue) for _ in range(read_threads)]]

    [(im.set_shape([224, 224, 3]), fl.set_shape(None), presence.set_shape(None), pt.set_shape((3006,))) for (im, fl, presence, pt) in data_list]
#    data_list = [(preprocessImage(im), fl, presence, pt) for (im, fl, presence, pt) in data_list]
    image_batch, fl_batch, presence_batch, points_batch = tf.train.shuffle_batch_join( data_list, batch_size=batch_size, capacity=capacity,
            min_after_dequeue=min_after_dequeue)
    return  tf.to_float(image_batch, name='ToFloat'), fl_batch, presence_batch, points_batch

def getFileList(datapath):
    from glob import glob1
    from os import path
    file_names_csv = glob1(datapath ,"*.csv")
    file_list_csv = [path.join(datapath, fname) for fname in file_names_csv]
    file_list_png = [path.join(datapath, fname[:-3] + 'png') for fname in file_names_csv] #remove extension
    return (file_list_png, file_list_csv)

def build_input(datapath, batch_size, mode):

#    trainpath = "/home/isit/armine/Dropbox/tube/train"
#    testpath = "/home/isit/armine/Dropbox/tube/test"
    
    
    _, filenames = getFileList(datapath)
#    filenames = filenames[:20]
#        train_size = len(filenames_train)
    if mode =='train':
        images_batch, fl_batch, presence_batch, points_batch = input_pipeline(datapath, filenames, batch_size=batch_size, num_epochs=100000, read_threads=20,shuffle=True)
    else:
        images_batch, fl_batch, presence_batch, points_batch = input_pipeline(datapath, filenames, batch_size=batch_size, num_epochs=1, read_threads=20,shuffle=False)
    return images_batch, fl_batch, presence_batch, points_batch

def optimistic_restore(session, save_file):
	reader = tf.train.NewCheckpointReader(save_file)
	saved_shapes = reader.get_variable_to_shape_map()
	var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
			if var.name.split(':')[0] in saved_shapes])
	restore_vars = []
        name2var = dict(zip(map(lambda x:x.name.split(':')[0], tf.global_variables()), tf.global_variables()))
	with tf.variable_scope('', reuse=True):
		for var_name, saved_var_name in var_names:
			curr_var = name2var[saved_var_name]
			var_shape = curr_var.get_shape().as_list()
			if var_shape == saved_shapes[saved_var_name]:
				restore_vars.append(curr_var)
	saver = tf.train.Saver(restore_vars)
	saver.restore(session, save_file)
    
def train(hps):
  """Training loop."""
  with tf.device('/cpu:0'):
      images, fl, presence, labels = build_input(FLAGS.train_data_path, hps.batch_size, FLAGS.mode)
      
  model = resnet_model.ResNet(hps, images, labels, fl, presence, FLAGS.mode)#, labels, fl, presence, mode):
  model.build_graph()

  param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
      tf.get_default_graph(),
      tfprof_options=tf.contrib.tfprof.model_analyzer.
          TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
  sys.stdout.write('total_params: %d\n' % param_stats.total_parameters)

  tf.contrib.tfprof.model_analyzer.print_model_analysis(
      tf.get_default_graph(),
      tfprof_options=tf.contrib.tfprof.model_analyzer.FLOAT_OPS_OPTIONS)

#  print(model.rmse.get_shape())
#  print(presence.get_shape())
  e = model.rmse*presence
#  print(e.get_shape())
  rmse = tf.divide(tf.reduce_sum(e), tf.cast(tf.count_nonzero(presence),tf.float32))# tf.reduce_mean(tf.sqrt(tf.reduce_mean(tf.square(tf.sub(model.pred, model.labels)))))
  fle = tf.reduce_mean(model.fl_re) #tf.reduce_mean(tf.divide(tf.abs(fl - model.pred_fl), fl))
  cost = model.cost
#  truth = tf.argmax(model.labels, axis=1)
#  predictions = tf.argmax(model.predictions, axis=1)
#  precision = tf.reduce_mean(tf.to_float(tf.equal(predictions, truth)))
  
#  with tf.Session() as sess:
#  model.loadWeights('/home/isit/armine/SFT_with_CNN/src/weights/resnet.npz')
  
  summary_hook = tf.train.SummarySaverHook(
      save_steps=100,
      output_dir=FLAGS.train_dir,
      summary_op=tf.summary.merge([model.summaries,
                                   tf.summary.scalar('loss', cost), tf.summary.scalar('3D_error', rmse), tf.summary.scalar('fl_error', fle)]))

  logging_hook = tf.train.LoggingTensorHook(
      tensors={'step': model.global_step,
               'loss': cost,
               'rmse': rmse,
               'fl_e': fle,
               'acc' : model.accuracy},
      every_n_iter=100)

  class _LearningRateSetterHook(tf.train.SessionRunHook):
    """Sets learning_rate based on global step."""

    def begin(self):
      self._lrn_rate = 0.1

    def before_run(self, run_context):
      #-------------------------------------------------
      return tf.train.SessionRunArgs(
          model.global_step,  # Asks for global step value.
          feed_dict={model.lrn_rate: self._lrn_rate})  # Sets learning rate

    def after_run(self, run_context, run_values):
      train_step = run_values.results
      if train_step < 40000:
        self._lrn_rate = 0.1
      elif train_step < 60000:
        self._lrn_rate = 0.01
      elif train_step < 80000:
        self._lrn_rate = 0.001
      else:
        self._lrn_rate = 0.0001
#        self._lrn_rate = 0.0001
#  with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
#      optimistic_restore(sess, '/home/isit/armine/SFT_with_CNN/src/checkpoints1/model.ckpt-504958')
      
  with tf.train.MonitoredTrainingSession(
      checkpoint_dir=FLAGS.log_root,
      hooks=[logging_hook, _LearningRateSetterHook()],
      chief_only_hooks=[summary_hook],
      # Since we provide a SummarySaverHook, we need to disable default
      # SummarySaverHook. To do that we set save_summaries_steps to 0.
      save_summaries_steps=0,
      config=tf.ConfigProto(allow_soft_placement=True)) as mon_sess: 
#    model.saveWeights(mon_sess, '/home/isit/armine/SFT_with_CNN/src/weights/resnet2.npz')
    
    while not mon_sess.should_stop():
      mon_sess.run(model.train_op)


def evaluate(hps):
  """Eval loop."""
  with tf.device('/cpu:0'):
      images, fl, presence, labels = build_input(FLAGS.eval_data_path, hps.batch_size, FLAGS.mode)
      
  model = resnet_model.ResNet(hps, images, labels, fl, presence, FLAGS.mode)#, labels, fl, presence, mode):
  
#  images, labels = cifar_input.build_input(
#      FLAGS.dataset, FLAGS.eval_data_path, hps.batch_size, FLAGS.mode)
#  model = resnet_model.ResNet(hps, images, labels, FLAGS.mode)
  model.build_graph()
  saver = tf.train.Saver()
  summary_writer = tf.summary.FileWriter(FLAGS.eval_dir)

  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
  sess.run(tf.local_variables_initializer())
  tf.train.start_queue_runners(sess)

  best_precision = 0.0
  while True:
    try:
      ckpt_state = tf.train.get_checkpoint_state(FLAGS.log_root)
    except tf.errors.OutOfRangeError as e:
      tf.logging.error('Cannot restore checkpoint: %s', e)
      continue
    if not (ckpt_state and ckpt_state.model_checkpoint_path):
      tf.logging.info('No model to eval yet at %s', FLAGS.log_root)
      continue
    tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
    saver.restore(sess, ckpt_state.model_checkpoint_path)

    
    total_prediction, correct_prediction = 0, 0
    for _ in six.moves.range(FLAGS.eval_batch_count):
      (summaries, loss, predictions, truth, train_step) = sess.run(
          [model.summaries, model.cost, model.predictions,
           model.labels, model.global_step])

      truth = np.argmax(truth, axis=1)
      predictions = np.argmax(predictions, axis=1)
      correct_prediction += np.sum(truth == predictions)
      total_prediction += predictions.shape[0]

    precision = 1.0 * correct_prediction / total_prediction
    best_precision = max(precision, best_precision)

    precision_summ = tf.Summary()
    precision_summ.value.add(
        tag='Precision', simple_value=precision)
    summary_writer.add_summary(precision_summ, train_step)
    best_precision_summ = tf.Summary()
    best_precision_summ.value.add(
        tag='Best Precision', simple_value=best_precision)
    summary_writer.add_summary(best_precision_summ, train_step)
    summary_writer.add_summary(summaries, train_step)
    tf.logging.info('loss: %.3f, precision: %.3f, best precision: %.3f' %
                    (loss, precision, best_precision))
    summary_writer.flush()

    if FLAGS.eval_once:
      break

    time.sleep(60)

def runTest(hps):
#def runTest(data, cost, cost_fl, accuracy, sess, print_step=10, saveData=False):
    
#    mean, std = runTest(data=[images_batch_test, fl_batch_test, presence_batch_test, points_batch_test],cost=cost_test,cost_fl=cost_test_fl, accuracy=accuracy, sess=sess, print_step=1, saveData=True)
    
    print_step = 10
    saveData = True

#    print(FLAGS.eval_data_path)
    _, filenames = getFileList(FLAGS.eval_data_path)
    n_files = len(filenames)
#    print(n_files)
    with tf.device('/cpu:0'):
      images_batch, fl_batch, presence_batch, points_batch = build_input(FLAGS.eval_data_path, hps.batch_size, FLAGS.mode)
#      print(images.get_shape())
#      print(fl.get_shape())
#      print(presence.get_shape())
#      print(points.get_shape())
      
      
    x = tf.placeholder(tf.float32, [hps.batch_size, 224, 224, 3])
    y = tf.placeholder(tf.float32, [hps.batch_size, 3006])
    fl = tf.placeholder(tf.float32, shape=[hps.batch_size])
    presence = tf.placeholder(tf.float32, shape=[hps.batch_size])

    model = resnet_model.ResNet(hps, x, y, fl, presence, FLAGS.mode)#, labels, fl, presence, mode):
    model.build_graph()
    saver = tf.train.Saver()
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allocator_type = 'BFC'


    with tf.Session(config = config) as sess:
        sess.run(tf.group(tf.initialize_all_variables(), tf.initialize_local_variables()))
        
        try:
          ckpt_state = tf.train.get_checkpoint_state(FLAGS.log_root)
        except tf.errors.OutOfRangeError as e:
          tf.logging.error('Cannot restore checkpoint: %s', e)
          
        if not (ckpt_state and ckpt_state.model_checkpoint_path):
          tf.logging.info('No model to eval yet at %s', FLAGS.log_root)
          
        tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
        saver.restore(sess, ckpt_state.model_checkpoint_path)


        rmse = tf.reduce_mean(tf.sqrt(tf.reduce_mean(tf.square(tf.sub(model.pred, model.labels)))))
        cost_fl = tf.reduce_mean(tf.divide(tf.abs(fl - model.pred_fl), fl))
        
#        presence_one_hot = tf.one_hot(tf.cast(presence, tf.int32), 2)
#        correct = tf.cast(tf.equal(tf.cast(presence, tf.int64), tf.argmax(model.predictions, 1)), tf.float32)
        accuracy = model.accuracy#tf.reduce_mean(correct)
            
        
    #    images = data[0]
    #    efl = data[1]
    #    pres = data[2]
    #    points = data[3]
        losses = []
        losses_fl = []
        acc = []
        fname = str(time.time())
        coord2 = tf.train.Coordinator()
        threads2 = tf.train.start_queue_runners(coord=coord2, sess=sess)
        n_saved = 100
        data = {}
        data['pred'] = np.empty((n_saved, 3006))
        data['pred_fl'] = np.empty((n_saved))
        data['gt'] = np.empty((n_saved, 3006))
        data['gt_fl'] = np.empty((n_saved))
        data['presence'] = np.empty((n_saved))
        data['detected'] = np.empty((n_saved))
        
        try:
            print("Testing..")
            step = 0
#            while step < n_files:
            while not coord2.should_stop():
                
                start_time = time.time()
                #images_batch, fl_batch, presence_batch, points_batch
                image_test, fl_test, presence_test, points_test = sess.run([images_batch, fl_batch, presence_batch, points_batch])
    #            test_loss = cost.eval(feed_dict={x: image_test, y:points_test,keep_prob: 1.0})
                test_loss = model.pred.eval(feed_dict={x: image_test, fl: fl_test, presence: presence_test, y:points_test})
                test_fl_loss = model.fl_re.eval(feed_dict={x: image_test, fl: fl_test, presence: presence_test, y:points_test})
                
                test_loss, test_fl_loss, test_acc = sess.run([rmse, cost_fl, accuracy], feed_dict={x: image_test, fl: fl_test, presence: presence_test, y:points_test})
                
                if presence_test == 1:
                    losses.append(test_loss)
                losses_fl.append(test_fl_loss)
                acc.append(test_acc)
                duration = time.time() - start_time
                if print_step!=-1 and step % print_step == 0:
                    print('Step %d: loss = %.2f loss_fl = %.2f acc = %.2f (%.3f sec)' % (step, test_loss*presence_test, test_fl_loss, test_acc, duration))
                if saveData:
                    if step < n_saved:
                        data['gt'][step,:] = points_test
                        data['gt_fl'][step] = fl_test
                        data['presence'][step] = presence_test
        #                data['pred'][step,:] = model.pred.eval(feed_dict={x: image_test, y:points_test,keep_prob: 1.0})
        #                data['pred'][step,:] = model.pred.eval(feed_dict={x: image_test, fl:fl_test, y:points_test})
        #                data['pred_fl'][step] = model.pred_fl.eval(feed_dict={x: image_test, fl:fl_test, y:points_test})
        #                data['detected'] = model.detected.eval(feed_dict={x: image_test, fl:fl_test, y:points_test})
                        if presence_test == 1:
                            data['pred'][step,:], data['pred_fl'][step], det = sess.run([model.pred, model.pred_fl, model.predictions],feed_dict={x: image_test, fl:fl_test,  presence: presence_test, y:points_test})
                            data['detected'][step] = np.argmax(det)
                        else:
                            data['pred'][step,:] = 0
                            data['pred_fl'][step], data['detected'][step] = sess.run([model.pred_fl, model.predictions],feed_dict={x: image_test, fl:fl_test, presence: presence_test, y:points_test})
#                             = np.argmax(det)
#                    else:
#                        break
                step += 1

        except tf.errors.OutOfRangeError:
            pass
        finally:
            coord2.request_stop()
            coord2.join(threads2)
        mean_loss = np.mean(losses)
        mean_loss_fl = np.mean(losses_fl)
        mean_acc = np.mean(acc)
    
    print("Mean testing loss: {} std={} min={} max={}".format(mean_loss,np.std(losses), min(losses), max(losses)))
    print("Mean testing FL loss: {} std={} min={} max={}".format(mean_loss_fl,np.std(losses_fl), min(losses_fl), max(losses_fl)))
    print("Mean accuracy: {}".format(mean_acc))

    if saveData:
        np.savez('results/test'+fname+'.npz', **data)
        print("Results saved to {}.".format('results/test'+fname+'.npz'))
    
    return mean_loss, np.std(losses)

def main(_):
#  if FLAGS.num_gpus == 0:
#    dev = '/cpu:0'
#  elif FLAGS.num_gpus == 1:
#    dev = '/gpu:0'
#  else:
#    raise ValueError('Only support 0 or 1 gpu.')


  if FLAGS.mode == 'train':
    batch_size = 25
  elif FLAGS.mode == 'eval':
    batch_size = 1

#  if FLAGS.dataset == 'cifar10':
#    num_classes = 10
#  elif FLAGS.dataset == 'cifar100':
#    num_classes = 100

  hps = resnet_model.HParams(batch_size=batch_size,
                             num_classes=3006,
                             min_lrn_rate=0.0001,
                             lrn_rate=0.01,
                             num_residual_units=5,
                             use_bottleneck=False,
                             weight_decay_rate=0.0002,
                             relu_leakiness=0.1,
                             optimizer='adam')

  with tf.device('gpu:0'):
    if FLAGS.mode == 'train':
      train(hps)
    elif FLAGS.mode == 'eval':
      runTest(hps)
      


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()