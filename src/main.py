from vgg16_model import vgg16

import tensorflow as tf
import numpy as np

import os
from glob import glob1
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import time


        
def getFileList(datapath):
    file_names_csv = glob1(datapath ,"*.csv")
    file_list_csv = [os.path.join(datapath, fname) for fname in file_names_csv]
    file_list_png = [os.path.join(datapath, fname[:-3] + 'png') for fname in file_names_csv] #remove extension
    return (file_list_png, file_list_csv)

def read_files(path, filename_queue):
    textReader = tf.TextLineReader()
    _, csv_content = textReader.read(filename_queue)    
    record_defaults = [tf.constant([], dtype=tf.string)] + [tf.constant([], dtype=tf.float32)]*3006
    all_data = tf.decode_csv(csv_content, record_defaults=record_defaults, field_delim=",")


    im_name = tf.string_join([path, all_data[0]],"/")
#    im_name = all_data[0]
    coords = tf.pack(all_data[1:])
    
    im_cont = tf.read_file(im_name)
    example = tf.image.decode_png(im_cont, channels=3)
    return example, coords

def input_pipeline(path, filenames, batch_size, num_epochs=None, read_threads=1):
    filenames_tensor = ops.convert_to_tensor(filenames, dtype=dtypes.string)
    filename_queue = tf.train.string_input_producer(filenames_tensor,num_epochs=num_epochs, shuffle=False)

    
    min_after_dequeue = 1000
    capacity = min_after_dequeue + 3 * batch_size
    
    data_list=[(im, pts) for im, pts in [read_files(path,filename_queue) for _ in range(read_threads)]]
    
    [(im.set_shape([224, 224, 3]), pt.set_shape((3006,))) for (im,pt) in data_list]
    
    image_batch, points_batch = tf.train.shuffle_batch_join( data_list, batch_size=batch_size, capacity=capacity,
            min_after_dequeue=min_after_dequeue)   

    return image_batch, points_batch


    
def isometric_loss(pred, edges, dist):
    pred = tf.reshape(pred, (1002, 3))
    
    rand_indices = np.random.choice(range(len(dist)), size = 1000)
    
    s = 0
    for i in rand_indices:
        pred_dist = tf.sqrt(tf.reduce_sum(tf.square(pred[edges[i,0],:] - pred[edges[i, 1],:])))
        s += tf.abs(pred_dist - dist[i])
    return s

def saveWeights(vgg, retrained_layers_list, fname, print_message = False):
    #Saving the weights            
    weights = {}
    for l in retrained_layers_list:
        exec("weights['fc{}_W'] = vgg.fc{}w.eval()".format(l, l))
        exec("weights['fc{}_b'] = vgg.fc{}b.eval()".format(l, l))
    full_name = 'weights/weights_fc_' + fname
    np.savez(full_name, **weights)
    if print_message:
        print('weights saved to {}.npz.'.format(full_name))

def runTest(data, cost, print_step=10):
    images = data[0]
    points = data[1]
    losses = []
    
#    pred_arr = np.empty((10, 3007))
#    gt_arr = np.empty((10, 3006))
    
    coord2 = tf.train.Coordinator()
    threads2 = tf.train.start_queue_runners(coord=coord2, sess=sess)
    
    n_saved = 20
    
    data = {}
    data['pred'] = np.empty((n_saved, 3007))
    data['gt'] = np.empty((n_saved, 3006))
    data['image'] = np.empty((n_saved, 224, 224, 3))
    
    try:
        print("Testing..")
        step = 0
        while not coord2.should_stop():
            
            
            start_time = time.time()
            image_test, points_test = sess.run([images, points])
            test_loss = cost.eval(feed_dict={x: image_test, y:points_test})
            losses.append(test_loss)
            
            duration = time.time() - start_time
            if step % print_step == 0:
                print('Step %d: loss = %.2f (%.3f sec)' % (step, test_loss, duration))
            
            
            if step < n_saved:
                
                data['gt'][step,:] = points_test
                data['pred'][step,1:] = vgg.pred.eval(feed_dict={x: image_test, y:points_test})
                data['pred'][step, 0] = test_loss
                data['image'][step] = image_test
            
            step += 1
            
        
    except tf.errors.OutOfRangeError:
        pass
        #print('Done testing -- epoch limit reached')
        #            except Exception, e:
        #                print repr(e)
        #except KeyError, e:
        #   print('I got a KeyError - reason "%s"' % str(e))
    finally:
        coord2.request_stop()
        coord2.join(threads2)
    mean_loss = np.mean(losses)
    print("Mean testing loss: {}".format(mean_loss))
    
#    np.savetxt('results/test'+fname+'.csv', data)
    np.savez('results/test'+fname+'.npz', **data)
    print("Results saved to {}.".format('results/test'+fname+'.npz'))
    
    return mean_loss
    #gt = coords_test
    #pred = vgg.pred.eval(feed_dict={x: example_test, y:coords_test})
    #np.savetxt(fname+'pred_test.csv', pred)
    #np.savetxt(fname+'gt_test.csv', gt)
            
if __name__ == '__main__':
    
    datapath = "../datasets/dataset_rt+fl"

    #params

    #Base learning rate
    lr = 0.0002
    reg_constant = 0.03
    
    
#    train_it = 100
    num_epochs = 1
    batch_size = 10
    
    
    train = True
    test = False
    
    retrained_layers = range(1,4)
    
        
    with tf.Graph().as_default():
      #  with tf.device('/cpu:0'):

        _, filenames = getFileList(datapath)
        filenames = filenames[:100]
        
        # Divide train/test 
        train_size = int(round(len(filenames) * 0.9))
        filenames_train = filenames[:train_size]
        filenames_test = filenames[train_size:]

        images_batch_train, points_batch_train = input_pipeline(datapath, filenames_train, batch_size, num_epochs=num_epochs, read_threads=1)
        images_batch_test, points_batch_test = input_pipeline(datapath, filenames_test, batch_size=1, num_epochs=1, read_threads=1)


        x = tf.placeholder(tf.float32, [None, 224, 224, 3])
        y = tf.placeholder(tf.float32, [None, 3006])
        
        
        
        writer = tf.summary.FileWriter("logs/", graph=tf.get_default_graph())

        vgg = vgg16(x)

        #Regularization term
        vars   = tf.trainable_variables()
        lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name ])

        #isometric_loss(vgg.pred, edges, dist)
        cost_train = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(y, vgg.pred)))) + lossL2 * reg_constant
        cost_test = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(y, vgg.pred))))

        batch = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(
          lr,                # Base learning rate.
          batch * batch_size,  # Current index into the dataset.
          train_size,          # Decay step.
          0.95,                # Decay rate.
          staircase=True)
        
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost_train, global_step=batch)
        

        #Summary for cost
        tf.summary.scalar("cost", cost_test)
        
        summary_op = tf.merge_all_summaries()
#        summary_op = tf.contrib.deprecated.merge_all_summaries()
        
        fname = str(time.time())#timestamp used for saved file names



        #config = tf.ConfigProto()
        #config.gpu_options.allocator_type = 'BFC'
        #with tf.Session(config = config) as sess:

        #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        #with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())


            vgg.load_weights('weights/vgg16_weights.npz', sess)
            vgg.load_retrained_weights('weights/weights_fc_1489763677.84.npz',sess)
#            vgg.load_retrained_weights('weights/weights_trained_on_dec_norm_v2.npz',sess)
            
#            vgg.load_retrained_weights('weights/weights_fc_1489169964.99.npz', sess)            

#            vgg.load_retrained_weights('weights/weights_fc_1489495674.53.npz',sess)

            ## Traininng ######################################################
           
            if train:
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord, sess=sess)
        #            sess.graph.finalize()
    
                try:
                    step = 0
                    train_start_time = time.time()
                    epoch_step = int(train_size / batch_size)
                    current_epoch = 1
                    epoch_start_time = time.time()
                    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.75)
                    #config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False, allow_soft_placement=False)
                    #with tf.device('/gpu:0',):
                    while not coord.should_stop():
    
                        start_time = time.time()
    
                        example, coords = sess.run([images_batch_train, points_batch_train])                                        
                        duration = time.time() - start_time
    
                        # Write the summaries and print an overview fairly often.
                        if step % 10 == 0:
    #                        # Update the events file.
                            _, loss, summary_str = sess.run([optimizer, cost_test, summary_op], feed_dict={x: example, y:coords})
#                            _, loss = sess.run([optimizer, cost_test], feed_dict={x: example, y:coords})
                            # Print status to stdout.
                            print('Step %d: loss = %.2f (%f sec)' % (step, loss, duration))
                            writer.add_summary(summary_str, step)
                        else:
                            sess.run(optimizer, feed_dict={x: example, y:coords})
                        step += 1
    
                        #Test if epoch ended
                        if (step % epoch_step == 0):
                            with tf.device('/cpu:0'):
                                print("Time elapsed: {} min".format((time.time() - epoch_start_time)/60.0))
                                print("----epoch {} ---------------".format(current_epoch))
                                saveWeights(vgg, retrained_layers, fname, True)
                                current_epoch += 1
                                epoch_start_time = time.time()

                            
                except tf.errors.OutOfRangeError:
                    print('Done training -- epoch limit reached\n Training time: {} min'.format((time.time()-train_start_time)/60))
                finally:
                    coord.request_stop()
                    coord.join(threads)
                    saveWeights(vgg, retrained_layers, fname, True)
                    writer.close()
                    # Testing
            if test:
                runTest(data=[images_batch_test, points_batch_test],cost=cost_test, print_step=10)

   

    