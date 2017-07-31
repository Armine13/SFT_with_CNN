from vgg16_model import vgg16

import tensorflow as tf
import numpy as np

import os
from glob import glob1
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
#from tensorflow import losses
import time
from matplotlib import pyplot as plt
from datetime import datetime


#from tensorflow.python.client import timeline


def getFileList(datapath):
    file_names_csv = glob1(datapath ,"*.csv")
    file_list_csv = [os.path.join(datapath, fname) for fname in file_names_csv]
    file_list_png = [os.path.join(datapath, fname[:-3] + 'png') for fname in file_names_csv] #remove extension
    return (file_list_png, file_list_csv)

def preprocessImage(im_tensor):
    mean = tf.constant([69, 68, 64], shape=[1, 1, 3], name='img_mean', dtype=tf.uint8)
    image = tf.cast(tf.cast(im_tensor,tf.int32) - tf.cast(mean,tf.int32), tf.float32)
#    image = tf.image.per_image_standardization(im_tensor)
    return image

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

def input_pipeline(path, filenames, batch_size, num_epochs=None, read_threads=1, shuffle=True):
    filenames_tensor = ops.convert_to_tensor(filenames, dtype=dtypes.string)
    filename_queue = tf.train.string_input_producer(filenames_tensor,num_epochs=num_epochs, shuffle=shuffle)


    min_after_dequeue = 1000
    capacity = min_after_dequeue + 3 * batch_size

    data_list=[(im, fl, presence, pts) for im, fl, presence, pts in [read_files(path,filename_queue) for _ in range(read_threads)]]

    [(im.set_shape([224, 224, 3]), fl.set_shape(None), presence.set_shape(None), pt.set_shape((3006,))) for (im, fl, presence, pt) in data_list]
    data_list = [(preprocessImage(im), fl, presence, pt) for (im, fl, presence, pt) in data_list]
    image_batch, fl_batch, presence_batch, points_batch = tf.train.shuffle_batch_join( data_list, batch_size=batch_size, capacity=capacity,
            min_after_dequeue=min_after_dequeue)

    return image_batch, fl_batch, presence_batch, points_batch


def isometric_loss(pred, edges, dist):
    pred = tf.reshape(pred, (1002, 3))

    rand_indices = np.random.choice(range(len(dist)), size = 1000)

    s = 0
    for i in rand_indices:
        pred_dist = tf.sqrt(tf.reduce_sum(tf.square(pred[edges[i,0],:] - pred[edges[i, 1],:])))
        s += tf.abs(pred_dist - dist[i])
    return s

def saveWeights(vgg, fname, print_message = False):
    full_name = 'weights/weights_' "latest"#+ fname
    model.save_weights(full_name)
    
    if print_message:
        print('weights saved to {}.npz.'.format(full_name))

def runTest(data, cost, cost_fl, accuracy, sess, print_step=10, saveData=False):
    images = data[0]
    efl = data[1]
    pres = data[2]
    points = data[3]
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
        while not coord2.should_stop():
            
            start_time = time.time()
            image_test, fl_test, presence_test, points_test = sess.run([images,efl,pres, points])
#            test_loss = cost.eval(feed_dict={x: image_test, y:points_test,keep_prob: 1.0})
            test_loss = cost.eval(feed_dict={x: image_test, fl: fl_test, presence: presence_test, y:points_test})
            test_fl_loss = cost_fl.eval(feed_dict={x: image_test, fl: fl_test, presence: presence_test, y:points_test})
            
            test_loss, test_fl_loss, test_acc = sess.run([cost, cost_fl, accuracy], feed_dict={x: image_test, fl: fl_test, presence: presence_test, y:points_test})
            
            if presence_test == 1:
                losses.append(test_loss)
            losses_fl.append(test_fl_loss)
            acc.append(test_acc)
            duration = time.time() - start_time
            if print_step!=-1 and step % print_step == 0:
                print('Step %d: loss = %.2f loss_fl = %.2f acc = %.2f (%.3f sec)' % (step, test_loss*presence_test, test_fl_loss, test_acc, duration))
            if saveData and step < n_saved:
                data['gt'][step,:] = points_test
                data['gt_fl'][step] = fl_test
                data['presence'][step] = presence_test
#                data['pred'][step,:] = model.pred.eval(feed_dict={x: image_test, y:points_test,keep_prob: 1.0})
#                data['pred'][step,:] = model.pred.eval(feed_dict={x: image_test, fl:fl_test, y:points_test})
#                data['pred_fl'][step] = model.pred_fl.eval(feed_dict={x: image_test, fl:fl_test, y:points_test})
#                data['detected'] = model.detected.eval(feed_dict={x: image_test, fl:fl_test, y:points_test})
                if presence_test == 1:
                    data['pred'][step,:], data['pred_fl'][step], det = sess.run([model.pred, model.pred_fl, model.detected],feed_dict={x: image_test, fl:fl_test,  presence: presence_test, y:points_test})
                    data['detected'][step] = np.argmax(det)
                else:
                    data['pred'][step,:] = 0
                    data['pred_fl'][step], det = sess.run([model.pred_fl, model.detected],feed_dict={x: image_test, fl:fl_test, presence: presence_test, y:points_test})
                    data['detected'][step] = np.argmax(det)
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
           
if __name__ == '__main__':

    trainpath = "/home/isit/armine/Dropbox/tube/train"
    testpath = "/home/isit/armine/Dropbox/tube/test"

    weights_path = 'weights/weights_latest.npz'
    test_weights_path = 'weights/weights_latest.npz'#weights_bg_best_1.56.npz'#weights_1492787682.16_best.npz'#weights_latest_main.npz'

    
    #params
    learning_rate = 0.00001 #0.000008
    #do_train = 1.0
#    lr_decay = 0.9
    reg_constant = 0#0.23

    read_threads = 30
    num_epochs = 10000000
    batch_size = 4

    fname = str(time.time())#timestamp used for saved file names
    summaries_dir = 'logs/' + fname
    train = True
#    train = False
#    test = True
    test = False
    
#

    if train:
        with tf.Graph().as_default():
            with tf.device('/cpu:0'):
                _, filenames_train = getFileList(trainpath)
                
                
#                filenames_train = filenames_train[:10]
                train_size = len(filenames_train)
                _, filenames_test = getFileList(testpath)

                images_batch_train, fl_batch_train, presence_batch_train, points_batch_train = input_pipeline(trainpath, filenames_train, batch_size, num_epochs=num_epochs, read_threads=read_threads,shuffle=True)
                images_batch_test, fl_batch_test, presence_batch_test, points_batch_test = input_pipeline(testpath, filenames_test, batch_size=batch_size, num_epochs=num_epochs, read_threads=read_threads,shuffle=True)
                
                images = tf.placeholder_with_default(images_batch_train, shape=[None, 224, 224, 3])
                points = tf.placeholder_with_default(points_batch_train, shape=[None, 3006])
                fl = tf.placeholder_with_default(fl_batch_train, shape=None)
                presence = tf.placeholder_with_default(presence_batch_train, shape=None)
                
#                images = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])
#                points = tf.placeholder(tf.float32, shape=[None, 3006])
#                fl = tf.placeholder(tf.float32, shape=None)
                
            model = vgg16(images)
            
#            
            model.add_histogram_summary()
            
            
            with tf.name_scope("costs"):
                def rmse(a, b):
                    return tf.sqrt(tf.reduce_mean(tf.square(tf.sub(a, b)),axis=1))
                def mse(a, b):
                    return tf.reduce_mean(tf.square(tf.sub(a, b)),axis=1)
                def big_error():
                    return tf.constant(10000, dtype=tf.float32, name='tenthousand')
                def abs_(a):
                    return tf.sqrt((a)**2+10**-8)
                
                def l1(a, b):
                    return tf.reduce_mean(tf.abs(a-b))
                #reg_losses = tf.nn.l2_loss(model.fc1w)+tf.nn.l2_loss(model.fc3w) #+tf.nn.l2_loss(model.fc2w) 
#                cost_train_reg = 0.5*rmse(points, model.rnn1)+0.5*rmse(points, model.rnn2)
#                cost_train_reg = tf.reduce_mean(tf.cond(tf.equal(presence, model.detected), rmse, big_error))
#                cost_train = tf.reduce_mean(tf.cond(tf.equal(presence, model.detected), rmse, big_error))
#                cost_test = tf.reduce_mean(tf.cond(tf.equal(presence, model.detected), rmse, big_error))
                
                presence_one_hot = tf.one_hot(tf.cast(presence, tf.int32), 2)
                correct_prediction = tf.cast(tf.equal(tf.argmax(presence_one_hot, 1), tf.argmax(model.detected, 1)), tf.float32)                
                correct = tf.cast(tf.equal(tf.cast(presence, tf.int64), tf.argmax(model.detected, 1)), tf.float32)
                accuracy = tf.reduce_mean(correct_prediction)
                
                cost_train_reg = tf.reduce_mean(rmse(points, model.pred)) + tf.reduce_mean(tf.divide(tf.abs(fl - model.pred_fl), fl))
                
#                cost_train_reg = tf.reduce_mean(presence*correct*rmse(points, model.pred) + (1-correct)*100)
#                cost_test = tf.reduce_mean(rmse(points, model.pred)*presence)
#                cost_train = cost_train_reg

#                cost_train_reg = tf.reduce_mean(mse(points, model.pred))
#                cost_train_reg = tf.contrib.losses.mean_pairwise_squared_error(model.pred, points)#l1(points, model.pred)#tf.reduce_mean(rmse(points, model.pred))

#                cost_train_reg = tf.contrib.losses.mean_squared_error(model.pred, points)
#                cost_train_reg = tf.contrib.losses.absolute_difference(model.pred, points)
                cost_test = tf.reduce_mean(rmse(points, model.pred))#tf.reduce_mean(rmse(points, model.pred)*presence)
                cost_train = tf.reduce_mean(rmse(points, model.pred))
                
                
#                cost_train_reg = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=presence,logits=model.detected))
#                cost_test = cost_train_reg
#                cost_train = cost_train_reg
                
                
#                cost_train = rmse(points, model.pred)
#                cost_test = rmse(points, model.pred)
#                cost_train_reg = rmse(tf.multiply(points, 227), tf.multiply(model.pred, 227))
#                cost_train = rmse(tf.multiply(points, 227), tf.multiply(model.pred, 227))
#                cost_test = rmse(tf.multiply(points, 227), tf.multiply(model.pred, 227))
                cost_train_fl = tf.reduce_mean(tf.divide(tf.abs(fl - model.pred_fl), fl))
                cost_test_fl = cost_train_fl
                
                val_summary = tf.summary.scalar("val", cost_test)
                tf.summary.scalar("train", cost_train)
#                fl_val_summary = tf.summary.scalar("fl_train", cost_train_fl)
#                tf.summary.scalar("fl_test", cost_test_fl)
            summary_op = tf.summary.merge_all()
               
            optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost_train_reg)#, global_step=batch)
#            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost_train_reg)
#            cost_train_reg = cost_train_fl# + 0.0001*(tf.nn.l2_loss(model.fcfl_1w)+tf.nn.l2_loss(model.fcfl_3w)) #+tf.nn.l2_loss(model.fc2w) 
#            cost_train = cost_train_fl
#            cost_test = cost_test_fl
#            optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).mpresenceize(cost_train_reg)#, global_step=batch)

                    
               
            saver = tf.train.Saver()
            config = tf.ConfigProto()
            config.gpu_options.allocator_type = 'BFC'
            with tf.Session(config = config) as sess:
                
                train_writer = tf.summary.FileWriter(summaries_dir + '/train', graph=sess.graph)
                val_writer = tf.summary.FileWriter(summaries_dir + '/val', graph=sess.graph)
                
#                sess.run(tf.group(tf.initialize_all_variables(), tf.initialize_local_variables()))
                sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
                model.load_retrained_weights(weights_path, sess)#21/03

#                saver.restore(sess, "checkpoints/model.ckpt")
                
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord, sess=sess)
            
            
                try:
                    step = 0
                    train_start_time = time.time()
                    epoch_losses = []
                    epoch_test_losses = []
                    epoch_acc = []
                    
                    all_epoch_losses = []
                    all_epoch_test_losses = []
                    
                    epoch_start_time = time.time()
                    print("----epoch {} ---------------".format(0))
                    current_epoch = 1
                    
#                    run_metadata = tf.RunMetadata()
                    while not coord.should_stop(): 
##########################################################                        
#                        ims, fl, pts = sess.run([images_batch_train, fl_batch_train, points_batch_train])

#                        for i in range(3):
#                            im = ims[i,:]
#                            gt = pts[i,:]
#                            f = fl[i]
#                            gt = gt.reshape((1002, 3))
#                            
#                            K_blender = np.array([[f,   0.0000, 112.0000],[0.0000, f, 112.0000],[0.0000,   0.0000,   1.0000]])
#                            print(gt)
#                            gt = gt[:,:3] / np.repeat(gt[:,2].reshape(1002,1),3,axis=1)
#                            gt = np.asarray(gt)
#                            gt = np.matmul(K_blender, gt.transpose()).transpose()     
#    
#                            fig = plt.figure(figsize=plt.figaspect(0.5))
#                            ax = fig.add_subplot(1, 2, 1)               
#        
#                                 
#                            ax.imshow(im)
#                            
#    #                        ax = fig.add_subplot(2,2,1)
#                            ax.plot(gt[:,0], gt[:,1], 'bx')
#                            
#                            plt.show()
#                        break
#########################################################
                        
                        start_time = time.time()
                        # Write the summaries and print an overview fairly often.
                        if step % int(train_size/batch_size/10) == 0:
                            
#                            _, loss, summary = sess.run([optimizer, cost_train, summary_op],feed_dict={keep_prob: do_train})
#                            _, loss, summary = sess.run([optimizer, cost_train, summary_op], options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),run_metadata=run_metadata)
                            _, loss, summary  = sess.run([optimizer, cost_train, summary_op])
                            train_writer.add_summary(summary, step)
                            epoch_losses.append(loss)
                            
                            
#                            im_test, pts_test = sess.run([images_batch_test, points_batch_test], options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),run_metadata=run_metadata)
                            im_test, fl_test, presence_test, pts_test = sess.run([images_batch_test, fl_batch_test, presence_batch_test, points_batch_test])
#                            summary, loss_test = sess.run([val_summary, cost_test], feed_dict={images: im_test, points: pts_test}, options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),run_metadata=run_metadata)
                            summary, loss_test, acc_val = sess.run([val_summary, cost_test, accuracy], feed_dict={images: im_test, fl:fl_test, presence:presence_test, points: pts_test})
                            epoch_acc.append(acc_val)
                            
                            val_writer.add_summary(summary, step)
                            epoch_test_losses.append(loss_test)
                            
							
                            #print('Step %d: train_loss = %.2f (%f sec)' % (step, loss, time.time() - start_time))
                            print('Step %d: train_loss = %.2f val_loss = %.2f test_acc = %.2f(%f s)' % (step, loss, loss_test, acc_val, time.time() - start_time))
#                            print(pres_val)
#                            print(correct_prediction.eval(session=sess))
                            
                        else:
#                            sess.run(optimizer,feed_dict={keep_prob: do_train})#, feed_dict={x: images_batch_train.eval(), y: points_batch_train.eval()})
#                            sess.run(optimizer, options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),run_metadata=run_metadata)#, feed_dict={x: images_batch_train.eval(), y: points_batch_train.eval()})
                            sess.run(optimizer)
                        #Test if epoch ended
                        if step > 0 and (step % int(train_size / batch_size) == 0):
                            with tf.device('/cpu:0'):
                                print("Epoch-- train: {} test: {} test_accuracy: {} Time elapsed: {} min".format(np.mean(epoch_losses),np.mean(epoch_test_losses),np.mean(epoch_acc),(time.time() - epoch_start_time)/60.0))
                                print("----epoch {} ---------------".format(current_epoch))
                                all_epoch_losses.append(np.mean(epoch_losses))
                                all_epoch_test_losses.append(np.mean(epoch_test_losses))
                                epoch_losses = []
                                epoch_acc = []
                                
                                epoch_test_losses = []
                                current_epoch += 1
                                epoch_start_time = time.time()
                                                                
                                if step > 0 and (step % (10*int(train_size / batch_size)) == 0):#######################
                                    saveWeights(model, fname, True)
#                                    print("Model saved in file: %s" % save_path)
#                                    saveWeights(model, 'latest_main', True)
#                                    save_path = saver.save(sess, "checkpoints/model.ckpt")
#                                    print("Model saved in file: %s" % save_path)
#                                    saver.save(sess, 'my_vgg16_model', global_step=step)
#                                if (len(all_epoch_test_losses) > 1) and  (all_epoch_test_losses[-1] < min(all_epoch_test_losses[:-1])):
#									saveWeights(model, fname+'_best', True)
#                                            pass
                                #if (len(all_epoch_test_losses) > 3) and (all_epoch_test_losses[-1]>all_epoch_test_losses[-2]) and (all_epoch_test_losses[-2] > all_epoch_test_losses[-3]):
								#	break
#                        if step == 15:
#                            trace = timeline.Timeline(step_stats=run_metadata.step_stats)
#                            trace_file = open('timeline.ctf.json', 'w')
#                            trace_file.write(trace.generate_chrome_trace_format())
#                            break
                        step += 1
                except tf.errors.OutOfRangeError:
                    print('Done training -- epoch limit reached\n Training time: {} min'.format((time.time()-train_start_time)/60))
                finally:
                    coord.request_stop()
                    coord.join(threads)
#                    save_path = saver.save(sess, "checkpoints/model.ckpt")
#                    print("Model saved in file: %s" % save_path)
                    saveWeights(model, fname, True)
#                    saveWeights(model, 'latest_main', True)
                    train_writer.close()
                    val_writer.close()

    if test:
        with tf.Graph().as_default():
            with tf.device('/cpu:0'):
                _, filenames_test = getFileList(testpath)
                images_batch_test, fl_batch_test, presence_batch_test, points_batch_test = input_pipeline(testpath, filenames_test, batch_size=1, num_epochs=1, read_threads=1, shuffle=False)
            
            x = tf.placeholder(tf.float32, [None, 224, 224, 3])
            y = tf.placeholder(tf.float32, [None, 3006])
            fl = tf.placeholder(tf.float32, shape=None)
            presence = tf.placeholder(tf.float32, shape=None)
#            keep_prob = tf.placeholder(tf.float32)
#            vgg = vgg16(x, keep_prob)
            model = vgg16(x)
            config = tf.ConfigProto()
            config.gpu_options.allocator_type = 'BFC'

            with tf.Session(config = config) as sess:
                sess.run(tf.group(tf.initialize_all_variables(), tf.initialize_local_variables()))

                model.load_retrained_weights(test_weights_path, sess)
                cost_test = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(y, model.pred))))
                cost_test_fl = tf.reduce_mean(tf.divide(tf.abs(fl - model.pred_fl), fl))
                
                presence_one_hot = tf.one_hot(tf.cast(presence, tf.int32), 2)
                correct = tf.cast(tf.equal(tf.cast(presence, tf.int64), tf.argmax(model.detected, 1)), tf.float32)
                accuracy = tf.reduce_mean(correct)
                
                
                mean, std = runTest(data=[images_batch_test, fl_batch_test, presence_batch_test, points_batch_test],cost=cost_test,cost_fl=cost_test_fl, accuracy=accuracy, sess=sess, print_step=1, saveData=True)
                

                    
#