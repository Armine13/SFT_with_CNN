from vgg16_model_do1_2fc import vgg16

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

def preprocessImage(im_tensor):
    #mean = tf.constant([69, 68, 64], shape=[1, 1, 3], name='img_mean', dtype=tf.uint8)
    #image = tf.cast(tf.cast(im_tensor,tf.int32) - tf.cast(mean,tf.int32), tf.float32)
    image = tf.image.per_image_whitening(im_tensor)
    return image

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
    filename_queue = tf.train.string_input_producer(filenames_tensor,num_epochs=num_epochs, shuffle=True)


    min_after_dequeue = 1000
    capacity = min_after_dequeue + 3 * batch_size

    data_list=[(im, pts) for im, pts in [read_files(path,filename_queue) for _ in range(read_threads)]]

    [(im.set_shape([224, 224, 3]), pt.set_shape((3006,))) for (im,pt) in data_list]
    data_list = [(preprocessImage(im),pt) for (im, pt) in data_list]
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

def saveWeights(vgg, fname, print_message = False):
    #Saving the weights
    weights = {}
    layer_arch = [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2), (3, 3), (4, 1), (4, 2), (4, 3), (5, 1), (5, 2), (5, 3)]
    for block, layer in layer_arch:
        exec("weights['conv{}_{}_Kernel'] = vgg.conv{}_{}_kernel.eval()".format(block, layer, block, layer))
        exec("weights['conv{}_{}_biases'] = vgg.conv{}_{}_biases.eval()".format(block, layer, block, layer))

    for l in range(1,4):
        exec("weights['fc{}_W'] = vgg.fc{}w.eval()".format(l, l))
        exec("weights['fc{}_b'] = vgg.fc{}b.eval()".format(l, l))

    full_name = 'weights/weights_' + fname
    np.savez(full_name, **weights)
    if print_message:
        print('weights saved to {}.npz.'.format(full_name))

def runTest(data, cost, sess, print_step=10, saveData=False):
    images = data[0]
    points = data[1]
    losses = []
    fname = str(time.time())
    coord2 = tf.train.Coordinator()
    threads2 = tf.train.start_queue_runners(coord=coord2, sess=sess)
    n_saved = 60
    data = {}
    data['pred'] = np.empty((n_saved, 3006))
    data['gt'] = np.empty((n_saved, 3006))
    try:
        print("Testing..")
        step = 0
        while not coord2.should_stop():
            
            start_time = time.time()
            image_test, points_test = sess.run([images, points])
            test_loss = cost.eval(feed_dict={x: image_test, y:points_test,keep_prob: 1.0})
            losses.append(test_loss)
            
            duration = time.time() - start_time
            if print_step!=-1 and step % print_step == 0:
                print('Step %d: loss = %.2f (%.3f sec)' % (step, test_loss, duration))
            if saveData and step < n_saved:
                data['gt'][step,:] = points_test
                data['pred'][step,:] = vgg.pred.eval(feed_dict={x: image_test, y:points_test,keep_prob: 1.0})
            step += 1
    except tf.errors.OutOfRangeError:
        pass
    finally:
        coord2.request_stop()
        coord2.join(threads2)
    mean_loss = np.mean(losses)
    print("Mean testing loss: {} min={} max={}".format(mean_loss, min(losses), max(losses)))

    if saveData:
        np.savez('results/test'+fname+'.npz', **data)
        print("Results saved to {}.".format('results/test'+fname+'.npz'))
    
    return mean_loss        

            
if __name__ == '__main__':
    trainpath = "../datasets/dataset_rt+fl+l/train"
    testpath = "../datasets/dataset_rt+fl+l/test"#############################

        
#    weights_path = 'weights/weights_1491387395.44.npz'#weights_1491396095.94.npz'
#    weights_path = 'weights/weights_latest.npz'#'weights/weights_fc_1491233872.99.npz'
    weights_path = 'weights/weights_latest__do1_2fc.npz'
    
    test_weights_path = 'weights/weights_latest__do1_2fc.npz'
    #from glob import glob1
    #weight_files = glob1("weights/", "*.npz")
    #weight_files = ['weights/'+f for f in weight_files]
    #for test_weights_path in weight_files:
    learning_rate = 0.00001
#    lr_decay = 0.9
    reg_constant = 0.23
    read_threads = 28
    num_epochs = 60
    batch_size = 5
    train = True
    test = False
    if train:
		with tf.Graph().as_default():
			with tf.device('/cpu:0'):
				_, filenames_train = getFileList(trainpath)
				
				train_size = len(filenames_train)
				_, filenames_test = getFileList(testpath)

				images_batch_train, points_batch_train = input_pipeline(trainpath, filenames_train, batch_size, num_epochs=num_epochs, read_threads=read_threads)
				images_batch_test, points_batch_test = input_pipeline(testpath, filenames_test, batch_size=batch_size, num_epochs=num_epochs, read_threads=read_threads)
				
				images = tf.placeholder_with_default(images_batch_train, shape=[None, 224, 224, 3])
				points = tf.placeholder_with_default(points_batch_train, shape=[None, 3006])
				
				#images = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])
				#points = tf.placeholder(tf.float32, shape=[None, 3006])
				
				keep_prob = tf.placeholder(tf.float32)
				
#            x = tf.placeholder(tf.float32, [None, 224, 224, 3])
#            y = tf.placeholder(tf.float32, [None, 3006])
			vgg = vgg16(images, keep_prob)
#            vgg = vgg16(images_batch_train)
			
			
			reg_losses = tf.nn.l2_loss(vgg.fc1w)+tf.nn.l2_loss(vgg.fc3w) #+tf.nn.l2_loss(vgg.fc2w) 
#            cost_train = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(y, vgg.pred)))) + reg_constant * reg_losses
#            cost_test = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(y, vgg.pred))))
			cost_train_reg = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(points, vgg.pred)))) + reg_constant * reg_losses
			cost_test = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(points, vgg.pred))))
			
			
			fname = str(time.time())#timestamp used for saved file names

			
			   
#            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
#            with tf.control_dependencies(update_ops):
			#optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost_train_reg)#, global_step=batch)
			optimizer = tf.train.RMSPropOptimizer(learning_rate = learning_rate).minimize(cost_train_reg)

			config = tf.ConfigProto()
			config.gpu_options.allocator_type = 'BFC'
			with tf.Session(config = config) as sess:

				sess.run(tf.group(tf.initialize_all_variables(), tf.initialize_local_variables()))
				vgg.load_retrained_weights(weights_path, sess)#21/03

				coord = tf.train.Coordinator()
				threads = tf.train.start_queue_runners(coord=coord, sess=sess)
				
				
				try:
					step = 0
					train_start_time = time.time()
					epoch_losses = []
					epoch_test_losses = []
					all_epoch_losses = []
					epoch_start_time = time.time()
					print("----epoch {} ---------------".format(0))
					current_epoch = 1
					while not coord.should_stop():    
						start_time = time.time()
						
				#        im_train, pts_train = sess.run([images_batch_train, points_batch_train])

						
						# Write the summaries and print an overview fairly often.
						if step % 200 == 0:
							_, loss = sess.run([optimizer, cost_test],feed_dict={keep_prob: 0.98})
							epoch_losses.append(loss)

							im_test, pts_test = sess.run([images_batch_test, points_batch_test])
							loss_test = sess.run(cost_test, feed_dict={images: im_test, points: pts_test,keep_prob: 1.0})
							epoch_test_losses.append(loss_test)
							
							#print('Step %d: train_loss = %.2f (%f sec)' % (step, loss, time.time() - start_time))
							print('Step %d: train_loss = %.2f test_loss = %.2f (%f sec)' % (step, loss, loss_test, time.time() - start_time))
						else:
							sess.run(optimizer,feed_dict={keep_prob: 0.98})#, feed_dict={x: images_batch_train.eval(), y: points_batch_train.eval()})
						#Test if epoch ended
						if step > 0 and (step % int(train_size / batch_size) == 0):
							with tf.device('/cpu:0'):
								print("Mean train Loss: {}  Mean test Loss: {} Time elapsed: {} min".format(np.mean(epoch_losses),np.mean(epoch_test_losses),(time.time() - epoch_start_time)/60.0))
								print("----epoch {} ---------------".format(current_epoch))
								all_epoch_losses.append(np.mean(epoch_losses))
								epoch_losses = []
								epoch_test_losses = []
								current_epoch += 1
								epoch_start_time = time.time()
																
								if step > 0 and (step % (30*int(train_size / batch_size)) == 0):#######################
									saveWeights(vgg, fname+str(step), True)
							
						step += 1
				except tf.errors.OutOfRangeError:
					print('Done training -- epoch limit reached\n Training time: {} min'.format((time.time()-train_start_time)/60))
				finally:
					coord.request_stop()
					coord.join(threads)
					#saveWeights(vgg, fname, True)
					saveWeights(vgg, 'latest__do1_2fc', True)
					np.savetxt('weights/'+fname+'_do1_2fc_losses.txt', all_epoch_losses)
    if test:
        with tf.Graph().as_default():
			with tf.device('/cpu:0'):
				_, filenames_test = getFileList(testpath)
				images_batch_test, points_batch_test = input_pipeline(testpath, filenames_test, batch_size=1, num_epochs=1, read_threads=1)
			
			x = tf.placeholder(tf.float32, [None, 224, 224, 3])
			y = tf.placeholder(tf.float32, [None, 3006])
			keep_prob = tf.placeholder(tf.float32)
			vgg = vgg16(x, keep_prob)
			config = tf.ConfigProto()
			config.gpu_options.allocator_type = 'BFC'

			with tf.Session(config = config) as sess:
				sess.run(tf.group(tf.initialize_all_variables(), tf.initialize_local_variables()))

				vgg.load_retrained_weights(test_weights_path, sess)
				cost_test = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(y, vgg.pred))))
				
				runTest(data=[images_batch_test, points_batch_test],cost=cost_test,sess=sess, print_step=10, saveData=False)
