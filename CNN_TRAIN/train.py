import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.misc 
import scipy.io
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.tools import freeze_graph
print ("Packages loaded.")


cwd = os.getcwd()
loadpath = cwd + "/data/custom_data.npz"
print (loadpath)
l = np.load(loadpath)

# Show Files
print (l.files)

# Parse data
trainimg = l['trainimg']
trainlabel = l['trainlabel']
testimg = l['testimg']
testlabel = l['testlabel']
imgsize = l['imgsize']
use_gray = l['use_gray']
ntrain = trainimg.shape[0]
nclass = trainlabel.shape[1]
dim    = trainimg.shape[1]
ntest  = testimg.shape[0]
print ("%d train images loaded" % (ntrain))
print ("%d test images loaded" % (ntest))
print ("%d dimensional input" % (dim))
print ("Image size is %s" % (imgsize))
print ("%d classes" % (nclass))

tf.set_random_seed(0)
n_input  = dim
print ("input dimension", n_input, nclass)
n_output = nclass
use_gray = 0


if use_gray:
    weights  = {
        'wc1': tf.Variable(tf.random_normal([5, 5, 1, 128], stddev=0.1)),
        'wc2': tf.Variable(tf.random_normal([5, 5, 128, 128], stddev=0.1)),
        'wd1': tf.Variable(tf.random_normal(
                [(int)(imgsize[0]/4*imgsize[1]/4)*128, 128], stddev=0.1)),
        'wd2': tf.Variable(tf.random_normal([128, n_output], stddev=0.1))
    }
else:
    weights  = {
        'wc1': tf.Variable(tf.random_normal([5, 5, 3, 128], stddev=0.1)),
        'wc2': tf.Variable(tf.random_normal([5, 5, 128, 128], stddev=0.1)),
        'wd1': tf.Variable(tf.random_normal(
                [(int)(imgsize[0]/4*imgsize[1]/4)*128, 128], stddev=0.1)),
        'wd2': tf.Variable(tf.random_normal([128, n_output], stddev=0.1))
    }
biases   = {
    'bc1': tf.Variable(tf.random_normal([128], stddev=0.1)),
    'bc2': tf.Variable(tf.random_normal([128], stddev=0.1)),
    'bd1': tf.Variable(tf.random_normal([128], stddev=0.1)),
    'bd2': tf.Variable(tf.random_normal([n_output], stddev=0.1))
}


def conv_basic(_input, _w, _b, _keepratio, _use_gray):
	 """return the network layers dictionary


    :param str arg1 : img array
    :param str arg1 : weights
    :param str arg1 : biases
    :param str arg1 : keep ratio
    :param str arg1 : flag for using gray scalee
    :return : the network 
    :rtype : dictionary
    """
    # INPUT
    if _use_gray:
        _input_r = tf.reshape(_input, shape=[-1, imgsize[0], imgsize[1], 1])
    else:
        _input_r = tf.reshape(_input, shape=[-1, imgsize[0], imgsize[1], 3])
    # CONVOLUTION LAYER 1
    _conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(_input_r
        , _w['wc1'], strides=[1, 1, 1, 1], padding='SAME'), _b['bc1']))
    _pool1 = tf.nn.max_pool(_conv1, ksize=[1, 2, 2, 1]
        , strides=[1, 2, 2, 1], padding='SAME')
    _pool_dr1 = tf.nn.dropout(_pool1, _keepratio)
    # CONVOLUTION LAYER 2
    _conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(_pool_dr1
        , _w['wc2'], strides=[1, 1, 1, 1], padding='SAME'), _b['bc2']))
    _pool2 = tf.nn.max_pool(_conv2, ksize=[1, 2, 2, 1]
        , strides=[1, 2, 2, 1], padding='SAME')
    _pool_dr2 = tf.nn.dropout(_pool2, _keepratio)
    # VECTORIZE
    _dense1 = tf.reshape(_pool_dr2
                         , [-1, _w['wd1'].get_shape().as_list()[0]])
    # FULLY CONNECTED LAYER 1
    _fc1 = tf.nn.relu(tf.add(tf.matmul(_dense1, _w['wd1']), _b['bd1']))
    _fc_dr1 = tf.nn.dropout(_fc1, _keepratio)
    # FULLY CONNECTED LAYER 2
    _out = tf.add(tf.matmul(_fc_dr1, _w['wd2']), _b['bd2'])
    # RETURN
    out = {
        'input_r': _input_r, 'conv1': _conv1, 'pool1': _pool1
        , 'pool1_dr1': _pool_dr1, 'conv2': _conv2, 'pool2': _pool2
        , 'pool_dr2': _pool_dr2, 'dense1': _dense1, 'fc1': _fc1
        , 'fc_dr1': _fc_dr1, 'out': _out
    }
    return out

print ("NETWORK READY")



# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_output])
keepratio = tf.placeholder(tf.float32)

# Functions! 
_pred = conv_basic(x, weights, biases, keepratio, use_gray)['out']
Y_inf = tf.nn.softmax(_pred, name='modelOutput')

# softmax_cross_entropy_with_logits
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = _pred, labels=y))
WEIGHT_DECAY_FACTOR = 0.0001
l2_loss = tf.add_n([tf.nn.l2_loss(v) 
            for v in tf.trainable_variables()])
cost = cost + WEIGHT_DECAY_FACTOR*l2_loss
optm = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
_corr = tf.equal(tf.argmax(_pred,1), tf.argmax(y,1)) # Count corrects
accr = tf.reduce_mean(tf.cast(_corr, tf.float32)) # Accuracy
init = tf.initialize_all_variables()
print ("FUNCTIONS READY")



# Parameters
# training_epochs = 4
batch_size      = 100
display_step    = 1

TUTORIAL_NAME = 'classification'
MODEL_NAME = 'convnetTFon'
SAVED_MODEL_PATH = TUTORIAL_NAME+'_Saved_model/'

# Saving learned weights(variables) in checkpoint file
saver = tf.train.Saver()

with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())
    for step in range(15000):
    	avg_cost = 0.
    	num_batch = int(ntrain/batch_size)+1

    	for i in range(num_batch): 
    		randidx = np.random.randint(ntrain, size=batch_size)
	        batch_xs = trainimg[randidx, :]
	        batch_ys = trainlabel[randidx, :] 
        	trainStep,loss = sess.run([optm, cost], feed_dict={x: batch_xs, y: batch_ys, keepratio:0.7})
        	train_acc = sess.run(accr, feed_dict={x: batch_xs
                               , y: batch_ys, keepratio:1.})
        	trainStep, val_loss = sess.run([optm, cost], feed_dict={x: testimg, y: testlabel, keepratio:0.7})
        	val_accuracy = sess.run(accr, feed_dict={x: testimg
                               , y: testlabel, keepratio:1.})
        	print ("step : ", step, "train loss : ", loss, "accuracy : ", train_acc, "val loss : ", val_loss , "val accuracy :", val_accuracy)

        save_path = saver.save(sess, "checkpoints_2/" + str(step) + "_checkpoint.ckpt")
        tf.train.write_graph(sess.graph_def, SAVED_MODEL_PATH , MODEL_NAME + '.pbtxt')
        tf.train.write_graph(sess.graph_def, SAVED_MODEL_PATH , MODEL_NAME + '.pb',as_text=False)
        if step==37:
            print "losses after step ",step, " iteration: ",loss
            break

	    

	    print("Model saved in file: %s" % save_path)



# Freeze the graph
input_graph = SAVED_MODEL_PATH+MODEL_NAME+'.pb'
input_saver = ""
input_binary = True
input_checkpoint = 'checkpoints_2/36_checkpoint'+'.ckpt' # change this value TRAIN_STEPS here as per your latest checkpoint saved
output_node_names = 'modelOutput'
restore_op_name = 'save/restore_all'
filename_tensor_name = 'save/Const:0'
output_graph = SAVED_MODEL_PATH+'frozen_'+MODEL_NAME+'.pb'
clear_devices = True
initializer_nodes = ""
variable_names_blacklist = ""

freeze_graph.freeze_graph(
    input_graph,
    input_saver,
    input_binary,
    input_checkpoint,
    output_node_names,
    restore_op_name,
    filename_tensor_name,
    output_graph,
    clear_devices,
    initializer_nodes,
    variable_names_blacklist
)

