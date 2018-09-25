import numpy as np
import os
import tensorflow as tf


from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

tf.logging.set_verbosity(tf.logging.INFO)

# Our application logic will be added here

def get_simple_sigmoid(input_dims):
    #for traeting a picture as a 1-d data set and seeing what happens
    inputs = tf.placeholder(tf.float32, shape=[None,input_dims])
    weights = tf.get_variable("weights", initializer=tf.random_uniform([input_dims, 10], minval=-1., maxval = 1.))
    bias = tf.get_variable("biases", initializer=tf.zeros(10))

    output = tf.sigmoid(tf.matmul(inputs, weights) + bias)

    return inputs, output

def get_holdout_data(data, target):
    # divide data set into training and test data set
    n_data_points = np.shape(data)[0]
    test_idxs = np.random.choice(np.arange(n_data_points), size=5000, replace=False)
    training_idxs = []
    for i in range(n_data_points):
        if i not in test_idxs:
            training_idxs.append(i)

    training_answers = target[test_idxs, :]
    training_data = data[test_idxs, :]

    validation_answers = target[training_idxs, :]
    validation_data = data[training_idxs, :]

    return training_data, training_answers, validation_data, validation_answers

def get_convolutional_network():
    inputs = tf.placeholder(tf.float32, shape=[None,28*28])

    convolutional_inputs = tf.reshape(inputs, [-1, 28, 28, 1]) #shapei n batch, height, width, channels
    weights_first = tf.get_variable("weights_first", initializer=tf.constant(0.1, shape=[5,5,1,32]))
    first_conv_output = tf.nn.conv2d(convolutional_inputs, weights_first, strides=[1,1,1,1], padding="SAME") # 32 5x5 filters with zero padding
    first_pool_output = tf.nn.max_pool(first_conv_output, ksize=[1, 2, 2, 1], strides=[1,2,2,1], padding="SAME")

    weights_second = tf.get_variable("weights_second", initializer=tf.constant(0.1, shape=[5,5,32,32]))
    second_conv_output = tf.nn.conv2d(first_pool_output, weights_second, strides=[1,1,1,1], padding="SAME")
    second_pool_output = tf.nn.max_pool(second_conv_output, ksize=[1, 2, 2, 1], strides=[1,2,2,1], padding="SAME")

    fully_reshape = tf.reshape(second_pool_output, [-1, 7*7*32]) # should just be a row of 16 possible things.

    final_variables = tf.get_variable("weights_final", initializer=tf.constant(0.1, shape=[7*7*32, 10]))
    final_biases = tf.get_variable("bias_final", initializer=tf.constant(0., shape=[10]))

    final_output = tf.matmul(fully_reshape, final_variables) + final_biases

    return inputs, final_output

def train_model(sess, model_name, training_data, target_answers, validation_data=None, validation_answers=None, restore=True):
    save_dir = model_name
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    check_point_file = "%s/simple.ckpt" % save_dir
    if (validation_data is None) or (validation_answers is None):
        calculate_validation = False
        v_acc = 0.
    else:
        calculate_validation = True

    max_iter = 10000
    epoch_size = 100
    accuracy_cutoff = 10 ** (-11)
    #in_neuron, out_neuron = get_simple_sigmoid(np.shape(training_data)[1])
    in_neuron, raw_out_neuron = get_convolutional_network()
    target_neuron = tf.placeholder(tf.float32, shape=[None,10])
    #loss = tf.reduce_sum(tf.square(target_neuron - out_neuron))

    loss = tf.nn.softmax_cross_entropy_with_logits(labels=target_neuron, logits=raw_out_neuron)
    out_neuron = tf.sigmoid(raw_out_neuron)

    predictions_check = tf.equal(tf.argmax(out_neuron,axis=1), tf.argmax(target_neuron,axis=1))
    accuracy = tf.reduce_mean(tf.cast(predictions_check, tf.float32))
    optimizer = tf.train.AdamOptimizer(1e-4)
    train = optimizer.minimize(loss)
    saver = tf.train.Saver()

    init = tf.global_variables_initializer()
    sess.run(init)

    if restore and os.path.isfile("%s.index" % check_point_file):
        try:
            saver.restore(sess, check_point_file)
        except:
            print "Attempted Restoration but Failed"


    print "Starting Accuracy: %f" % sess.run(accuracy, {in_neuron:training_data, target_neuron:target_answers})


    go = True
    count = 0
    iterations = []
    validate_accuracy = []
    training_accuracy = []
    old_accuracy = 0.
    t_acc = sess.run(accuracy, {in_neuron:training_data, target_neuron:target_answers})
    if calculate_validation:
        v_acc = sess.run(accuracy, {in_neuron:validation_data, target_neuron:validation_answers})
    iterations.append(count)
    validate_accuracy.append(v_acc)
    training_accuracy.append(t_acc)
    while go:
        idxs = np.random.choice(np.arange(np.shape(training_data)[0]), size=1000, replace=False)
        in_sample = training_data[idxs, :]
        out_sample = target_answers[idxs,:]
        sess.run(train, {in_neuron:in_sample, target_neuron:out_sample})
        count += 1

        if (count % epoch_size) == 0:
            print "On Iteration %d" % count
            t_acc = sess.run(accuracy, {in_neuron:training_data, target_neuron:target_answers})
            if calculate_validation:
                v_acc = sess.run(accuracy, {in_neuron:validation_data, target_neuron:validation_answers})
            iterations.append(count)
            validate_accuracy.append(v_acc)
            training_accuracy.append(t_acc)
            saver.save(sess, check_point_file)

        if count >= max_iter:
            go = False
            t_acc = sess.run(accuracy, {in_neuron:training_data, target_neuron:target_answers})
            if calculate_validation:
                v_acc = sess.run(accuracy, {in_neuron:validation_data, target_neuron:validation_answers})
            iterations.append(count)
            validate_accuracy.append(v_acc)
            training_accuracy.append(t_acc)

    stuff = np.array([iterations, validate_accuracy, training_accuracy]).transpose()
    np.savetxt("%s/training_results" % save_dir, stuff, fmt="%d     %g  %g")
    print "Final Accuracy: %f" % sess.run(accuracy, {in_neuron:training_data, target_neuron:target_answers})
    saver.save(sess, check_point_file)

if __name__ == "__main__":
    #input_data = np.loadtxt("train.csv", skiprows=1, delimiter=",") #training data
    input_data = np.loadtxt("small_training_data.dat")
    #evaluate_data = np.loadtxt("test.csv", skiprows=1, delimiter=",")

    model_name = "conv2d"

    answers = input_data[:,0].astype(int)
    all_data = input_data[:,1:] / 255.
    target_answers = np.zeros((np.shape(answers)[0],10))
    for idx,jdx in enumerate(answers):
        target_answers[idx,jdx] = 1.

    training_data, training_answers, validation_data, validation_answers = get_holdout_data(all_data, target_answers)

    sess = tf.Session()

    train_model(sess, model_name, all_data, target_answers, restore=False)
