import numpy as np
import tensorflow as tf
import os

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

tf.logging.set_verbosity(tf.logging.INFO)

# Our application logic will be added here

def dumb_neuron(alphas, inputs, verbose=None):
    if verbose is not None:
        print verbose
    for idx in range(len(alphas)):
        if idx == 0:
            total = alphas[idx] * inputs[idx]
        else:
            total = tf.add((alphas[idx] * inputs[idx]), total)


    cutoff = tf.zeros_like(total, dtype=tf.float32)
    ones = tf.ones_like(total, dtype=tf.float32)
    predictions = tf.less(total, cutoff)
    val_if_true = cutoff
    val_if_false = total
    floor_result = tf.where(predictions, val_if_true, val_if_false)
    cpredictions = tf.less(floor_result, ones)
    val_if_true = floor_result
    val_if_false = ones
    ceil_result = tf.where(cpredictions, val_if_true, val_if_false)

    return ceil_result

def better_neuron(alphas, inputs, bias, verbose=None):
    if verbose is not None:
        print verbose
    for idx in range(len(alphas)):
        if idx == 0:
            total = alphas[idx] * inputs[idx]
        else:
            total = tf.add((alphas[idx] * inputs[idx]), total)

    total = tf.subtract(total, bias)
    ceil_result = tf.sigmoid(total)
    return ceil_result

def proper_sigmoid_network(nlayers=2, nperlayer=2, num_inputs=2):
    x = tf.placeholder(tf.float32, shape=[None, num_inputs])

    if nlayers == 0:
        exit_weights = tf.get_variable("weight_ie", initializer=tf.ones([num_inputs,1]), dtype=tf.float32)
        exit_bias = tf.get_variable("bias_e", dtype=tf.float32, initializer=tf.zeros([1]))
        #exit_weights = tf.get_variable("weight_ie", shape=[num_inputs,1], dtype=tf.float32)
        #exit_bias = tf.get_variable("bias_e", dtype=tf.float32, shape=[1])
        last_neuron = tf.sigmoid(tf.matmul(x, exit_weights) + exit_bias)
    else:
        input_weights = tf.get_variable("weight_xi", initializer=tf.random_uniform([num_inputs, nperlayer], minval=-1, maxval=1), dtype=tf.float32)
        layer0_bias = tf.get_variable("bias_0", initializer=tf.zeros([nperlayer]), dtype=tf.float32)

        first_layer = tf.sigmoid(tf.matmul(x, input_weights) + layer0_bias)

        #multi layers
        current_layer = first_layer
        for i in range(1, nlayers+1):
            new_weights = tf.get_variable("weight_%d%d" % (i-1, i), initializer=tf.random_uniform([nperlayer,nperlayer], minval=-1, maxval=1), dtype=tf.float32)
            new_biases = tf.get_variable("bias_%d" % i, initializer=tf.zeros([nperlayer]), dtype=tf.float32)
            new_layer = tf.sigmoid(tf.matmul(current_layer, new_weights) + new_biases)
            current_layer = new_layer

        #exit layer
        exit_weights = tf.get_variable("weight_%de" % i, initializer=tf.random_uniform([nperlayer,1], minval=-1, maxval=1), dtype=tf.float32)
        exit_bias = tf.get_variable("bias_e", initializer=tf.zeros([1]), dtype=tf.float32)
        last_neuron = tf.sigmoid(tf.matmul(current_layer, exit_weights) + exit_bias)
    result = last_neuron
    #result = (tf.sign(last_neuron) + 1.) / 2.
    return result, x

def get_better_xor_network():
    nlayers = 2
    nperlayer = 2
    num_exit = 1
    num_inputs = 2
    input_variables = []
    input_biases = []
    # Define all Variables
    default = [0.]
    for j in range(nperlayer):
        this_input = []
        input_biases.append(tf.get_variable("bias%d%d" % (0, j), shape=[1],  dtype=tf.float32))
        for count in range(num_inputs):
            this_input.append(tf.get_variable("%s%d%d%d"%("i",count,0,j), shape=[1], dtype=tf.float32))
        input_variables.append(this_input)

    variables = []
    biases = []
    for i in range(nlayers-1):
        this_layer_variables = []
        this_layer_count = i+1
        this_b = []
        for j in range(nperlayer):
            this_var = []
            this_b.append(tf.get_variable("bias%d%d" % (this_layer_count, j), shape=[1], dtype=tf.float32))
            for count in range(nperlayer):
                this_var.append(tf.get_variable("%d%d%d%d"%(i,count,this_layer_count,j), shape=[1], dtype=tf.float32))
            this_layer_variables.append(this_var)
        biases.append(this_b)
        variables.append(this_layer_variables)

    exit_layer = []
    exit_bias = tf.get_variable("biase0", shape=[1], dtype=tf.float32)
    for i in range(nperlayer):
        exit_layer.append(tf.get_variable("%d%d%s%d"%(nlayers-1,i,"e",0), shape=[1], dtype=tf.float32))

    #define all inputs
    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)
    num_features = np.shape(x)[0]
    results = []
    first_values = []
    for i in range(nperlayer):
        res = better_neuron(input_variables[i], [x,y], input_biases[i])
        first_values.append(res)

    assert len(first_values) == nperlayer
    old_values = first_values
    for i in range(nlayers-1):
        this_alphas = variables[i]
        this_biases = biases[i]
        new_values = []
        for j in range(nperlayer):
            res = better_neuron(this_alphas[j], old_values, this_biases[j])
            new_values.append(res)

        old_values = new_values

    final_result = better_neuron(exit_layer, new_values, exit_bias)

    return final_result, x, y


def get_simple_network():
    first = tf.placeholder(tf.float32)
    second = tf.placeholder(tf.float32)
    a = tf.get_variable("a", initializer=[0.], dtype=tf.float32)
    b = tf.get_variable("b", initializer=[0.], dtype=tf.float32)
    q = tf.multiply(first, a)
    w = tf.multiply(second, b)
    final_neuron = tf.add(q,w)

    return final_neuron, first, second

def get_xor_network():
    nlayers = 3
    nperlayer = 4
    num_exit = 1
    num_inputs = 2
    input_variables = []
    # Define all Variables
    default = [0.25]
    for j in range(nperlayer):
        this_input = []
        for count in range(num_inputs):
            this_input.append(tf.get_variable("%s%d%d%d"%("i",count,0,j), initializer=default, dtype=tf.float32))
        input_variables.append(this_input)

    variables = []
    for i in range(nlayers-1):
        this_layer_variables = []
        this_layer_count = i+1
        for j in range(nperlayer):
            this_var = []
            for count in range(nperlayer):
                this_var.append(tf.get_variable("%d%d%d%d"%(i,count,this_layer_count,j), initializer=default, dtype=tf.float32))
            this_layer_variables.append(this_var)
        variables.append(this_layer_variables)

    exit_layer = []
    for i in range(nperlayer):
        exit_layer.append(tf.get_variable("%d%d%s%d"%(nlayers-1,i,"e",0), initializer=default, dtype=tf.float32))

    #define all inputs
    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)
    num_features = np.shape(x)[0]
    results = []
    first_values = []
    for i in range(nperlayer):
        res = dumb_neuron(input_variables[i], [x,y])
        first_values.append(res)

    assert len(first_values) == nperlayer
    old_values = first_values
    for i in range(nlayers-1):
        this_alphas = variables[i]
        new_values = []
        for j in range(nperlayer):
            res = dumb_neuron(this_alphas[j], old_values)
            new_values.append(res)

        old_values = new_values

    final_result = dumb_neuron(exit_layer, new_values)

    return final_result, x, y

def model_xor(features, labels, mode):
    nlayers = 3
    nperlayer = 4
    num_exit = 1
    num_inputs = 2
    input_variables = []
    # Define all Variables

    for j in range(nperlayer):
        this_input = []
        for count in range(num_inputs):
            this_input.append(tf.get_variable("%s%d%d%d"%("i",count,0,j), shape=[1], dtype=tf.float32))
        input_variables.append(this_input)

    variables = []
    for i in range(nlayers-1):
        this_layer_variables = []
        this_layer_count = i+1
        for j in range(nperlayer):
            this_var = []
            for count in range(nperlayer):
                this_var.append(tf.get_variable("%d%d%d%d"%(i,count,this_layer_count,j), shape=[1], dtype=tf.float32))
            this_layer_variables.append(this_var)
        variables.append(this_layer_variables)

    exit_layer = []
    for i in range(nperlayer):
        exit_layer.append(tf.get_variable("%d%d%s%d"%(nlayers-1,i,"e",0), shape=[1], dtype=tf.float32))

    #define all inputs
    x = features["x"]
    y = features["y"]
    num_features = np.shape(x)[0]
    results = []
    first_values = []
    for i in range(nperlayer):
        res = dumb_neuron(input_variables[i], [x,y])
        first_values.append(res)

    assert len(first_values) == nperlayer
    old_values = first_values
    for i in range(nlayers-1):
        this_alphas = variables[i]
        new_values = []
        for j in range(nperlayer):
            res = dumb_neuron(this_alphas[j], old_values)
            new_values.append(res)

        old_values = new_values

    final_result = dumb_neuron(exit_layer, new_values)

    loss = tf.reduce_sum(tf.square(final_result - labels))
    global_step = tf.train.get_global_step()
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = tf.group(optimizer.minimize(loss), tf.assign_add(global_step,1))
    return tf.contrib.learn.ModelFnOps(
        mode=mode,
        predictions=results,
        loss=loss,
        train_op=train)


def model_lr(features, labels, mode):
    # Build a linear model and predict values
    W = tf.get_variable("W", [1], dtype=tf.float32)
    b = tf.get_variable("b", [1], dtype=tf.float32)
    y = W*features["x"] + b
    # Loss sub-graph
    loss = tf.reduce_sum(tf.square(y - labels))
    # Training sub-graph
    global_step = tf.train.get_global_step()
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = tf.group(optimizer.minimize(loss),
                   tf.assign_add(global_step, 1))
    # ModelFnOps connects subgraphs we built to the
    # appropriate functionality.
    return tf.contrib.learn.ModelFnOps(
      mode=mode, predictions=y,
      loss=loss,
      train_op=train)


if __name__ == "__main__":
    linear_regression = False
    xor = True
    api = False # use the specialized API...
    cwd = os.getcwd()

    if xor and (not api):
        save_dir = "xor_training"
        check_point = "%s/xor_model.ckpt" % save_dir
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        xlist = [0., 0., 1., 1.]
        ylist = [0., 1., 0., 1.]
        target = [0., 1., 1., 0.]

        input_array = np.array([xlist, ylist]).transpose()
        target_array = np.reshape(np.array(target), (4,1)).astype(int)

        print "Training Data"
        print "Inputs:"
        print input_array
        print "Outputs:"
        print target_array
        print "shape of input: "
        print np.shape(input_array)
        num_inputs = np.shape(input_array)[1]

        final_neuron, first = proper_sigmoid_network(nperlayer=3, nlayers=1, num_inputs=num_inputs)

        target_neuron = tf.placeholder(tf.float32, shape=[4,1])
        #loss = tf.reduce_sum(target_neuron * tf.log(tf.clip_by_value(final_neuron, 1e-10, 1.0)))
        loss = tf.reduce_sum(tf.square(target_neuron - final_neuron))

        #optimizer = tf.train.GradientDescentOptimizer(0.01)
        optimizer = tf.train.AdamOptimizer()
        train = optimizer.minimize(loss)
        saver = tf.train.Saver()


        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)

        """
        for var in tf.trainable_variables():
            np.savetxt("xor_tests/%s-%s_initial" % (prefix,var.name), sess.run(var), fmt="%g")
        """
        if os.path.isfile("%s.index" % check_point):
            saver.restore(sess, check_point)

        print (sess.run(final_neuron, {first:input_array}))
        for i in range(10000):
            sess.run(train, {first:input_array, target_neuron:target_array})

        print sess.run([final_neuron, loss], {first:input_array, target_neuron:target_array})

        saver.save(sess, check_point)
        """
        if not os.path.isdir("xor_tests"):
            os.mkdir("xor_tests")

        for var in tf.trainable_variables():
            np.savetxt("xor_tests/%s-%s" % (prefix,var.name), sess.run(var), fmt="%g")
        """
    if xor and api:
        save_dir = "%s/parameters_xor" % cwd
        readable_save_dir = "%s/readable" % save_dir
        x_list = [0., 0., 1., 1.]
        ylist = [0., 1., 0., 1.]
        x = np.array(xlist)
        y = np.array(ylist)
        target = np.array([0., 1., 1., 0.])
        features = {"x":x, "y":y}

        """
        final_neuron, first, second = get_xor_network()

        init = tf.global_variables_initializer()

        sess = tf.Session()
        sess.run(init)
        print (sess.run(final_neuron, {first:xlist, second:ylist}))

        """
        estimator = tf.contrib.learn.Estimator(model_fn=model_xor, model_dir=save_dir)
        input_fn = tf.contrib.learn.io.numpy_input_fn(features, target, 4, num_epochs=1000)

        estimator.fit(input_fn=input_fn, steps=1000)
        all_variable_names = estimator.get_variable_names()
        for varname in all_variable_names:
            print "%s : %f" %(varname, estimator.get_variable_value(varname))
        train_loss = estimator.evaluate(input_fn=input_fn)
        print train_loss

    if linear_regression:
        x = np.array([1., 2., 3., 4.])
        target = np.array([0., -1., -2., -3.])
        features = {"x":x}
        estimator = tf.contrib.learn.Estimator(model_fn=model_lr)
        input_fn = tf.contrib.learn.io.numpy_input_fn(features, target, 4, num_epochs=1000)

        estimator.fit(input_fn=input_fn, steps=1000)
        train_loss = estimator.evaluate(input_fn=input_fn)
        print train_loss
        #features = {"x":x, "y":y}
