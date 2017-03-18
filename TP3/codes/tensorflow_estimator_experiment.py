from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python import debug as tf_debug
from tensorflow.contrib import layers
import tensorflow as tf
import numpy as np
import utils
"""
 Bonus code: Experimentation with Tensorflow Estimator... well I didn't like it
 
"""
epsilon = 1e-4
tf.logging.set_verbosity(tf.logging.INFO)


def get_input():
    def reshape_input(data):
        return np.reshape(data.images, (-1, 28 * 28)).astype(np.float32), data.labels.astype(np.float32)

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    return reshape_input(mnist.train), reshape_input(mnist.validation), reshape_input(mnist.test)


def model_fn(features, targets, mode, params):
    # Input shape
    x = tf.reshape(features, (-1, 28 * 28))

    # Retrieve parameters
    activation = params["activation"]
    lys = params["layers"]
    dropout_p = params["dropout"]
    l1_reg = params["l1_reg"]
    l2_reg = params["l2_reg"]
    lr = params["learning_rate"]
    optimizer = params["optimizer"]
    is_training = True if (mode == "train") else False

    for layer_shape in lys:
        x = layers.dropout(
            activation(
                layers.fully_connected(inputs=x,
                                       num_outputs=layer_shape
                                       # weights_regularizer=
                                       # utils.l1_l2_regularizer(l1_reg, l2_reg)
                                       )),
            is_training=is_training,
            keep_prob=dropout_p)

    x = tf.nn.softmax(x)
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(targets * tf.log(x), axis=1))
    loss = cross_entropy  # + tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    accuracy = tf.reduce_mean(
        tf.cast(tf.equal(tf.argmax(x, 1), tf.argmax(targets, 1)), tf.float32),
        name="accuracy")

    trainable_variable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    for var in trainable_variable:
        tf.summary.histogram(var.op.name, var)

    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=lr,
        optimizer=(lambda l_r: optimizer(l_r)),
        summaries=tf.GraphKeys.TRAIN_OP)

    return model_fn_lib.ModelFnOps(
        mode=mode,
        predictions=tf.argmax(x, 1),
        loss=loss,
        train_op=train_op)


def to_string(model_params):
    return "lr:{}bn{}".format(model_params["learning_rate"], model_params["batch_norm"])


def accuracy(predictions, labels):
    return predictions


def main(model_params):
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = get_input()

    validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
        x=x_val,
        y=y_val,
        early_stopping_metric="loss",
        early_stopping_metric_minimize=True,
        early_stopping_rounds=3)

    tensors_to_log = {"Accuracy": "accuracy"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=10)

    nn = tf.contrib.learn.Estimator(model_fn=model_fn,
                                    params=model_params,
                                    model_dir="logs/" + to_string(model_params))
    # config=tf.contrib.learn.RunConfig(save_checkpoints_secs=10))

    nn.fit(x=x_train, y=y_train, steps=200, monitors=[logging_hook, validation_monitor])
    print(nn.evaluate(x=x_test, y=y_test, batch_size=10))


def new_test():
    model_params = {"batch_norm": True,
                    "dropout": 1,
                    "layers": [256, 10],
                    "activation": tf.nn.relu,
                    "optimizer": tf.train.AdamOptimizer,
                    "learning_rate": 0.001,
                    "l1_reg": 0.0,
                    "l2_reg": 0.0}

    main(model_params)


new_test()
