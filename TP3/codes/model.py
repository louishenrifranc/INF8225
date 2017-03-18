from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.framework import ops
from tensorflow.contrib import layers
from termcolor import cprint
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import random
import shutil
import utils

epsilon = 1e-4
# tf.logging.set_verbosity(tf.logging.INFO)
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


class Model:
    def __init__(self, args):
        """
        Build a Classification model
        :param args: argparse.Namespace
            A list of variable representing the model
        """
        self.lr = args.lr
        self.layers = args.layers
        self.l1_reg = args.l1_reg
        self.l2_reg = args.l2_reg
        self.nb_iter = args.nb_iter
        self.patience = args.patience
        self.dropout_p = args.dropout_p
        self.input_size = args.input_size
        self.batch_size = args.batch_size
        self.nb_targets = args.nb_targets
        self.activation = args.activation
        self.optimizer = args.optimizer
        self.batch_norm = layers.batch_norm if args.batch_norm == True else None
        ops.reset_default_graph()
        self.sess = tf.InteractiveSession()
        self.experiment_name = args.experiment_name

    def _build_placeholder(self):
        """
        Helper to create all placeholder in the graph
        :return:
        """
        self.x = tf.placeholder(tf.float32, (None, self.input_size ** 2))
        self.y = tf.placeholder(tf.float32, [None, self.nb_targets])
        self.is_training = tf.placeholder(tf.bool, name="is_training")
        self.learning_rate_placholder = tf.placeholder(dtype=tf.float32)
        self.optimizer = self.optimizer(learning_rate=self.learning_rate_placholder)

    def _feed_dict(self, batch):
        """
        Helper to feed next batch
        :param batch: tf.Dataset
            mnist.train or mnist.test or mnist.validation
        :param batch_size: Int (default: None)
            Batch size, or the first dimension of all placeholder
        :return:
            A dictionnary to use to feed the graph placeholder
        """
        is_training = True if batch == mnist.train else False
        batch_x, batch_y = batch.next_batch(self.batch_size)
        return {
            self.x: batch_x,
            self.y: batch_y,
            self.is_training: is_training,
            self.learning_rate_placholder: self.lr
        }

    def _forward(self):
        """
        Helper to create the forward pass in the graph
        :return:
        """
        x = self.x
        # Reshape input
        x = tf.reshape(x, (-1, self.input_size, self.input_size, 1))
        # List of length maintained during the building because some
        # dimensions can't be inferred at build time, such as the Dense layer size
        length = [self.input_size, self.input_size, 1]
        # Implement dropout every three layers
        nb_dropout_layers = len(self.layers) // 3
        # Iterate over all built State
        for index in range(len(self.layers)):
            prev_layer = None
            # Pass the previous shape (needed for reshaping)
            if index > 0:
                prev_layer = self.layers[index - 1]
            if index % 3 == 0 and index > 0:
                keep_prob = 1 - (index // 3) / (2 * nb_dropout_layers)
                x = layers.dropout(inputs=x, is_training=self.is_training,
                                   keep_prob=keep_prob)
            x, length = utils.get_layer(x, self.layers[index], self.nb_targets, length, prev_layer)
            # No activation, at the last layer
            if index != len(self.layers) - 1:
                x = self.activation(x)
        return x

    def _loss(self, x):
        """
        Helper to create variable in the graph for loss and accuracy
        :param x: tf.Tensor
            Output of the forward pass
        :return:
        """
        x = tf.nn.softmax(x)
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y * tf.log(x), axis=1))
        self.loss = cross_entropy  # + tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        self.accuracy = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(x, 1), tf.argmax(self.y, 1)), tf.float32))

    def _optimize(self):
        """
        Helper to create mechanism for computing the derivative wrt to the loss
        :return:
        """
        train_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        # Update batch norm mean and variance which are not part of the graph by default
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            grads = self.optimizer.compute_gradients(loss=self.loss, var_list=train_variables)
            self.train_dis = self.optimizer.apply_gradients(grads)

    def _summary(self):
        """
        Helper that create operation for tf.Summary
        :return:
        """
        trainable_variable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        for var in trainable_variable:
            tf.summary.histogram(var.op.name, var)
        self.merged_summary_op = tf.summary.merge_all()

    def _restore(self):
        """
        Helper to restore a model
        :return:
        """
        self.sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=5)
        last_saved_model = tf.train.latest_checkpoint('model')
        if last_saved_model is not None:
            saver.restore(self.sess, last_saved_model)
            print("Restoring model  {}".format(last_saved_model))
        return saver

    def build(self):
        """
        Build the graph
        :return:
        """
        self._build_placeholder()
        x = self._forward()
        self._loss(x)
        self._optimize()
        self._summary()

    def train(self, lab):
        """
        Train the graph
        :param lab: QTable
            QTable will decide to save or not the model depending on the previous computation
        :return:
        """
        summary_writer = tf.summary.FileWriter('logs/{}'.format(self.experiment_name), graph=self.sess.graph,
                                               flush_secs=20)
        self.sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        for epoch in tqdm(range(self.nb_iter)):
            self._train_epoch()

            accuracy, summary_accuracy, summary_loss = self._test_epoch(mnist.validation)
            summary_writer.add_summary(summary_accuracy, epoch)
            summary_writer.add_summary(summary_loss, epoch)

            if epoch == 0:
                retry = 0
                while accuracy <= 0.5 and retry < 1:  # In the paper it's 5!
                    cprint("Model is worst than random, retry {}".format(retry), color="red")
                    self.lr /= 4
                    self._train_epoch()
                    accuracy, _, _ = self._test_epoch(mnist.validation)
                    retry += 1
                if accuracy <= 0.5:
                    break
            if epoch % 5 == 0:
                self.lr *= 0.2

        test_accuracy, summary_accuracy, summary_loss = self._test_epoch(mnist.test)
        if lab.to_save(test_accuracy, self.experiment_name):
            summary_writer.add_summary(summary_accuracy, 1)
            summary_writer.add_summary(summary_loss, 1)
            summary_writer.add_summary(self.sess.run(self.merged_summary_op))
            saver.save(self.sess, global_step=0, save_path="model/{}".format(self.experiment_name))
        summary_writer.flush()
        summary_writer.close()

        self.sess.close()
        return test_accuracy

    def _train_epoch(self):
        for itr in range(mnist.train.images.shape[0] // self.batch_size):
            self.sess.run([self.train_dis], feed_dict=self._feed_dict(mnist.train))

    def _test_epoch(self, batch):
        name = "test" if batch == mnist.test else "val"
        mean_accuracy, mean_loss = 0, 0
        nb_itr = batch.images.shape[0] // self.batch_size
        for itr in range(nb_itr):
            accuracy_itr, loss_itr = self.sess.run([self.accuracy, self.loss],
                                                   feed_dict=self._feed_dict(mnist.validation))
            mean_accuracy += accuracy_itr
            mean_loss += loss_itr
        mean_accuracy /= nb_itr
        mean_loss /= nb_itr
        summary_accuracy = tf.Summary(value=[
            tf.Summary.Value(tag="mean_accuracy{}".format(name), simple_value=mean_accuracy),
        ])
        summary_loss = tf.Summary(value=[
            tf.Summary.Value(tag="summary_loss{}".format(name), simple_value=mean_loss),
        ])
        return mean_accuracy, summary_accuracy, summary_loss
