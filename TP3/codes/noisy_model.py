import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import layers
from tqdm import tqdm
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import numpy as np

K.set_image_dim_ordering('tf')
epsilon = 1e-4

mnist = input_data.read_data_sets('MNIST_data', one_hot=True, validation_size=5000)


def __reshape_mnist_image(batch, twoD_to_fourD=False):
    if twoD_to_fourD:
        return np.reshape(batch, (len(batch), 28, 28, 1))
    else:
        return np.reshape(batch, (len(batch), 784))


def get_next_batch_training(data_generator, batch, batch_size):
    nb_batches = 0
    for x_batch, y_batch in data_generator.flow(__reshape_mnist_image(batch.images, True), batch.labels,
                                                batch_size=batch_size):
        if nb_batches == (len(batch.images) // batch_size):
            break
        nb_batches += 1
        yield __reshape_mnist_image(x_batch), y_batch


def get_data_transformer(dataset):
    datagen = ImageDataGenerator(featurewise_center=True,  # Centrer les données
                                 featurewise_std_normalization=True,  # Reduire les données
                                 zca_whitening=True,  # Decorreler les pixels
                                 width_shift_range=0.2,  # Shifter l'image en largeur
                                 height_shift_range=0.2)  # Shifter l'image en hauteur
    datagen.fit(__reshape_mnist_image(dataset, True), augment=True)
    return datagen


train_gene = get_data_transformer(mnist.train.images)
val_gene = get_data_transformer(mnist.validation.images)
test_gene = get_data_transformer(mnist.test.images)


class NN:
    def __init__(self, args):
        """
        Build a Classification model
        :param args: argparse.Namespace
            A list of variable representing the model
        """
        self.lr = args.lr
        self.l1_reg = args.l1_reg
        self.l2_reg = args.l2_reg
        self.nb_epochs = args.nb_iter
        self.dropout_p = args.dropout_p
        self.input_size = args.input_size
        self.batch_size = args.batch_size
        self.nb_targets = args.nb_targets
        self.activation = args.activation
        self.optimizer = args.optimizer

        self.global_step = tf.Variable(initial_value=0, trainable=False, name="global_step")
        self.sess = tf.InteractiveSession()

    def _build_placeholder(self):
        """
        Helper to create all placeholder in the graph
        :return:
        """
        self.x = tf.placeholder(tf.float32, (None, self.input_size))
        self.y = tf.placeholder(tf.float32, [None, self.nb_targets])
        self.is_training = tf.placeholder(tf.bool, name="is_training")
        self.learning_rate_placholder = tf.placeholder(dtype=tf.float32)
        self.optimizer = self.optimizer(learning_rate=self.learning_rate_placholder)

    def _forward(self):
        """
        Helper to create the forward pass in the graph
        :return:
        """

        tf.summary.image("image", tensor=tf.reshape(self.x, (self.batch_size, 28, 28, 1)), max_outputs=10)
        x = self.x

        # x = layers.dropout(self.x, keep_prob=0.7)
        # with tf.variable_scope("layer1") as scope:
        h = tf.nn.relu(layers.fully_connected(x, num_outputs=self.input_size // 2, activation_fn=None))
        # tf.summary.histogram("moving_mean1", tf.get_variable(scope + "moving_mean"))
        # with tf.variable_scope("layer2") as scope:
        #     h = tf.nn.relu(layers.fully_connected(h, num_outputs=32, activation_fn=None))
        # tf.summary.histogram("moving_mean2", tf.get_variable("moving_mean"))
        # with tf.variable_scope("layer3") as scope:
        self.logits = layers.fully_connected(h, num_outputs=10, activation_fn=None)
        # tf.summary.histogram("moving_mean3", tf.get_variable("moving_mean"))

        self.probability = tf.nn.softmax(self.logits)
        self.prediction = tf.argmax(self.probability, axis=1)

    def _loss(self):
        """
        Helper to create variable in the graph for loss and accuracy
        :param x: tf.Tensor
            Output of the forward pass
        :return:
        """

        cross_entropy = tf.reduce_mean(-tf.log(self.probability + epsilon) * self.y)
        self.loss = cross_entropy

        self.accuracy = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(self.y, 1), self.prediction), tf.float32))

    def _optimize(self):
        """
        Helper to create mechanism for computing the derivative wrt to the loss
        :return:
        """
        # Retrieve all trainable variables
        train_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        # Compute the gradient (return a pair of variable and their respective gradient)
        grads = self.optimizer.compute_gradients(loss=self.loss, var_list=train_variables)
        self.train_dis = self.optimizer.apply_gradients(grads, global_step=self.global_step)

    def _summary(self):
        """
        Helper that create operation for tf.Summary
        :return:
        """
        trainable_variable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        for var in trainable_variable:
            tf.summary.histogram(var.op.name, var)

        self.merged_summary_op = tf.summary.merge_all()

    def build(self):
        """
        Build the graph
        :return:
        """
        self._build_placeholder()
        self._forward()
        self._loss()
        self._optimize()
        self._summary()

    def train(self):
        """
        Train the graph
        :param lab: QTable
            QTable will decide to save or not the model depending on the previous computation
        :return:
        """
        summary_writer = tf.summary.FileWriter('logs/{}'.format("logistic_regression"), graph=self.sess.graph,
                                               flush_secs=20)
        self.sess.run(tf.global_variables_initializer())
        tf.train.global_step(self.sess, self.global_step)

        for _ in tqdm(range(self.nb_epochs)):
            self._train_epoch(summary_writer)
            self._test_epoch(mnist.validation, summary_writer)

        self._test_epoch(mnist.test, summary_writer)

        summary_writer.flush()
        summary_writer.close()

        self.sess.close()

    def _train_epoch(self, summary_writer):
        self.n_train_batches = mnist.train.images.shape[0] // self.batch_size
        for x_batch, y_batch in get_next_batch_training(train_gene, mnist.train, self.batch_size):
            _, loss, accuracy, summary_str = self.sess.run(
                [self.train_dis, self.loss, self.accuracy, self.merged_summary_op],
                feed_dict={self.x: x_batch,
                           self.y: y_batch,
                           self.is_training: True,
                           self.learning_rate_placholder: self.lr})

            summary_accuracy = tf.Summary(value=[
                tf.Summary.Value(tag="accuracy_train", simple_value=accuracy),
            ])
            summary_loss = tf.Summary(value=[
                tf.Summary.Value(tag="loss_train", simple_value=loss),
            ])
            summary_writer.add_summary(summary_accuracy,
                                       global_step=self.sess.run(self.global_step))
            summary_writer.add_summary(summary_loss,
                                       global_step=self.sess.run(self.global_step))
            summary_writer.add_summary(summary_str, global_step=self.sess.run(self.global_step))

    def _test_epoch(self, batch, summary_writer):
        name = "test" if batch == mnist.test else "val"
        generator = test_gene if name == "test" else val_gene
        step = self.sess.run(self.global_step) // self.n_train_batches
        mean_accuracy, mean_loss = 0, 0
        n_batches = len(batch.images) // self.batch_size

        for x_batch, y_batch in get_next_batch_training(generator, batch, self.batch_size):
            accuracy_itr, loss_itr = self.sess.run([self.accuracy, self.loss],
                                                   feed_dict={
                                                       self.x: x_batch,
                                                       self.y: y_batch,
                                                       self.is_training: False,
                                                       self.learning_rate_placholder: self.lr
                                                   })
            mean_accuracy += accuracy_itr
            mean_loss += loss_itr
        mean_accuracy /= n_batches
        mean_loss /= n_batches

        summary_accuracy = tf.Summary(value=[
            tf.Summary.Value(tag="mean_accuracy{}".format(name), simple_value=mean_accuracy),
        ])
        summary_loss = tf.Summary(value=[
            tf.Summary.Value(tag="summary_loss{}".format(name), simple_value=mean_loss),
        ])
        summary_writer.add_summary(summary_accuracy,
                                   global_step=step)
        summary_writer.add_summary(summary_loss,
                                   global_step=step)
