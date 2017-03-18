import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import layers
from tqdm import tqdm

epsilon = 1e-4

mnist = input_data.read_data_sets('MNIST_data', one_hot=True, validation_size=5000)


class NN_with_l2:
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

        # self.batch_norm = layers.batch_norm if args.batch_norm == True else None
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

    def _feed_dict(self, batch, batch_size):
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
        batch_x, batch_y = batch.next_batch(batch_size)
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

        # Hidden layer
        w1 = tf.get_variable("W1", initializer=tf.contrib.layers.xavier_initializer(),
                             shape=(self.input_size, self.input_size / 2))
        b1 = tf.get_variable("b1", initializer=tf.contrib.layers.xavier_initializer(), shape=self.input_size / 2)
        self.h1 = tf.nn.relu(tf.matmul(self.x, w1) + b1)

        # Logit
        w2 = tf.get_variable("W2", initializer=tf.contrib.layers.xavier_initializer(),
                             shape=(self.input_size / 2, self.nb_targets))
        b2 = tf.get_variable("b2", initializer=tf.contrib.layers.xavier_initializer(), shape=self.nb_targets)
        self.logits = tf.matmul(self.h1, w2) + b2

        # h1 = layers.fully_connected(self.x, num_outputs=self.input_size // 2, activation_fn=None)
        # self.logits = layers.fully_connected(h1, num_outputs=self.nb_targets, activation_fn=None)

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
        self.loss = cross_entropy  # + tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        self.loss += 0.001 * tf.add_n([  # Somme sur toutes les variables
                                        tf.nn.l2_loss(v) for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)  #
                                        if 'bias' not in v.name])

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
        for itr in range(self.n_train_batches):
            _, loss, accuracy = self.sess.run([self.train_dis, self.loss, self.accuracy],
                                              feed_dict=self._feed_dict(mnist.train, self.batch_size))
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

    def _test_epoch(self, batch, summary_writer):
        name = "test" if batch == mnist.test else "val"

        step = self.sess.run(self.global_step) // self.n_train_batches

        mean_accuracy, mean_loss = self.sess.run([self.accuracy, self.loss],
                                                 feed_dict=self._feed_dict(batch, len(batch.images)))
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
