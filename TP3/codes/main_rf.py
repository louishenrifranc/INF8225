from QTable import QTable
import tensorflow as tf
import argparse

max_size_network = 4000000
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=100, help='Size of a batch')
    parser.add_argument('--activation', default=tf.nn.relu, help='Activation function')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--l1_reg', type=float, default=0.00, help='L1 regularization')
    parser.add_argument('--l2_reg', type=float, default=0.00, help='L2 regularization')
    parser.add_argument('--batch_norm', type=bool, default=True, help='Batch normalization')
    parser.add_argument('--patience', type=int, default=10, help='Patience')
    parser.add_argument('--dropout_p', type=float, default=0.7, help='Dropout probability')
    parser.add_argument('--input_size', type=int, default=28, help='Input size')
    parser.add_argument('--nb_targets', type=int, default=10, help='Nb targets')
    parser.add_argument('--nb_iter', type=int, default=20, help='Number of epochs')
    parser.add_argument('--max_size_network', type=int, default=2000000, help='Maximum size (in bytes) for a NN')
    parser.add_argument('--optimizer',
                        default=tf.train.AdamOptimizer,
                        help='Size of a batch')
    parser.add_argument('--experiment_name')
    parser.add_argument('--layers')
    args = parser.parse_args()

    max_size_network = args.max_size_network
    q = QTable(M=1000)
    q.q_learning(args)
