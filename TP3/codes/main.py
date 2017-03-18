import tensorflow as tf
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64, help='Size of a batch')
    parser.add_argument('--activation', default=tf.nn.relu, help='Activation function')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--l1_reg', type=float, default=0.00, help='L1 regularization')
    parser.add_argument('--l2_reg', type=float, default=0.00, help='L2 regularization')
    parser.add_argument('--batch_norm', type=bool, default=False, help='Batch normalization')
    parser.add_argument('--patience', type=int, default=10, help='Patience')
    parser.add_argument('--dropout_p', type=float, default=0.7, help='Dropout probability')
    parser.add_argument('--input_size', type=int, default=784, help='Input size')
    parser.add_argument('--nb_targets', type=int, default=10, help='Nb targets')
    parser.add_argument('--nb_iter', type=int, default=100, help='Number of epochs')
    parser.add_argument('--optimizer',
                        default=tf.train.AdamOptimizer,
                        help='Size of a batch')
    parser.add_argument('--experiment', type=str, default="NN")
    args = parser.parse_args()

    if args.experiment == "LR":
        from logistic_regression import LogisticRegression as LR

        lr = LR(args)
    elif args.experiment == "NM":
        from noisy_model import NN

        lr = NN(args)
    elif args.experiment == "BM":
        from best_model import NN

        lr = NN(args)
    elif args.experiment == "NN":
        from one_hidden_layer import NN

        lr = NN(args)
    elif args.experiment == "L2":
        from l2_regularization import NN_with_l2

        lr = NN_with_l2(args)

    lr.build()
    lr.train()
