import sys
import argparse

import tensorflow as tf
import data_utils
import model

tf.logging.set_verbosity(tf.logging.DEBUG)
MODELS = ['simplecnn']


def get_simple_cnn_experiment(args):
    """
    Function for creating an experiment using the SimpleCNN model on CIFAR
    """
    train_input_fn = data_utils.get_input_fn(
        data_dir=args.data_dir,
        is_training=True,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        shuffle=True)

    val_input_fn = data_utils.get_input_fn(
        data_dir=args.data_dir,
        is_training=False,
        num_epochs=1,
        batch_size=2*args.batch_size,
        shuffle=False)

    simplecnn = model.CIFARSimpleCNNModel(
        num_classes=args.num_classes,
        scope='CIFAR{}SimpleCNN'.format(args.num_classes))

    config = tf.estimator.RunConfig(
        keep_checkpoint_max=10000,
        tf_random_seed=1234,
        save_summary_steps=50,
        save_checkpoints_secs=120)

    estimator = tf.estimator.Estimator(
        model_fn=simplecnn.get_model_fn(),
        model_dir=args.model_dir,
        config=config,
        params={'learning_rate': args.lr}
    )

    experiment = tf.contrib.learn.Experiment(
        estimator=estimator,
        train_input_fn=train_input_fn,
        eval_input_fn=val_input_fn,
        eval_metrics=None,
        train_steps=None,
        eval_steps=None,
        train_monitors=[],
        min_eval_frequency=1,
    )

    return experiment


def main(args):
    parser = argparse.ArgumentParser(
        description='Train a model on the CIFAR10/CIFAR100 dataset.')

    parser.add_argument('-m', '--model', required=True, choices=MODELS,
                        help='Select which model to train')
    parser.add_argument('-md', '--model-dir', required=True,
                        help='the directory where the model and related'
                        'files are saved')
    parser.add_argument('-dd', '--data-dir', required=True,
                        help='directory which contains the data files')
    parser.add_argument('-nc', '--num-classes', required=True, type=int,
                        help='number of classes')
    parser.add_argument('-b', '--batch-size', default=64,
                        type=int, help='the batch size')
    parser.add_argument('-e', '--num-epochs', default=15,
                        type=int, help='number of steps (minibatches)')
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='the learning rate of the model')
    args = parser.parse_args(args)

    if args.model == 'simplecnn':
        experiment = get_simple_cnn_experiment(args)
    else:
        raise NotImplementedError()

    experiment.train_and_evaluate()


if __name__ == '__main__':
    main(sys.argv[1:])
