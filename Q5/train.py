import sys
import argparse

import numpy as np
import tensorflow as tf

from load_mnist import MNIST
import data_utils
import model


tf.logging.set_verbosity(tf.logging.INFO)
MODELS = ['simpleNN']

def get_simple_nn_experiment(args):
    """
    Function for creating an experiment using the SimpleNN model on MNIST
    """
    train_input_fn = data_utils.get_input_fn(
        data_dir=args.data_dir,
        is_training=True,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        shuffle=True,
        normalize=args.normalize)

    val_input_fn = data_utils.get_input_fn(
        data_dir=args.data_dir,
        is_training=False,
        num_epochs=1,
        batch_size=2*args.batch_size,
        shuffle=False,
        normalize=args.normalize)

    simplecnn = model.SimpleMnistModel(
        num_classes=args.num_classes,
        scope='SimpleMnist')

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
        description='Train a model on the MNIST dataset.')

    parser.add_argument('-m', '--model', required=True, choices=MODELS,
                        help='Select which model to train')
    parser.add_argument('-md', '--model-dir', required=True,
                        help='the directory where the model and related'
                        'files are saved')
    parser.add_argument('-dd', '--data-dir', default='./',
                        help='directory which contains the data files')
    parser.add_argument('-nc', '--num-classes', default=10, type=int,
                        help='number of classes')
    parser.add_argument('-b', '--batch-size', default=60,
                        type=int, help='the batch size')
    parser.add_argument('-e', '--num-epochs', default=20,
                        type=int, help='number of steps (minibatches)')
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='the learning rate of the model')
    parser.add_argument('-n', '--normalize', default=True,
                        type=bool, help='normalize images' )
    args = parser.parse_args(args)

    if args.model == 'simpleNN':
        experiment = get_simple_nn_experiment(args)
    else:
        raise NotImplementedError()

    experiment.train_and_evaluate()

if __name__ == '__main__':
    main(sys.argv[1:])