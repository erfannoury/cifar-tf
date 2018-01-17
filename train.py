import sys
import os
import json
import argparse

import tensorflow as tf
import data_utils
import model

tf.logging.set_verbosity(tf.logging.INFO)
MODELS = ['simplecnn']
DIST_TYPES = ['master', 'ps', 'worker']


def get_simple_cnn_experiment(args):
    """
    Function for creating an experiment using the SimpleCNN model on CIFAR
    """
    shard_index = 0
    if args.distributed and args.dist_type == 'worker':
        shard_index = args.worker_index + 1

    train_input_fn = data_utils.get_input_fn(
        data_dir=args.data_dir,
        is_training=True,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        shuffle=True,
        num_shards=(args.worker_count + 1 if args.distributed else 1),
        shard_index=shard_index)

    val_input_fn = data_utils.get_input_fn(
        data_dir=args.data_dir,
        is_training=False,
        num_epochs=1,
        batch_size=2*args.batch_size,
        shuffle=False)

    simplecnn = model.CIFARSimpleCNNModel(
        num_classes=args.num_classes,
        scope='CIFAR{}SimpleCNN'.format(args.num_classes))

    if args.distributed:
        dist_config = {}
        dist_config['cluster'] = {
            'master': ['127.0.0.1:{}'.format(args.dist_start_port + 1)],
            'ps': ['127.0.0.1:{}'.format(args.dist_start_port - i)
                   for i in range(args.ps_count)],
            'worker': ['127.0.0.1:{}'.format(args.dist_start_port + 2 + i)
                       for i in range(args.worker_count)]
        }
        index = 0
        if args.dist_type == 'ps':
            index = args.ps_index
        elif args.dist_type == 'worker':
            index = args.worker_index
        dist_config['task'] = {
            'type': args.dist_type,
            'index': index
        }
        dist_config['environment'] = 'cloud'
        os.environ['TF_CONFIG'] = json.dumps(dist_config)

    config = tf.contrib.learn.RunConfig(
        log_device_placement=False,
        gpu_memory_fraction=0.98,
        tf_random_seed=1234,
        save_summary_steps=50,
        save_checkpoints_secs=300,
        keep_checkpoint_max=10000,
        keep_checkpoint_every_n_hours=10000,
        log_step_count_steps=10,
    )

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
        min_eval_frequency=1000,
        eval_delay_secs=240
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
    parser.add_argument('--distributed', action='store_true',
                        help='Whether to use distributed training')
    parser.add_argument('--dist-type', choices=DIST_TYPES,
                        help='Type of the process for distributed training')
    parser.add_argument('--ps-count', type=int, default=2,
                        help='Number of distributed parameter servers')
    parser.add_argument('--worker-count', type=int, default=2,
                        help='Number of distributed training workers '
                        'not including the master node')
    parser.add_argument('--dist-start-port', type=int, default=5000,
                        help='Starting port number for distributed training.\n'
                        'Parameter server will start on this port, master '
                        'node will start on the next port, and worker nodes '
                        'will start on the port numbers after the master')
    parser.add_argument('--ps-index', type=int, default=0,
                        help='Index of the parameter server for '
                        'distributed training')
    parser.add_argument('--worker-index', type=int, default=0,
                        help='Index of the worker for distributed training')
    args = parser.parse_args(args)

    if args.model == 'simplecnn':
        experiment = get_simple_cnn_experiment(args)
    else:
        raise NotImplementedError()

    if args.distributed:
        if args.dist_type == 'master':
            experiment.train_and_evaluate()
        elif args.dist_type == 'ps':
            experiment.run_std_server()
        else:
            experiment.train()
    else:
        experiment.train_and_evaluate()


if __name__ == '__main__':
    main(sys.argv[1:])
