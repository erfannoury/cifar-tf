import sys
import os
import json
import argparse

import tensorflow as tf
import pipeline
import models

tf.logging.set_verbosity(tf.logging.INFO)
MODELS = ['simplecnn']
DIST_TYPES = ['master', 'ps', 'worker', 'evaluator']


def main(args):
    parser = argparse.ArgumentParser(
        description='Train a model on the CIFAR10/CIFAR100 dataset.')

    parser.add_argument('-m', '--model', required=True, choices=MODELS,
                        help='Select which model to train')
    parser.add_argument('-md', '--model-dir', required=True,
                        help='the directory where the model and related'
                        'files are saved')
    parser.add_argument('-nc', '--num-classes', required=True, type=int,
                        help='number of classes')
    parser.add_argument('-b', '--batch-size', default=64,
                        type=int, help='the batch size')
    parser.add_argument('-e', '--num-epochs', default=15,
                        type=int, help='number of steps (minibatches)')
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='the learning rate of the model')
    parser.add_argument('--weight-decay', type=float, default=0.0,
                        help='Weight decay coefficient for parameters')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate model on validation data')
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

    shard_index = 0
    if args.distributed and args.dist_type == 'worker':
        shard_index = args.worker_index + 1

    train_input_fn = pipeline.get_cifar10_dataset(
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        num_shards=(args.worker_count + 1 if args.distributed else 1),
        shard_index=shard_index,
        shuffle=True)

    val_input_fn = pipeline.get_cifar10_dataset(
        batch_size=args.batch_size,
        num_epochs=1,
        shuffle=True)

    if args.model == 'simplecnn':
        model = models.CIFARSimpleCNNModel(
            num_classes=args.num_classes,
            scope='CIFAR{}SimpleCNN'.format(args.num_classes))
    else:
        raise NotImplementedError()

    if args.distributed and args.dist_type != 'evaluator':
        dist_config = {}
        dist_config['cluster'] = {
            'master': ['127.0.0.1:{}'.format(args.dist_start_port + 1)],
            'ps': ['127.0.0.1:{}'.format(args.dist_start_port - i)
                   for i in range(args.ps_count)],
            'worker': ['127.0.0.1:{}'.format(args.dist_start_port + 2 + i)
                       for i in range(args.worker_count)]}
        index = 0
        if args.dist_type == 'ps':
            index = args.ps_index
        elif args.dist_type == 'worker':
            index = args.worker_index
        dist_config['task'] = {
            'type': args.dist_type,
            'index': index}
        dist_config['environment'] = 'cloud'
        os.environ['TF_CONFIG'] = json.dumps(dist_config)

    config = tf.estimator.RunConfig(
        model_dir=args.model_dir,
        tf_random_seed=1234,
        save_summary_steps=50,
        save_checkpoints_steps=1000,
        keep_checkpoint_max=10000,
        log_step_count_steps=25)

    estimator = tf.estimator.Estimator(
        model_fn=model.get_model_fn(),
        config=config,
        params={
            'learning_rate': args.lr,
            'weight_decay': args.weight_decay})

    train_spec = tf.estimator.TrainSpec(
        input_fn=train_input_fn,
        max_steps=None)

    eval_spec = tf.estimator.EvalSpec(
        input_fn=val_input_fn,
        steps=None,
        name='cifar10',
        start_delay_secs=600,
        throttle_secs=300)

    if args.evaluate:
        estimator.evaluate(input_fn=val_input_fn)
    else:
        tf.estimator.train_and_evaluate(
            estimator, train_spec, eval_spec)


if __name__ == '__main__':
    main(sys.argv[1:])
