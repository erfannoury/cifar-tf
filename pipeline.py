import sys
import os

import numpy as np
import tensorflow as tf


def get_cifar10_dataset(batch_size, num_epochs=1, num_shards=1, shard_index=0,
                        shuffle=False):
    cifar = tf.keras.datasets.cifar10

    (x_train, y_train), (x_test, y_test) = cifar.load_data()
    x_train = (x_train * 1.0) / 255.
    x_test = (x_test * 1.0) / 255.
    y_train = np.squeeze(y_train)
    y_test = np.squeeze(y_test)

    if shuffle:
        X = x_train
        Y = y_train
        shuffle_indices = np.arange(X.shape[0])
        np.random.shuffle(shuffle_indices)
        X = X[shuffle_indices]
        Y = Y[shuffle_indices]
    else:
        X = x_test
        Y = y_test

    def ds_fn():
        def gen():
            for x, y in zip(X, Y):
                yield (x, y)
        ds = tf.data.Dataset.from_generator(
            gen, output_types=(tf.float32, tf.int32),
            output_shapes=(tf.TensorShape([32, 32, 3]), tf.TensorShape([])))
        ds = ds.shard(num_shards, shard_index)
        if shuffle:
            ds = ds.shuffle(buffer_size=128)
            ds = ds.map(lambda i, l: (tf.image.random_flip_left_right(i), l),
                        num_parallel_calls=16)
        ds = ds.batch(batch_size=batch_size, drop_remainder=False)
        ds = ds.repeat(num_epochs)

        feat, label = ds.make_one_shot_iterator().get_next()
        return feat, label
    return ds_fn
