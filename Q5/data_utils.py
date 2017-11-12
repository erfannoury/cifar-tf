import sys
import os
from six.moves import urllib
import tarfile
import pickle
import numpy as np
import tensorflow as tf

from load_mnist import MNIST

CHANNELS = 1
HEIGHT = 28
WIDTH = 28

def get_data_generator(data_dir, is_training, normalize=True):
    """
    Gets a generator function that will be used for the input_fn

    Parameters
    ----------
    data_dir: str
        Path to where the mnist data resides
    is_training: bool   
        Whether to read the training or the test portion of the data

    Returns
    -------
    generator_fn: callable
        A generator function that will yield feature dict and label
    """
    if is_training:
        data, labels = MNIST(path=data_dir,return_type="numpy",mode="vanilla").load_training()
        print('Train data.shape: {}, labels.shape: {}'.format(
            data.shape, labels.shape))
    else:
        data, labels = MNIST(path=data_dir,return_type="numpy",mode="vanilla").load_testing()
        print('Test data.shape: {}, labels.shape: {}'.format(
            data.shape, labels.shape))

    if normalize:
        print("normalized")
        data = data.astype(np.float32)
        data = np.multiply(data, 1.0 / 255.0)
        tf.nn.l2_normalize(data, dim=0)
    data = np.reshape(data, (-1, CHANNELS, HEIGHT, WIDTH))
    data = np.transpose(data, axes=(0, 2, 3, 1))

    def generator():
        for i in range(data.shape[0]):
            yield (data[i, :], labels[i])

    return generator

def get_input_fn(data_dir, is_training, num_epochs, batch_size, shuffle, normalize=True):
    """
    This will return input_fn from which batches of data can be obtained.

    Parameters
    ----------
    data_dir: str
        Path to where the mnist data resides
    is_training: bool
        Whether to read the training or the test portion of the data
    num_epochs: int
        Number of data epochs
    batch_size: int
        Batch size
    shuffle: bool
        Whether to shuffle the data or not

    Returns
    -------
    input_fn: callable
        The input function which returns a batch of images and labels
        tensors, of shape (batch size, CHANNELS, HEIGTH, WIDTH) and
        (batch size), respectively.
    """
    gen = get_data_generator(data_dir, is_training, normalize)
    ds = tf.data.Dataset.from_generator(
        generator=gen,
        output_types=(tf.float32, tf.int64),
        output_shapes=(tf.TensorShape([HEIGHT, WIDTH, CHANNELS]),
                       tf.TensorShape([]))
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=2000, reshuffle_each_iteration=True)
    ds = ds.repeat(count=num_epochs)
    ds = ds.batch(batch_size=batch_size)

    def input_fn():
        ds_iter = ds.make_one_shot_iterator()
        images, labels = ds_iter.get_next()
        return images, labels

    return input_fn


if __name__ == '__main__':

    train_input_fn = get_input_fn(
        data_dir='./',
        is_training=True,
        num_epochs=1,
        batch_size=13,
        shuffle=True)

    val_input_fn = get_input_fn(
        data_dir='./',
        is_training=False,
        num_epochs=1,
        batch_size=15,
        shuffle=False)

    train_im, train_lbl = train_input_fn()
    val_im, val_lbl = val_input_fn()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        tf.train.start_queue_runners(sess)
        im, lbl = sess.run((train_im, train_lbl))
        print(im.shape, lbl.shape)

        im, lbl = sess.run((val_im, val_lbl))
        print(im.shape, lbl.shape)