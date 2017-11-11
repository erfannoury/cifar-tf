import sys
import os
from six.moves import urllib
import tarfile
import pickle
import numpy as np
import tensorflow as tf

CHANNELS = 3
HEIGHT = 32
WIDTH = 32


def maybe_download_and_extract():
    DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    """Download and extract the tarball from Alex's website."""
    dest_directory = 'data'
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (
                filename,
                float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    extracted_dir_path = os.path.join(dest_directory, 'cifar-10-batches-bin')
    if not os.path.exists(extracted_dir_path):
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def get_cifar_generator(data_dir, is_training):
    """
    Gets a generator function that will be used for the input_fn

    Parameters
    ----------
    data_dir: str
        Path to where the cifar data resides
    is_training: bool
        Whether to read the training or the test portion of the data

    Returns
    -------
    generator_fn: callable
        A generator function that will yield feature dict and label
    """
    if is_training:
        data = []
        labels = []
        for i in range(1, 6):
            train_batch = pickle.load(
                open(os.path.join(data_dir, 'data_batch_' + str(i)), 'rb'),
                encoding='bytes')
            data.append(train_batch[b'data'])
            labels.append(train_batch[b'labels'])
        data = np.vstack(tuple(data)).astype(np.float32)
        labels = np.hstack(tuple(labels)).astype(np.int64)
        print('Train data.shape: {}, labels.shape: {}'.format(
            data.shape, labels.shape))
    else:
        test_batch = pickle.load(
            open(os.path.join(data_dir, 'test_batch'), 'rb'),
            encoding='bytes')
        data = test_batch[b'data'].astype(np.float32)
        labels = np.asarray(test_batch[b'labels'], dtype=np.int64)
        print('Test data.shape: {}, labels.shape: {}'.format(
            data.shape, labels.shape))

    data = np.reshape(data, (-1, CHANNELS, HEIGHT, WIDTH))
    data = np.transpose(data, axes=(0, 2, 3, 1))

    def generator():
        for i in range(data.shape[0]):
            yield (data[i, :], labels[i])

    return generator


def get_input_fn(data_dir, is_training, num_epochs, batch_size, shuffle):
    """
    This will return input_fn from which batches of data can be obtained.

    Parameters
    ----------
    data_dir: str
        Path to where the cifar data resides
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
    gen = get_cifar_generator(data_dir, is_training)
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
    maybe_download_and_extract()
    get_cifar_generator('data/cifar-10-batches-py', True)
    get_cifar_generator('data/cifar-10-batches-py', False)

    train_input_fn = get_input_fn(
        data_dir='data/cifar-10-batches-py',
        is_training=True,
        num_epochs=1,
        batch_size=13,
        shuffle=True)

    val_input_fn = get_input_fn(
        data_dir='data/cifar-10-batches-py',
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
