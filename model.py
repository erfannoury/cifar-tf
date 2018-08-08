import tensorflow as tf


class CIFARSimpleCNNModel(object):
    """
    A simple model using CNN layers for the task of image classification on
    CIFAR (CIFAR10, CIFAR100) datasets.

    Parameters
    ----------
    num_classes: int
        Number of classes (10 for CIFAR10 and 100 for CIFAR100)
    scope: str
        Name scope of the model
    """

    def __init__(self, num_classes, scope):
        self.num_classes = num_classes
        self.scope = scope
        self.__debug_tensors__ = {}

    def get_model_fn(self):
        """
        Creates the model function pertaining to the `Estimator` class
        interface.

        Returns
        -------
        model_fn: callable
            The model function with the following signature:
            model_fn(features, labels, mode, params)
        """
        def model_fn(features, labels, mode, params):
            """
            Parameters
            ----------
            features: Tensor
                A batch of images of shape `(batch size, image height, image
                width, num channels)`.
            labels: Tensor
                If mode is ModeKeys.INFER, `labels=None` will be passed.
            mode: tf.estimator.ModeKeys
                Specifies if this training, evaluation, or prediction.
            params: dict
                Optional dictionary of hyperparameters. Will receive what
                is passed to Estimator in params. This allows to configure
                Estimator's for hyperparameter tuning.

            Returns
            -------
            predictions: Tensor
                Predictions of the network for input features
            loss: Tensor
                Prediction loss of the network for the given input features and
                labels
            train_op: TensorOp
                The training operation that when run in a session, will update
                model parameters, given input features and labels
            """
            if mode == tf.estimator.ModeKeys.PREDICT:
                logits, predictions = self.create_model_graph(
                    images_var=features,
                    labels_var=labels,
                    mode=mode
                )

                return tf.estimator.EstimatorSpec(
                    mode=mode,
                    predictions={'label': predictions}
                )
            else:
                predictions, loss = self.create_model_graph(
                    images_var=features,
                    labels_var=labels,
                    mode=mode
                )

                train_op = self.get_train_func(
                    loss=loss,
                    learning_rate=params['learning_rate'],
                    mode=mode
                )

                eval_metric_ops = {
                    'evalmetric/accuracy':
                        tf.contrib.metrics.streaming_accuracy(
                            predictions=predictions, labels=labels)
                }

                return tf.estimator.EstimatorSpec(
                    mode=mode,
                    predictions=predictions,
                    loss=loss,
                    train_op=train_op,
                    eval_metric_ops=eval_metric_ops)

        return model_fn

    def create_model_graph(self, images_var, labels_var, mode):
        """
        Create the main computational graph of the model

        Parameters
        ----------
        images_var: Tensor
            placeholder (or variable) for images of shape `(batch size, image
            height, image width, num channels)`
        labels_var: Tensor
            placeholder (or variable) for the class label of the image, of
            shape `(batch size, )`
        mode: tf.estimator.ModeKeys
            Run mode for creating the computational graph
        """
        with tf.variable_scope(self.scope, 'CIFARSimpleCNN'):
            kernel_conv1 = tf.get_variable(
                name='kernel_conv1',
                shape=[5, 5, 3, 64],
                dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(stddev=0.1),
                trainable=True)
            tf.summary.histogram('kernel_conv1', kernel_conv1)
            bias_conv1 = tf.get_variable(
                name='bias_conv1',
                shape=[64],
                dtype=tf.float32,
                initializer=tf.constant_initializer(5.0),
                trainable=True)
            tf.summary.histogram('bias_conv1', bias_conv1)
            conv1 = tf.nn.conv2d(
                input=images_var,
                filter=kernel_conv1,
                strides=[1, 1, 1, 1],
                padding='SAME',
                use_cudnn_on_gpu=True,
                data_format='NHWC',
                name='conv1_layer')
            conv1 = tf.nn.bias_add(
                value=conv1,
                bias=bias_conv1,
                data_format='NHWC')
            conv1 = tf.nn.relu(conv1)

            mp_conv1 = tf.nn.max_pool(
                value=conv1,
                ksize=[1, 3, 3, 1],
                strides=[1, 2, 2, 1],
                padding='SAME',
                data_format='NHWC',
                name='mp_conv1_layer')

            kernel_conv2 = tf.get_variable(
                name='kernel_conv2',
                shape=[5, 5, 64, 64],
                dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(stddev=0.1),
                trainable=True)
            tf.summary.histogram('kernel_conv2', kernel_conv2)
            bias_conv2 = tf.get_variable(
                name='bias_conv2',
                shape=[64],
                dtype=tf.float32,
                initializer=tf.constant_initializer(5.0),
                trainable=True)
            tf.summary.histogram('bias_conv2', bias_conv2)
            conv2 = tf.nn.conv2d(
                input=mp_conv1,
                filter=kernel_conv2,
                strides=[1, 1, 1, 1],
                padding='SAME',
                use_cudnn_on_gpu=True,
                data_format='NHWC',
                name='conv2_layer')
            conv2 = tf.nn.bias_add(
                value=conv2,
                bias=bias_conv2,
                data_format='NHWC')
            conv2 = tf.nn.relu(conv2)

            # mp_conv2 -> (batch size, 64, 8, 8)
            mp_conv2 = tf.nn.max_pool(
                value=conv2,
                ksize=[1, 3, 3, 1],
                strides=[1, 2, 2, 1],
                padding='SAME',
                data_format='NHWC',
                name='mp_conv2_layer')

            W_fc = tf.get_variable(
                name='W_fc',
                shape=[8 * 8 * 64, 512],
                dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(stddev=0.1),
                trainable=True)
            tf.summary.histogram('W_fc', W_fc)
            bias_fc = tf.get_variable(
                name='bias_fc',
                shape=[512],
                dtype=tf.float32,
                initializer=tf.constant_initializer(5.0),
                trainable=True)
            tf.summary.histogram('bias_fc', bias_fc)
            mp_conv2_rshp = tf.reshape(
                tensor=mp_conv2,
                shape=[tf.shape(mp_conv2)[0], -1])
            fc = tf.nn.xw_plus_b(
                x=mp_conv2_rshp,
                weights=W_fc,
                biases=bias_fc,
                name='fc_layer')
            fc = tf.nn.relu(fc)

            W_logit = tf.get_variable(
                name='W_logit',
                shape=[512, self.num_classes],
                dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(stddev=0.1),
                trainable=True)
            tf.summary.histogram('W_logit', W_logit)

            logits = tf.matmul(fc, W_logit)

            predictions = tf.argmax(logits, axis=-1)

            if mode != tf.estimator.ModeKeys.PREDICT:
                with tf.variable_scope('loss'):
                    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=labels_var,
                        logits=logits
                    )
                    loss = tf.reduce_mean(losses, name='loss')
                tf.summary.scalar('loss', loss)
                with tf.variable_scope('accuracy'):
                    accuracy = tf.contrib.metrics.accuracy(
                        predictions, labels_var)
                tf.summary.scalar('accuracy', accuracy)

                return predictions, loss
            else:
                return logits, predictions

    def get_train_func(self, loss, learning_rate, mode):
        """
        Create the training function for the model.

        Parameters
        ----------
        loss: Tensor
            Tensor variable for the network loss
        learning_rate: float
            Learning rate value
        mode: tf.contrib.learn.ModeKeys
                Specifies if this training, evaluation, or prediction.

        Returns
        -------
        train_op
        """
        if mode != tf.estimator.ModeKeys.TRAIN or loss is None:
            return None

        global_step = tf.train.get_or_create_global_step()

        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=global_step,
            learning_rate=learning_rate,
            optimizer='Adam',
            summaries=['gradients'])

        return train_op
