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
                    mode=mode)

                return tf.estimator.EstimatorSpec(
                    mode=mode,
                    predictions={'label': predictions})
            else:
                predictions, loss = self.create_model_graph(
                    images_var=features,
                    labels_var=labels,
                    mode=mode)

                train_op = self.get_train_func(
                    loss=loss,
                    mode=mode,
                    params=params)

                eval_metric_ops = {
                    'evalmetric/accuracy':
                        tf.metrics.accuracy(
                            predictions=predictions, labels=labels)}

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
            conv1 = tf.layers.conv2d(
                images_var, 64, kernel_size=5,
                padding='same', data_format='channels_last',  # NHWC
                use_bias=True,
                activation=tf.nn.relu,
                kernel_initializer=tf.initializers.variance_scaling(
                    scale=2.0, mode='fan_avg'),
                trainable=True)

            mp_conv1 = tf.layers.max_pooling2d(
                conv1, 3, strides=2, padding='same')

            conv2 = tf.layers.conv2d(
                mp_conv1, filters=64, kernel_size=5, padding='same',
                use_bias=True,
                activation=tf.nn.relu,
                kernel_initializer=tf.initializers.variance_scaling(
                    scale=2.0, mode='fan_avg'),
                trainable=True)

            # mp_conv2 -> (batch size, 64, 8, 8)
            mp_conv2 = tf.layers.max_pooling2d(conv2, 3, 2, padding='same')
            mp_conv2 = tf.layers.flatten(mp_conv2)

            fc = tf.layers.dense(
                mp_conv2, 512, activation=tf.nn.relu,
                use_bias=True,
                kernel_initializer=tf.initializers.truncated_normal(
                    stddev=0.1),
                trainable=True)

            logits = tf.layers.dense(
                fc, self.num_classes,
                use_bias=False, trainable=True,
                kernel_initializer=tf.initializers.truncated_normal(
                    stddev=0.1))
            # logits -> (batch size, num_classes)

            predictions = tf.argmax(logits, axis=1)

            if mode != tf.estimator.ModeKeys.PREDICT:
                loss = tf.losses.sparse_softmax_cross_entropy(
                    labels=labels_var,
                    logits=logits,
                    reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)
                tf.summary.scalar('loss', loss)
                with tf.variable_scope('accuracy'):
                    _, accuracy = tf.metrics.accuracy(
                        predictions, labels_var)
                tf.summary.scalar('accuracy', accuracy)

                return predictions, loss
            else:
                return logits, predictions

    def get_train_func(self, loss, mode, params):
        """
        Create the training function for the model.

        Parameters
        ----------
        loss: Tensor
            Tensor variable for the network loss
        mode: tf.estimator.ModeKeys
                Specifies if this training, evaluation, or prediction.
        params: dict
            A dictionary of parameters for the optimizer

        Returns
        -------
        train_op
        """
        if mode != tf.estimator.ModeKeys.TRAIN or loss is None:
            return None

        global_step = tf.train.get_or_create_global_step()

        learning_rate = params['learning_rate']
        weight_decay = params['weight_decay']
        opt = tf.contrib.opt.AdamWOptimizer(
            weight_decay=weight_decay,
            learning_rate=learning_rate)

        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=global_step,
            learning_rate=None,
            optimizer=opt,
            summaries=['gradients', 'gradient_norm'])

        return train_op
