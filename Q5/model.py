import tensorflow as tf


class SimpleMnistModel(object):
    """
    A simple model using NN layers for the task of image classification on
    MNIST dataset.

    Parameters
    ----------
    num_classes: int
        Number of classes
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
            mode: tf.contrib.learn.ModeKeys
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
                'evalmetric/accuracy': tf.contrib.metrics.streaming_accuracy(
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
        mode: tf.contrib.learn.ModeKeys
            Run mode for creating the computational graph
        """
        with tf.variable_scope(self.scope, 'CIFARSimpleCNN'):

            W_fc = tf.get_variable(
                name='W_fc',
                shape=[28 * 28, 10],
                dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(stddev=0.1),
                trainable=True)
            tf.summary.histogram('W_fc', W_fc)
            bias_fc = tf.get_variable(
                name='bias_fc',
                shape=[10],
                dtype=tf.float32,
                initializer=tf.constant_initializer(5.0),
                trainable=True)
            tf.summary.histogram('bias_fc', bias_fc)
            images_rshp = tf.reshape(
                tensor=images_var,
                shape=[tf.shape(images_var)[0], -1])
            fc = tf.nn.xw_plus_b(
                x=images_rshp,
                weights=W_fc,
                biases=bias_fc,
                name='fc_layer')
            fc = tf.nn.sigmoid(fc)

            W_logit = tf.get_variable(
                name='W_logit',
                shape=[10, self.num_classes],
                dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(stddev=0.1),
                trainable=True)
            tf.summary.histogram('W_logit', W_logit)

            logits = tf.matmul(fc, W_logit)

            predictions = tf.argmax(logits, axis=-1)

            if mode != tf.contrib.learn.ModeKeys.INFER:
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
        if mode != tf.contrib.learn.ModeKeys.TRAIN or loss is None:
            return None

        global_step = tf.train.get_or_create_global_step()

        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=global_step,
            learning_rate=learning_rate,
            optimizer='SGD',
            summaries=['gradients'])

        return train_op
