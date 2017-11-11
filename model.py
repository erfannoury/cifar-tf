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
                A batch of images of shape `(batch size, num channels, image
                height, image width)`.
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
                hpparams=params,
                mode=mode
            )

            eval_metric_ops = {
                'evalmetric/accuracy': tf.contrib.metrics.streaming_accuracy(
                    predictions=predictions, labels=labels),
                'evalmetric/auroc': tf.contrib.metrics.streaming_auc(
                    predictions=predictions, labels=labels),
                'evalmetric/recall': tf.contrib.metrics.streaming_recall(
                    predictions=predictions, labels=labels),
                'evalmetric/precision': tf.contrib.metrics.streaming_precision(
                    predictions=predictions, labels=labels),
                'evalmetric/tp': tf.contrib.metrics.streaming_true_positives(
                    predictions=predictions, labels=labels),
                'evalmetric/fn': tf.contrib.metrics.streaming_false_negatives(
                    predictions=predictions, labels=labels),
                'evalmetric/fp': tf.contrib.metrics.streaming_false_positives(
                    predictions=predictions, labels=labels),
                'evalmetric/tn': tf.contrib.metrics.streaming_true_negatives(
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
            placeholder (or variable) for images of shape `(batch size,
            num channels, image height, image width)`
        labels_var: Tensor
            placeholder (or variable) for the class label of the image, of
            shape `(batch size, )`
        mode: tf.contrib.learn.ModeKeys
            Run mode for creating the computational graph
        """
        with tf.variable_scope(self.scope, 'CIFARSimpleCNN'):
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

    def get_train_func(self, loss, hpparams, mode):
        """
        Create the training function for the model.

        Parameters
        ----------
        loss: Tensor
            Tensor variable for the network loss
        hpparams: dict
            A dictionary of hyperparameters passed to the Estimator
        mode: tf.contrib.learn.ModeKeys
                Specifies if this training, evaluation, or prediction.

        Returns
        -------
        train_op
        """
        if mode != tf.contrib.learn.ModeKeys.TRAIN or loss is None:
            return None

        global_step = tf.train.get_or_create_global_step()

        if hpparams['use_decaying_lr']:
            lr = tf.train.exponential_decay(
                learning_rate=hpparams['learning_rate'],
                global_step=global_step,
                decay_steps=hpparams['lr_decay_steps'],
                decay_rate=hpparams['lr_decay_rate'],
                staircase=hpparams['lr_decay_staircase'],
                name='exponential_decay_lr'
            )
            tf.summary.scalar('Exponential Decay LR', lr)
        else:
            lr = hpparams['learning_rate']

        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=global_step,
            learning_rate=lr,
            optimizer='Adam',
            clip_gradients=hpparams['gradient_clip'],
            summaries=['gradients', 'gradient_norm'],
            gradient_noise_scale=0.0
        )

        return train_op
