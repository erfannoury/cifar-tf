# cifar-tf
A simple model for image classification on the CIFAR datasets, demonstrating TF's new APIs in TF 1.12 (`tf.data` and `tf.estimator`).

Option for distributed training is also added.

## Training

### Training on a single node
To train normally using a single GPU, you can use the following command

```bash
$ python train.py -m simplecnn -md model/simplecnn -nc 10 -e 100
```

### Asynchronous distributed training
This is a data parallel distributed training, where the model is replicated in a number of workers and each worker is trained on its copy of the data and the latest weights that it has obtained from the parrameter server(s). After doing a backprop, each worker sends the latest weights to the parameter server(s).

To start distributed training, first run the parameter server.
```bash
$ CUDA_VISIBLE_DEVICES='' python train.py -m simplecnn -md model/distsimplecnn -nc 10 -e 100 --distributed --dist-type ps --ps-count 1 --worker-count 2 --dist-start-port 7000 --ps-index 0
```

Afterwards, start the master node. This node will train the model.
```bash
$ CUDA_VISIBLE_DEVICES=0 python train.py -m simplecnn -md model/distsimplecnn -nc 10 -e 100 --distributed --dist-type master --ps-count 1 --worker-count 2 --dist-start-port 7000
```

Then start the two worker nodes that will only train the model.
```bash
$ CUDA_VISIBLE_DEVICES=1 python train.py -m simplecnn -md model/distsimplecnn -nc 10 -e 100 --distributed --dist-type worker --ps-count 1 --worker-count 2 --dist-start-port 7000 --worker-index 0

$ CUDA_VISIBLE_DEVICES=2 python train.py -m simplecnn -md model/distsimplecnn -nc 10 -e 100 --distributed --dist-type worker --ps-count 1 --worker-count 2 --dist-start-port 7000 --worker-index 1
```

Finally, start the evaluator. This node is not a distributed node, and its job is to continuously evaluate the latest model checkpoint on the validation data.
```bash
$ CUDA_VISIBLE_DEVICES=3 python train.py -m simplecnn -md model/distsimplecnn -nc 10 -e 100 --distributed --dist-type evaluator --ps-count 1 --worker-count 2 --dist-start-port 7000
```
