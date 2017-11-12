import numpy as np
from load_mnist import MNIST

train_images, train_labels=MNIST(path="./",return_type="numpy",mode="vanilla").load_training()
test_images, test_labels=MNIST(path="./",return_type="numpy",mode="vanilla").load_testing()

for data in [test_images,train_images]:
	print("shape",data.shape)
	print("mean",data.mean())
	print("max",data.max())
	print("min",data.min())