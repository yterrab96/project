import math                 # python libraries
import numpy as np          #
import pathlib              #
import sys                  #

import tensorflow as tf     # tensorflow

import models               # our libraries
import datasets             #

# ---------------------------------------------------------------------------- #

# Parameters
f                  = 20
batch_size         = 50
learning_rate      = 0.05
activation_func    = tf.nn.relu
max_train_epoch    = 100000
max_train_accur    = 0.97
load_parameters    = True
parameters_path    = pathlib.Path("model.npy")

# ---------------------------------------------------------------------------- #

# Dataset instantiation
dataset   = datasets.load_mnist() # handwritten digit database
train_set = dataset.cut(0, 50000, 50000).shuffle().cut(0, 50000, batch_size) # 1000 batches of size 50
test_set  = dataset.cut(50000, 60000, 10000)                                 # 1 batch of size 10 000

# Model instantiator
builder_opt  = tf.train.AdagradOptimizer(learning_rate)
builder_dims = [784, 100, 10] # 3 layer neural network:
                              # input layer  : 784 neurons (1 image = 28*28 pixels)
                              # hidden layer : 100 neurons
                              # output layer :  10 neurons (digits 0-9)
def builder(inputs=None):
    return models.dense_classifier(builder_dims, inputs=inputs, act_fn=activation_func, optimizer=builder_opt, epoch=True)

# Model instantiation
graph = tf.Graph()
with graph.as_default():
   model = builder()

# Training
with graph.as_default():
  sess = tf.Session(graph=graph)
  with sess.as_default():
    sess.run(tf.global_variables_initializer())
    model.init()

    # Loading
    if load_parameters and parameters_path.exists():
      sys.stdout.write("Load model...")
      sys.stdout.flush()
      try:
        model.write(np.load(parameters_path))
        print(" done.")
      except Exception as err:
        print(" fail.")
        raise
    else:
      print("New model... done.")

    # Testing + training
    try:
      l = len(train_set)
      print(type(train_set))
      print(type(test_set))
      while True:
        epoch = model.epoch()
        sys.stdout.write("Epoch " + str(epoch) + ": accuracy = ")
        sys.stdout.flush()
        acc = model.eval(*test_set.get())[0]
        print(str(acc))
        if acc >= max_train_accur:
          break
        if epoch >= max_train_epoch:
          break
        c = 0
        p = -1
        for x, y in train_set:
          n = int(c / l * 100.)
          if n != p:
            p = n            
            sys.stdout.write("\rTraining... " + str(p) + "%")
            sys.stdout.flush()
          grad = model.backprop(x,y)
          model.update(grad)
          c += 1
        print("\rTraining... 100%")
    except KeyboardInterrupt:
      print("\rTraining... interrupted.")

    # Saving (if training done)
    #if epoch > 0:
    #  sys.stdout.write("Saving...")
    #  sys.stdout.flush()
    #  try:
    #    np.save(parameters_path, model.read())
    #    print(" done.")
    #  except Exception as err:
    #    print(" fail (" + type(err).__name__ + ")")

