# synchronous parameter server

import numpy as np
import sys
import socket
import time

import tensorflow as tf

import datasets
import models
import protocol as prot

def aggregate(grads):
    return np.sum(grads, axis=0)

# ------------------------------------------------------------------------- #

# Parameters
#worker_HOSTS     = [('lpdquad.epfl.ch', 5000)]
#worker_HOSTS     = [('lpdquad.epfl.ch', 5000), ('lpdquad.epfl.ch', 6000)]
worker_HOSTS     = [('lpdquad.epfl.ch', 5000), ('lpdquad.epfl.ch', 6000), ('lpdquad.epfl.ch', 7000),('lpdquad.epfl.ch', 8000), ('lpdquad.epfl.ch', 9000)]
n                = len(worker_HOSTS)
batch_size       = 50
learning_rate    = 0.05
activation_func  = tf.nn.relu
max_train_epoch  = 10000
max_train_accur  = 0.97
builder_opt      = tf.train.AdagradOptimizer(learning_rate)
builder_dims     = [784, 100, 10]

# ------------------------------------------------------------------------- #

# Dataset instantiation
dataset   = datasets.load_mnist()
train_set = dataset.cut(0, 50000, 50000).shuffle().cut(0, 50000, batch_size)
test_set  = dataset.cut(50000, 60000, 10000)

# Model instantiation
graph = tf.Graph()
with graph.as_default():
    model = models.dense_classifier(builder_dims, inputs=None, act_fn=activation_func, optimizer=builder_opt, epoch=True)

# Establish connections with workers
sockets = []
for worker_HOST in worker_HOSTS:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(worker_HOST)
    sockets.append(s)

# Testing + Training        
with graph.as_default():
    sess = tf.Session(graph=graph)

    with sess.as_default():
        sess.run(tf.global_variables_initializer())
        model.init()
        
        t_start = time.time()
        l = len(train_set)
        while True:
            # Test
            epoch = model.epoch()
            sys.stdout.write("Epoch " + str(epoch) + ": accuracy = ")
            sys.stdout.flush()
            acc = model.eval(*test_set.get())[0]
            print(str(acc))
            if acc >= max_train_accur or epoch >= max_train_epoch:
                break

            # Train
            it = iter(train_set)
            for i in range(0, l, n):
                grads = []
                # send batch, parameters to all workers
                for j in range(i, i+n):
                    p = int(j/l *100.)
                    sys.stdout.write("\rTraining... " + str(p) + "%")
                    sys.stdout.flush()
                    s = sockets[j%n]
                    batch = next(it)
                    prot.send_one_message(s, batch[0].tobytes())
                    prot.send_one_message(s, batch[1].tobytes())
                    prot.send_one_message(s, model.read().tobytes())

                # receive gradients
                for s in sockets:
                    data_grad = prot.recv_one_message(s)
                    grad = np.frombuffer(data_grad, np.float32)
                    grads.append(grad)

                # update model
                model.update(aggregate(grads))
            print("\rTraining... 100%")

        t_end = time.time()
        print('time: ', t_end - t_start)
        
for s in sockets:
    s.shutdown(socket.SHUT_RDWR)
    s.close()
