# asynchronous parameter server

import numpy as np
import sys
import socket
import threading
import time

import tensorflow as tf

import datasets
import models
import protocol as prot

def kardam(worker_id, epoch, grad): # decides to update or not
    with model_lock:
        model.update(grad)

# ------------------------------------------------------------------------- #

def worker_thread(worker_HOST):
    global model
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(worker_HOST)
    with sess.as_default():
        while not converged:
            batch = train_set.get()
            with model_lock:
                params = model.read()
                step = model.epoch()
            prot.send_one_message(s, batch[0].tobytes())
            prot.send_one_message(s, batch[1].tobytes())
            prot.send_one_message(s, params.tobytes())
            data_grad = prot.recv_one_message(s)
            grad = np.frombuffer(data_grad, np.float32)
            with model_lock:
                model.update(grad)
    s.shutdown(socket.SHUT_RDWR)
    s.close()

# ------------------------------------------------------------------------- #

# Parameters
#worker_HOSTS     = [('lpdquad.epfl.ch', 5000)]
#worker_HOSTS     = [('lpdquad.epfl.ch', 5000), ('lpdquad.epfl.ch', 6000)]
worker_HOSTS     = [('lpdquad.epfl.ch', 5000), ('lpdquad.epfl.ch', 6000), ('lpdquad.epfl.ch', 7000),('lpdquad.epfl.ch', 8000), ('lpdquad.epfl.ch', 9000)]
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

# Testing + Training        
with graph.as_default():
    sess = tf.Session(graph=graph)

    with sess.as_default():
        sess.run(tf.global_variables_initializer())
        model.init()
        model_lock = threading.Lock()
        
        t_start = time.time()
        converged = False
        for worker_HOST in worker_HOSTS:
            threading.Thread(target=worker_thread, args=(worker_HOST, )).start()

        while True:
            epoch = model.epoch()
            sys.stdout.write("Epoch " + str(epoch) + ": accuracy = ")
            sys.stdout.flush()
            acc = model.eval(*test_set.get())[0]
            print(str(acc))
            if acc >= max_train_accur or epoch >= max_train_epoch:
                converged = True
                break
            time.sleep(1.8)

        t_end = time.time()
        print('time: ', t_end - t_start)
