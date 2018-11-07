# asynchronous parameter server

import numpy as np
import sys
import socket
import threading
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf

import datasets
import models
import protocol as prot

# ------------------------------------------------------------------------- #

def worker_thread(worker_HOST):
    global model
    global converged
    global accs
    global epochs
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(worker_HOST)
    with sess.as_default():
        while True:
            if converged:
                break
            batch = train_set.get()
            with model_lock:
                epoch  = model.epoch()
                params = model.read()
            prot.send_one_message(s, batch[0].tobytes())
            prot.send_one_message(s, batch[1].tobytes())
            prot.send_one_message(s, params.tobytes())
            data_grad = prot.recv_one_message(s)
            grad = np.frombuffer(data_grad, np.float32)

            with model_lock:
                if converged:
                    break
                model.update(grad)
                new_epoch = model.epoch()
                acc       = model.eval(*test_set.get())[0]
                staleness = new_epoch - epoch
                epochs.append(new_epoch)
                accs.append(acc)
                stales.append(staleness)
                print(worker_HOST[1], epoch, '=>', new_epoch, acc, staleness)
                if acc >= max_train_accur or new_epoch > max_train_epoch:
                    converged = True
    s.shutdown(socket.SHUT_RDWR)
    s.close()

# ------------------------------------------------------------------------- #

# Parameters
#worker_HOSTS     = [('lpdquad.epfl.ch', 8881)]
#worker_HOSTS     = [('lpdquad.epfl.ch', 8881), ('lpdquad.epfl.ch', 8882)]
worker_HOSTS     = [('lpdquad.epfl.ch', 8881), ('lpdquad.epfl.ch', 8882), ('lpdquad.epfl.ch', 8883),('lpdquad.epfl.ch', 8884), ('lpdquad.epfl.ch', 8885)]
batch_size       = 50
learning_rate    = 0.001
activation_func  = tf.nn.relu
max_train_epoch  = 5000
max_train_accur  = 0.96
builder_opt      = tf.train.AdagradOptimizer(learning_rate)
builder_dims     = [784, 800, 10]

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
        epochs = []
        accs   = []
        stales = []
        model.init()
        model_lock = threading.Lock()
        epochs.append(model.epoch())
        accs.append(model.eval(*test_set.get())[0])
        stales.append(0)
        print(epochs[len(epochs)-1], accs[len(accs)-1])

        converged = False
        threads = []
        t_start = time.time()
        for worker_HOST in worker_HOSTS:
            thread = threading.Thread(target=worker_thread, args=(worker_HOST, ))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()
            
        t_end = time.time()
        print('time: ', t_end - t_start)
        
        fig, = plt.plot(epochs, accs, 'r', linewidth=0.5, label='2 stale workers')
        plt.legend(handles=[fig], loc=4)
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.savefig('2_stale_acc.png')

