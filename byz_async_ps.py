# byzantine-resilient asynchronous parameter server

import numpy as np
import socket
import threading
import time
import copy

import tensorflow as tf

import datasets
import models
import protocol as prot
from utils import Epoch_Params_Grad_Holder


def lipz_filters_in(lipz_coeff, candidate_lipz_coeffs):
    if f==0:
        return True
    print(candidate_lipz_coeffs)
    lipz_coeffs = np.fromiter(candidate_lipz_coeffs.values(), dtype=float)
    n_f_percentile = np.percentile(lipz_coeffs, 100*(n-f)/n)
    print(lipz_coeff, n_f_percentile)
    return lipz_coeff <= n_f_percentile

def kardam(lipz_coeff, grad):
    global model
    if lipz_filters_in(lipz_coeff):
        with model_lock:
            model.update(grad)
    else:
        pass

# ------------------------------------------------------------------------- #

def worker_thread(worker_HOST):
    global model
    global workers_lipz_coeffs
    global ps_grads
    worker_grads = Epoch_Params_Grad_Holder()
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
            grad      = np.frombuffer(data_grad, np.float32)

            worker_grads.add((epoch, params, grad))
            with model_lock:
                candidate_lipz_coeffs = copy.deepcopy(workers_lipz_coeffs)
                candidate_lipz_coeffs[worker_HOST[1]] = worker_grads.lipz_coeff()

                ps_grads.add((model.epoch(), model.read(), grad))
                lipz_coeff = ps_grads.lipz_coeff()
                delay = time.time() - t_start
                if delay < 3:
                    model.update(grad)
                    workers_lipz_coeffs = candidate_lipz_coeffs
                    print(workers_lipz_coeffs)
                    print('accepted1', worker_HOST[1], ps_grads.prev_epoch(), ps_grads.epoch())
                else:
                    if lipz_filters_in(lipz_coeff, candidate_lipz_coeffs):
                        model.update(grad)
                        workers_lipz_coeffs = candidate_lipz_coeffs
                        print('accepted:', worker_HOST[1], ps_grads.prev_epoch(), ps_grads.epoch())
                    else: 
                        worker_grads.revert() 
                        ps_grads.revert()
                        print('rejected:', worker_HOST[1], ps_grads.prev_epoch(), ps_grads.epoch())
                
    s.shutdown(socket.SHUT_RDWR)
    s.close()

# ------------------------------------------------------------------------- #

# Parameters
#worker_HOSTS     = [('lpdquad.epfl.ch', 8881)]
#worker_HOSTS     = [('lpdquad.epfl.ch', 8881), ('lpdquad.epfl.ch', 8885)]
worker_HOSTS     = [('lpdquad.epfl.ch', 8881), ('lpdquad.epfl.ch', 8882), ('lpdquad.epfl.ch', 8883),('lpdquad.epfl.ch', 8884), ('lpdquad.epfl.ch', 8885)]
batch_size       = 50
learning_rate    = 0.5
activation_func  = tf.nn.relu
max_train_epoch  = 10000
max_train_accur  = 0.97
builder_opt      = tf.train.AdagradOptimizer(learning_rate)
builder_dims     = [784, 100, 10]
n                = len(worker_HOSTS)
f                = 1

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
        #
        ps_grads = Epoch_Params_Grad_Holder()
        workers_lipz_coeffs = {}

        t_start = time.time()
        converged = False
        for worker_HOST in worker_HOSTS:
            threading.Thread(target=worker_thread, args=(worker_HOST, )).start()

        while True:
            with model_lock:
                epoch = model.epoch()
                acc   = model.eval(*test_set.get())[0]
            print('Epoch ' + str(epoch) + ': accuracy = ' + str(acc))
            if acc >= max_train_accur or epoch >= max_train_epoch:
                converged = True
                break
            time.sleep(1.5)

        t_end = time.time()
        print('time: ', t_end - t_start)
