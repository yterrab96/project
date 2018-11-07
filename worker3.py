# worker 3

import numpy as np
import socket

import tensorflow as tf

import models
import protocol as prot

# ------------------------------------------------------------------------- #

# Parameters
HOST            = ''
PORT            = 8883
learning_rate   = 0.001
activation_func = tf.nn.relu

# ------------------------------------------------------------------------- #

# model instantiator
builder_opt = tf.train.AdagradOptimizer(learning_rate)
builder_dims = [784, 800, 10]

def builder(inputs=None):
    return models.dense_classifier(builder_dims, inputs=inputs, act_fn=activation_func, optimizer=builder_opt, epoch=True)

graph = tf.Graph()
with graph.as_default():
    model = builder()

with graph.as_default():
    sess = tf.Session(graph=graph)

    with sess.as_default():
        sess.run(tf.global_variables_initializer())
        model.init()

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            # wait for ps connection
            s.bind((HOST,PORT))
            s.listen(1)
            conn, addr = s.accept()
            print('connected by:', addr[0], ':', addr[1], sep='')
            while True:
                # receive observations
                data_x = prot.recv_one_message(conn)
                if not data_x:
                    break
                x = np.frombuffer(data_x, np.float32)
                x = x.reshape((50,784))

                # receive labels
                data_y = prot.recv_one_message(conn)
                y = np.frombuffer(data_y, np.int32)
                print('received batch')

                # receive parameters
                data_params = prot.recv_one_message(conn)
                params = np.frombuffer(data_params, np.float32)
                print('received parameters')

                # calculate gradient
                model.write(params)
                grad = model.backprop(x, y)
                    
                # send gradient
                prot.send_one_message(conn, grad.tobytes())
                print('sent gradient\n')

            # close connection
            conn.shutdown(socket.SHUT_RDWR)
            conn.close()
            print('connection closed')
