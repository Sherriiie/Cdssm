"""
Chinese CDSSM model
"""

import math
import tensorflow as tf

class ChineseCdssmModel:
    """
    Chinese CDSSM graph definition
    """
    def __init__(self, vocab_size, embedding_size, win_size, conv_size, dense_size, share_weight=True):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.win_size = win_size
        self.conv_size = conv_size
        self.dense_size = dense_size
        self.share_weight = share_weight
        self.use_one_hot_embedding = False
        if self.embedding_size <= 0:
            self.embedding_size = self.vocab_size + 1
            self.use_one_hot_embedding = True

    def inference(self, query_vec, query_vec_length, doc_vec, doc_vec_length):
        """
        query and doc deep embedding inference
        """
        qnvec = self.forward_propagation(query_vec, query_vec_length)
        if self.share_weight:
            dnvec = self.forward_propagation(doc_vec, doc_vec_length)
        else:
            dnvec = self.forward_propagation(doc_vec, doc_vec_length, 'd')
        return qnvec, dnvec

    def binary(self, text_embedding):
        num_class = 2
        with tf.variable_scope("conv_maxpooling_layer_binary", reuse=tf.AUTO_REUSE):
            # weight = tf.get_variable(name='conv_weight',
            #                          shape=[kernel_width, 1, 1, cnn_conv_size],
            #                          initializer=tf.random_uniform_initializer(-random_range, random_range))
            weight = tf.get_variable(name='conv_weight',
                                     shape=[1, self.win_size, 1, self.conv_size],
                                     initializer=tf.random_uniform_initializer(-1.0, 1.0))
            # filter_shape = [kernel_width, 1, 1, cnn_conv_size]
            # weight  = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1))
            bias = tf.get_variable(name='conv_bias',
                                   shape=[self.conv_size],
                                   initializer=tf.constant_initializer(0.1))

        conv = tf.nn.conv2d(text_embedding, weight, strides=[1, 1, 1, 1], padding="SAME")       # text_embedding=(B, 1, 256, 1)
        print('==== classification(), conv: ', conv.get_shape())
        nonlinear = tf.nn.tanh(tf.nn.bias_add(conv, bias))
        maxpooling = tf.reduce_max(nonlinear, axis=2)
        maxpooling = tf.reshape(maxpooling, [-1, self.conv_size])
        print('==== classification(), maxpooling: ', maxpooling.get_shape())

        # add an output layer for 2 classes
        # random_range = math.sqrt(6.0 / (cnn_conv_size + num_class))
        with tf.variable_scope("dense_layer", reuse=tf.AUTO_REUSE):
            weight = tf.get_variable(name='dense_weight',
                                     shape=[self.conv_size, num_class],
                                     initializer=tf.random_uniform_initializer(-1, 1))
            bias = tf.get_variable(name='dense_bias',
                                   shape=[num_class],
                                   initializer=tf.constant_initializer(0.1))

        matmul = tf.matmul(maxpooling, weight)
        logits = tf.nn.bias_add(matmul, bias)
        return logits

    def forward_propagation(self, text_vec, text_vec_length, model_prefix='q'):
        """
        forward propagation process
        """
        text_embedding = self.embedding_layer(text_vec, text_vec_length, model_prefix)
        maxpooling = self.conv_maxpooling_layer(text_embedding, model_prefix)
        normalized_vec = self.dense_layer(maxpooling, model_prefix)
        return normalized_vec

    def classification(self, qnvec, dnvec):
        """
        based on the embedding of query and document, implement binary classification
        input: q- batch_size*128
               d- batch_size*128
        output: batch_size*1, binary
        """
        print(tf.shape(qnvec), tf.shape(dnvec))
        feature_in = tf.concat([qnvec, dnvec], 1)
        print(tf.shape(feature_in))
        feature_in = tf.expand_dims(feature_in, 2)
        feature_in = tf.expand_dims(feature_in, 3)


        cnn_conv_size = 256
        kernel_width = 3
        num_class = 2
        padding_margin = int(kernel_width / 2)
        padding = tf.constant([[0, 0], [padding_margin, padding_margin], [0, 0]])
        text_embedding = tf.pad(feature_in, padding)
        text_embedding = tf.expand_dims(text_embedding, -1)


        random_range = math.sqrt(6.0 / ( tf.shape(feature_in)[1]+ cnn_conv_size))
        with tf.variable_scope("conv_maxpooling_layer", reuse=tf.AUTO_REUSE):
            weight = tf.get_variable(name='conv_weight',
                                     shape=[kernel_width, 1, 1, cnn_conv_size],
                                     initializer=tf.random_uniform_initializer(-random_range, random_range))
            bias = tf.get_variable(name='conv_bias',
                                   shape=[cnn_conv_size],
                                   initializer=tf.constant_initializer(0.1))

        conv = tf.nn.conv2d(text_embedding, weight, strides=[1, 1, 1, 1], padding="VALID")
        nonlinear = tf.nn.tanh(tf.nn.bias_add(conv, bias))
        maxpooling = tf.reduce_max(nonlinear, axis=1)
        maxpooling = tf.reshape(maxpooling, [-1, cnn_conv_size])

        # random_range = math.sqrt(6.0 / (cnn_conv_size + num_class))
        with tf.variable_scope("dense_layer", reuse=tf.AUTO_REUSE):
            weight = tf.get_variable(name='dense_weight',
                                     shape=[cnn_conv_size, num_class],
                                     initializer=tf.random_uniform_initializer(-1, 1))
            bias = tf.get_variable(name='dense_bias',
                                   shape=[num_class],
                                   initializer=tf.constant_initializer(0.1))
        matmul = tf.matmul(maxpooling, weight)
        score = tf.nn.bias_add(matmul, bias)
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=score)
        loss = tf.reduce_sum(losses)
        label = read_labels(batch_size)

    def read_labels(batch_size):
        with open('labels', 'r') as file:
            lines = file.readlines()

    def embedding_layer(self, text_vec, text_vec_length, model_prefix='q'):
        """
        dense embedding layer
        """
        text_max_length = tf.reduce_max(text_vec_length)
        text_vec = text_vec[:, :text_max_length]

        if self.use_one_hot_embedding:
            text_embedding = tf.one_hot(text_vec, self.embedding_size)
        else:
            random_range = math.sqrt(6.0 / (self.vocab_size + 1 + self.embedding_size))
            with tf.variable_scope("dense_embedding_layer", reuse=tf.AUTO_REUSE):
                weight = tf.get_variable(name='embedding_weight_' + model_prefix,
                                         shape=[self.vocab_size + 1, self.embedding_size],
                                         initializer=tf.random_uniform_initializer(-random_range, random_range))
            text_embedding = tf.nn.embedding_lookup(weight, text_vec)

        seq_mask = tf.sequence_mask(text_vec_length, text_max_length)
        seq_mask = tf.expand_dims(seq_mask, axis=-1)
        seq_mask = tf.tile(seq_mask, [1, 1, self.embedding_size])
        text_embedding = tf.where(seq_mask, text_embedding, tf.zeros_like(text_embedding))

        return text_embedding

    def conv_maxpooling_layer(self, text_embedding, model_prefix='q'):
        """
        convolution and maxpooling layer
        """
        padding_margin = int(self.win_size / 2)
        padding = tf.constant([[0, 0], [padding_margin, padding_margin], [0, 0]])
        text_embedding = tf.pad(text_embedding, padding)
        text_embedding = tf.expand_dims(text_embedding, -1)

        random_range = math.sqrt(6.0 / (self.embedding_size + self.conv_size))
        with tf.variable_scope("conv_maxpooling_layer", reuse=tf.AUTO_REUSE):
            weight = tf.get_variable(name='conv_weight_' + model_prefix,
                                     shape=[self.win_size, self.embedding_size, 1, self.conv_size],
                                     initializer=tf.random_uniform_initializer(-random_range, random_range))
            bias = tf.get_variable(name='conv_bias_' + model_prefix,
                                   shape=[self.conv_size],
                                   initializer=tf.constant_initializer(0.1))

        conv = tf.nn.conv2d(text_embedding, weight, strides=[1, 1, 1, 1], padding="VALID")
        nonlinear = tf.nn.tanh(tf.nn.bias_add(conv, bias))
        maxpooling = tf.reduce_max(nonlinear, axis=1)
        maxpooling = tf.reshape(maxpooling, [-1, self.conv_size])
        return maxpooling

    def dense_layer(self, maxpooling, model_prefix='q'):
        """
        dense layer
        """
        random_range = math.sqrt(6.0 / (self.conv_size + self.dense_size))
        with tf.variable_scope("dense_layer", reuse=tf.AUTO_REUSE):
            weight = tf.get_variable(name='dense_weight_' + model_prefix,
                                     shape=[self.conv_size, self.dense_size],
                                     initializer=tf.random_uniform_initializer(-random_range, random_range))
            bias = tf.get_variable(name='dense_bias_' + model_prefix,
                                   shape=[self.dense_size],
                                   initializer=tf.constant_initializer(0.1))

        matmul = tf.matmul(maxpooling, weight)
        nonlinear = tf.nn.tanh(tf.nn.bias_add(matmul, bias))
        normalized_vec = tf.nn.l2_normalize(nonlinear, axis=1)
        return normalized_vec
