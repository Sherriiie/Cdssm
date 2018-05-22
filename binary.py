import math
import tensorflow as tf
from cdssm_pred import ModelPredictor
import os
import numpy as np
import random

"""
text: list of text
"""

batch_size = 163
eval_size = batch_size
dim = 128
learning_rate = 0.01
num_epoch = 500
save_checkpoint_frequency = 2000
eval_frequency = 500
cnn_conv_size = 500
kernel_width = 3

with tf.Graph().as_default():
    qnvec = tf.placeholder(tf.float32, [batch_size, dim], name="input_x1")
    dnvec = tf.placeholder(tf.float32, [batch_size, dim], name="input_x2")
    label = tf.placeholder(tf.int32, [batch_size], name="input_y")
    def embedding_converter(text, predictor):
        text_embedding = []
        for t in text:
            text_embedding.append(predictor.predict(t))
        return text_embedding

    def classification():
        """
        based on the embedding of query and document, implement binary classification
        input: q- batch_size*128
               d- batch_size*128
        output: batch_size*1, binary
        """
        print('==== classification():', qnvec.get_shape(), dnvec.get_shape())
        feature_in = tf.concat([qnvec, dnvec], 1)
        feature_in = tf.expand_dims(feature_in, 1)
        # feature_in = tf.expand_dims(feature_in, 3)

        num_class = 2
        # padding_margin = int(kernel_width / 2)
        # padding = tf.constant([[0, 0], [padding_margin, padding_margin], [0, 0]])
        # text_embedding = tf.pad(feature_in, padding)
        text_embedding = tf.expand_dims(feature_in, -1)
        print('==== classification(), text_embedding: ', text_embedding.get_shape())
        # random_range = math.sqrt(6.0 / (tf.shape(feature_in)[1] + cnn_conv_size))
        with tf.variable_scope("conv_maxpooling_layer", reuse=tf.AUTO_REUSE):
            # weight = tf.get_variable(name='conv_weight',
            #                          shape=[kernel_width, 1, 1, cnn_conv_size],
            #                          initializer=tf.random_uniform_initializer(-random_range, random_range))
            weight = tf.get_variable(name='conv_weight',
                                     shape=[1, kernel_width, 1, cnn_conv_size],
                                     initializer=tf.random_uniform_initializer(-1.0, 1.0))
            # filter_shape = [kernel_width, 1, 1, cnn_conv_size]
            # weight  = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1))
            bias = tf.get_variable(name='conv_bias',
                                   shape=[cnn_conv_size],
                                   initializer=tf.constant_initializer(0.1))

        conv = tf.nn.conv2d(text_embedding, weight, strides=[1, 1, 1, 1], padding="SAME")
        print('==== classification(), conv: ', conv.get_shape())
        nonlinear = tf.nn.tanh(tf.nn.bias_add(conv, bias))
        maxpooling = tf.reduce_max(nonlinear, axis=2)
        maxpooling = tf.reshape(maxpooling, [-1, cnn_conv_size])
        print('==== classification(), maxpooling: ', maxpooling.get_shape())

        # random_range = math.sqrt(6.0 / (cnn_conv_size + num_class))
        with tf.variable_scope("dense_layer", reuse=tf.AUTO_REUSE):
            weight = tf.get_variable(name='dense_weight',
                                     shape=[cnn_conv_size, num_class],
                                     initializer=tf.random_uniform_initializer(-1, 1))
            bias = tf.get_variable(name='dense_bias',
                                   shape=[num_class],
                                   initializer=tf.constant_initializer(0.1))

        matmul = tf.matmul(maxpooling, weight)
        logits = tf.nn.bias_add(matmul, bias)

        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=logits)
        loss_sum = tf.reduce_sum(losses)
        min_index = tf.argmin(losses, 0)
        print('loss shape ', loss_sum.get_shape())
        print('losses shape ', losses.get_shape())
        return [loss_sum, min_index, losses]


    class DataReader():
        def __init__(self, data_path_train, data_path_eval):
            self.data_path_random = data_path_train
            self.data_path_eval = data_path_eval
            # self.data_path = data_path
            self.positive_pairs = []
            self.queries = []
            self.entities = []
            self.counter = 0        # for train
            self.total_item_count = 0  # positive pairs
            self.read_data()

            self.eval_positive_pairs = []
            self.eval_queries = []
            self.eval_entities = []
            self.eval_counter = 0       # for eval
            self.eval_total_item_count = 0
            self.read_eval_data()
            # no dup in entities_neg
            # print('there are %d entities in neg no dup'%len(self.entities_neg))
            self.entities_neg = list(set(self.entities))
            # write entities_neg to file
            self.write_to_file(self.entities_neg)

            self.flag = True

        def write_to_file(self, entities):
            file_write = open('data/binary_data/entities_evaluation.txt', encoding='utf-8', mode='w')
            temp =[ent.replace(' ', '') for ent in entities]
            file_write.write(' '.join(temp))

        def read_data(self):
            with open(self.data_path_random, encoding='utf-8', mode='r') as file_reader:
                for line in file_reader:
                    line = line.strip()
                    self.positive_pairs.append(line)
                    parts = line.split('\t')
                    self.queries.append(parts[0])
                    self.entities.append(parts[1])

                self.total_item_count = len(self.positive_pairs)
                print('==== read training data into model finished')
                # return self.q1, self.q2, self.label

        def read_eval_data(self ):
            with open(self.data_path_eval, encoding='utf-8', mode='r') as file_reader:
                for line in file_reader:
                    line = line.strip()
                    self.eval_positive_pairs.append(line)
                    parts = line.split('\t')
                    self.eval_queries.append(parts[0])
                    self.eval_entities.append(parts[1])
                self.eval_total_item_count = len(self.eval_positive_pairs)
                print('==== read eval data finished')

        def next_batch_random(self, batch_size):
            b_query = []
            b_entity =[]
            b_label = []
            for i in range(batch_size):
                if self.flag == True:
                    self.flag = not self.flag
                    b_query.append(self.queries[self.counter])
                    b_entity.append((self.entities[self.counter]))
                    b_label.append(1)
                    self.counter += 1
                    self.counter = self.counter%self.total_item_count
                else:
                    self.flag = not self.flag
                    b_query.append(self.queries[self.counter])
                    index = random.randint(0, self.total_item_count-1)
                    while (self.entities[index] == self.entities[self.counter]):
                        index = random.randint(0, self.total_item_count - 1)
                    b_entity.append((self.entities[index]))
                    b_label.append(0)
            return [b_query, b_entity, b_label]
            # return [self.queries[self.counter - batch_size: self.counter], self.entities[self.counter - batch_size:self.counter],
            #         self.label[self.counter - batch_size:self.counter]]

        def next_evaluation_batch(self):
            b_query = []
            b_entity = []
            b_label = []
            entities_neg = self.entities_neg.copy()
            # if self.eval_entities[self.eval_counter] in entities_neg:
            #     entities_neg.remove(self.eval_entities[self.eval_counter])
            # else:
            #     random_number = random.randint(0,len(entities_neg)-1)
            #     del entities_neg[random_number]


            # add the postive sample first
            b_query.append(self.eval_queries[self.eval_counter])
            b_entity.append(self.eval_entities[self.eval_counter])
            b_label.append(1)
            if(len(entities_neg)!=162):
                print('bug')
            # add all other entities as negative samples
            for i in range(len(entities_neg)):
                b_query.append(self.eval_queries[self.eval_counter])
                b_entity.append((entities_neg[i]))
                b_label.append(0)
            # entities_neg.append(self.eval_entities[self.eval_counter])
            self.eval_counter += 1
            self.eval_counter = self.eval_counter % self.eval_total_item_count
            return [b_query, b_entity, b_label]

    def validate(step_):
        count_correct = 0
        print('\n============================> begin to validate, step = ', str(step_))
        # there is no validation data now. Put your validation data here
        validate_data_batch = data_reader.next_batch_random(batch_size)
        vector_batch = []
        vector_batch.append(embedding_converter(validate_data_batch[0], predictor))
        vector_batch.append(embedding_converter(validate_data_batch[1], predictor))
        vector_batch.append(validate_data_batch[2])

        feed_dict = {
            qnvec: vector_batch[0],
            dnvec: vector_batch[1],
            label: vector_batch[2]}
        _, step_, loss_ = sess.run([train_op, global_step, loss], feed_dict)
        loss_average = loss_ / batch_size
        print('step = {0}, loss = {1}, loss_average = {2:.4f}'.format(step_, loss_, loss_average))
        return

    def evaluate(step_, log_writer):
        count_correct = 0
        count_total = 10
        print('\n============================> begin to evaluate, step = ', str(step_))
        for i in range(count_total):
            eval_data_batch = data_reader.next_evaluation_batch()
            vector_batch = []
            vector_batch.append(embedding_converter(eval_data_batch[0], predictor))
            vector_batch.append(embedding_converter(eval_data_batch[1], predictor))
            vector_batch.append(eval_data_batch[2])
            feed_dict = {
                qnvec: vector_batch[0],
                dnvec: vector_batch[1],
                label: vector_batch[2]}
            min_index_, loss_, losses_ = sess.run([min_index, loss, losses], feed_dict)
            log_writer.write(' '.join(map(str, losses_)) + '\n')
            if min_index_ == 0:
                count_correct += 1
                print('index %d correct predict'%i)
            else:
                print('index %d false predict'%i)
        precision = count_correct / (count_total*1.00)
        print('==== evaluation precision:', precision)
        log_writer.flush()
        return precision

    data_reader = DataReader('data/binary_data/train.txt', 'data/binary_data/evaluation.txt')
    predictor = ModelPredictor()
    loss, min_index, losses = classification()
    with tf.Session() as sess:
        global_step = tf.Variable(0, name="global_step",
                                  trainable=False)  # The global step will be automatically incremented by one every time you execute　a train loop
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        grads_and_vars = optimizer.compute_gradients(loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Output directory for models and summaries
        checkpoint_dir = os.path.abspath(os.path.join(os.path.curdir, "data/binary_data/checkpoints_train"))
        print("Writing to {}\n".format(checkpoint_dir))
        log_writer = open('data/binary_data/log.txt', encoding='utf-8', mode='w')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables())
        sess.run(tf.global_variables_initializer())
        model = tf.train.get_checkpoint_state('data/binary_data/checkpoints_train/')
        if model and model.model_checkpoint_path:
            # saver.restore(sess, model.model_checkpoint_path)
            print('\n =======> Model restored ')
        else:
            print('\n =======> No older model found ')

        step_ = 0
        precision_max = 0
        evaluate(step_, log_writer)
        while step_ < num_epoch * 4269 / batch_size:        # 13176
            train_data_batch = data_reader.next_batch_random(batch_size)
            vector_batch = []
            vector_batch.append(embedding_converter(train_data_batch[0], predictor))
            vector_batch.append(embedding_converter(train_data_batch[1], predictor))
            vector_batch.append(train_data_batch[2])
            feed_dict = {
                qnvec: vector_batch[0],
                dnvec: vector_batch[1],
                label: vector_batch[2]}
            _, step_ , loss_= sess.run([train_op, global_step, loss], feed_dict)
            loss_average = loss_/batch_size
            print('step = {0}, loss_sum = {1}, loss_average = {2:.4f}'.format(step_, loss_, loss_average))
            if step_ % eval_frequency == 0:
                prec = evaluate(step_, log_writer)
                log_writer.write('step = {0},\tprecision = {1}, loss_sum = {2}'.format(step_, prec, loss_))
                log_writer.flush()
                if (prec >= precision_max):
                    print('\n improved =====> begin to save checkpoint, step = {0} precision = {1}'.format(step_, prec))
                    path = saver.save(sess,
                                      checkpoint_dir + '/step' + str(step_) + '_precision' + '{0:.4f}'.format(prec)+'_loss' + '{0:.4f}'.format(loss_average) )
                    print("Save checkpoint(model) to {}".format(path))

                elif step_ % save_checkpoint_frequency == 0:
                    print('\n =====> begin to save checkpoint, step = {0} precision = {1}'.format(step_, prec))
                    path = saver.save(sess,
                                      checkpoint_dir + '/step' + str(step_) + '_precision' + '{0:.4f}'.format(prec)+'_loss' + '{0:.4f}'.format(loss_average))
                    print("Save checkpoint(model) to {}".format(path))
                else:
                    print('\n step = {0} precision = {1}'.format(step_, prec))




