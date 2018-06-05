"""
Trainer and Metrics
"""

import time
from datetime import datetime
import tensorflow as tf
from cdssm_model import ChineseCdssmModel
from cdssm_input import DataProcessor
from cdssm_param import FLAGS
from cdssm_eval import ModelEvaluator
import numpy as np


class Metrics:
    """
    store evaluation metrics
    """
    def __init__(self, enable_early_stop=False, early_stop_steps=100000):
        self.best_value = 0
        self.best_step = 0
        self.bad_step = 0
        self.improved = False
        self.earlystop = False
        self.enable_early_stop = enable_early_stop
        self.early_stop_steps = early_stop_steps

    def update(self, value, step):
        """
        update metrics
        """
        if value > self.best_value:
            self.best_value = value
            self.best_step = step
            self.improved = True
            self.bad_step = 0
        else:
            self.improved = False
            self.bad_step += 1
            if self.enable_early_stop and self.bad_step > self.early_stop_steps:
                self.earlystop = True

class ModelTrainer:
    """
    CDSSM model trainer
    """
    def __init__(self,
                 model,
                 train_tfrecords,
                 eval_tfrecords,
                 log_dir,
                 log_frequency,
                 checkpoint_frequency,
                 input_previous_model_path,
                 output_model_path,
                 num_epochs=30,
                 train_batch_size=256,
                 eval_batch_size=1024,
                 max_length=100,
                 num_threads=1,
                 negative_sample=50,
                 softmax_gamma=10.0,
                 optimizer='adam',
                 learning_rate=0.001,
                 enable_early_stop=False,
                 early_stop_steps=100000):
        self.model = model
        self.train_tfrecords = train_tfrecords
        self.eval_tfrecords = eval_tfrecords
        self.log_dir = log_dir
        self.log_frequency = log_frequency
        self.checkpoint_frequency = checkpoint_frequency
        self.input_previous_model_path = input_previous_model_path
        self.output_model_path = output_model_path
        self.num_epochs = num_epochs
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.max_length = max_length
        self.num_threads = num_threads
        self.negative_sample = negative_sample
        self.softmax_gamma = softmax_gamma
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.total_loss = tf.Variable(0.)
        self.total_weight = tf.Variable(0.)
        self.metrics = Metrics(enable_early_stop, early_stop_steps)

    def calc_loss(self, qnvec, dnvec):
        """
        calculate loss for query and doc deep embedding result
        """
        batch_size = tf.shape(qnvec)[0]

        cosines = []
        cosines.append(tf.reduce_sum(tf.multiply(qnvec, dnvec), axis=1))

        def default():
            """
            default value
            """
            return tf.zeros([batch_size])

        def random_sample():
            """
            sample negative pairs randomly within the same batch
            """
            # generate random negative sample index
            random_indices = (tf.range(batch_size) + tf.random_uniform([batch_size], 1, batch_size,
                                                                       tf.int32)) % batch_size
            return tf.reduce_sum(tf.multiply(qnvec, tf.gather(dnvec, random_indices)), axis=1)

        for _ in range(0, self.negative_sample):
            cosines.append(tf.cond(batch_size > 1, random_sample, default))

        cosines = tf.stack(cosines, axis=1)
        softmax = tf.nn.softmax(cosines * self.softmax_gamma, axis=1)
        loss = tf.reduce_sum(-tf.log(softmax[:, 0]))
        tf.summary.scalar('softmax_losses', loss)
        return loss

    def update_loss(self, loss, weight):
        """
        update average loss
        """
        loss_inc = tf.assign_add(self.total_loss, loss / 10000)
        weight_inc = tf.assign_add(self.total_weight, tf.cast(weight / 10000, tf.float32))
        avg_loss = loss_inc / weight_inc
        tf.summary.scalar("avg_loss", avg_loss)
        return avg_loss

    def make_train_op(self, loss):
        """
        make tensorflow train operation
        """
        if self.optimizer == "sgd":
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        elif self.optimizer == "adam":
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        train_op = optimizer.minimize(loss)
        return train_op

    def evaluate(self, step, log_writer, data_reader, count_total, sess):
        count_correct = 0
        eval_q_vec = tf.placeholder(tf.int64)  # [b,query_len] , if query_len>seq_len, it will be truncated.
        eval_q_vec_len = tf.placeholder(tf.int64)
        eval_d_vec = tf.placeholder(tf.int64)
        eval_d_vec_len = tf.placeholder(tf.int64)
        eval_label = tf.placeholder(tf.int64)

        # print('eval_q_vec, eval_d_vec', eval_q_vec.get_shape(), eval_d_vec.get_shape())
        eval_qnvec, eval_dnvec = self.model.inference(eval_q_vec, eval_q_vec_len, eval_d_vec, eval_d_vec_len)
        feature_in = tf.concat([eval_qnvec, eval_dnvec], 1)
        feature_in = tf.expand_dims(feature_in, 1)
        feature_in = tf.expand_dims(feature_in, -1)
        logits = self.model.binary(feature_in)
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=eval_label, logits=logits)
        losses = tf.cast(losses, tf.float32)
        loss_sum = tf.reduce_sum(losses)
        min_index = tf.argmin(losses, 0)
        min_5 = tf.nn.top_k(tf.negative(losses), 5)
        min_loss_5 = min_5[0]
        min_index5 = min_5[1]
        log_writer.write('=====> step {0}, begin evaluation\n'.format(step))
        eval_data_batches = data_reader.random_batch_evaluation(count=count_total)
        for i in range(count_total):
            eval_data_batch = eval_data_batches[i]
            b_query_vec, b_query_length, b_entity_vec, b_entity_length, b_label = eval_data_batch
            # print(np.shape(np.array(b_query_vec)),np.shape(np.array(b_query_length)))
            # print(np.shape(np.array(b_entity_vec)),np.shape(np.array(b_entity_length)), np.shape(np.array(b_label)))
            feed_dict = {
                eval_q_vec: b_query_vec,
                eval_q_vec_len: b_query_length,
                eval_d_vec: b_entity_vec,
                eval_d_vec_len: b_entity_length,
                eval_label: b_label}
            min_loss_5_, min_index5_ = sess.run([min_loss_5, min_index5], feed_dict)
            if 0 in min_index5_:  # top5 for evaluation
                count_correct += 1
                print('index %d correct predict' % i)
                log_writer.write('index %d correct predict' % i + '\n')
            else:
                print('index %d false predict' % i)
                log_writer.write('index %d false predict' % i + '\n')

            # for logwriter
            q_vec = b_query_vec[0]
            q_name_raw = data_reader.devectorize(q_vec, '')
            q_name = filter(lambda ch: ch != '', q_name_raw)
            q_true_ent_vec = b_entity_vec[0]
            q_true_ent_name_raw = data_reader.devectorize(q_true_ent_vec, '')
            q_true_ent_name = filter(lambda ch: ch != '', q_true_ent_name_raw)
            q_true_ent_name = ' '.join(q_true_ent_name)
            # print('true entity name', q_true_ent_name)
            log_writer.write(' '.join(q_name) + '({0})'.format(q_true_ent_name) + '\n')
            entity_name = []  # candidates
            if 0 not in min_index5_:
                entity_name = data_reader.get_entity_by_index(min_index5_)
            else:
                for idx in min_index5_:
                    if idx != 0:
                        entity_name += data_reader.get_entity_by_index([idx])
                    else:
                        entity_name.append(q_true_ent_name)
            entity_name = '\t'.join(entity_name)
            log_writer.write(entity_name + '\n')
            log_writer.write(' '.join(map(str, min_loss_5_)) + '\n')
            log_writer.write(' '.join(map(str, min_index5_)) + '\n')
        precision = count_correct / (count_total * 1.00)
        print('==== evaluation precision:', precision)
        log_writer.flush()
        return precision

    # evaluate for 162 entities no duplicate
    def evaluate2(self, step, log_writer, data_reader, count_total, sess):
        count_correct = 0
        eval_q_vec = tf.placeholder(tf.int64)  # [b,query_len] , if query_len>seq_len, it will be truncated.
        eval_q_vec_len = tf.placeholder(tf.int64)
        eval_d_vec = tf.placeholder(tf.int64)
        eval_d_vec_len = tf.placeholder(tf.int64)
        eval_label = tf.placeholder(tf.int64)

        # print('eval_q_vec, eval_d_vec', eval_q_vec.get_shape(), eval_d_vec.get_shape())
        eval_qnvec, eval_dnvec = self.model.inference(eval_q_vec, eval_q_vec_len, eval_d_vec, eval_d_vec_len)
        feature_in = tf.concat([eval_qnvec, eval_dnvec], 1)
        feature_in = tf.expand_dims(feature_in, 1)
        feature_in = tf.expand_dims(feature_in, -1)
        logits = self.model.binary(feature_in)
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=eval_label, logits=logits)
        losses = tf.cast(losses, tf.float32)
        loss_sum = tf.reduce_sum(losses)
        min_index = tf.argmin(losses, 0)
        min_5 = tf.nn.top_k(tf.negative(losses), 5)
        min_loss_5 = min_5[0]
        min_index5 = min_5[1]
        log_writer.write('=====> step {0}, begin evaluation\n'.format(step))
        eval_data_batches, eval_qa = data_reader.random_batch_evaluation2(count=count_total)
        for i in range(count_total):
            eval_data_batch = eval_data_batches[i]
            b_query_vec, b_query_length, b_entity_vec, b_entity_length, b_label = eval_data_batch
            # print(np.shape(np.array(b_query_vec)),np.shape(np.array(b_query_length)))
            # print(np.shape(np.array(b_entity_vec)),np.shape(np.array(b_entity_length)), np.shape(np.array(b_label)))
            feed_dict = {
                eval_q_vec: b_query_vec,
                eval_q_vec_len: b_query_length,
                eval_d_vec: b_entity_vec,
                eval_d_vec_len: b_entity_length,
                eval_label: b_label}
            min_loss_5_, min_index5_ = sess.run([min_loss_5, min_index5], feed_dict)
            min_ent_vec5_ = [b_entity_vec[ind] for ind in min_index5_]
            if eval_qa[i][1] in min_ent_vec5_:  # top5 for evaluation
                count_correct += 1
                print('index %d correct predict' % i)
                log_writer.write('index %d correct predict' % i + '\n')
            else:
                print('index %d false predict' % i)
                log_writer.write('index %d false predict' % i + '\n')

            # for logwriter
            q_vec = eval_qa[i][0]
            q_name_raw = data_reader.devectorize(q_vec, '')
            q_name = filter(lambda ch: ch != '', q_name_raw)
            q_true_ent_vec = eval_qa[i][1]
            q_true_ent_name_raw = data_reader.devectorize(q_true_ent_vec, '')
            q_true_ent_name = filter(lambda ch: ch != '', q_true_ent_name_raw)
            q_true_ent_name = ' '.join(q_true_ent_name)
            # print('true entity name', q_true_ent_name)
            log_writer.write(' '.join(q_name) + ' \t({0})'.format(q_true_ent_name) + '\n')
            entity_name = data_reader.get_entity_by_index2(min_index5_)  # candidates
            entity_name = '\t'.join(entity_name)
            log_writer.write(entity_name + '\n')
            log_writer.write(' '.join(map(str, min_loss_5_)) + '\n')
            log_writer.write(' '.join(map(str, min_index5_)) + '\n')
        precision = count_correct / (count_total * 1.00)
        print('==== evaluation precision:', precision)
        log_writer.flush()
        return precision

    def _evaluate(self, data_processor):
        """
        train procedure
        """
        if not tf.gfile.Exists(self.log_dir):
            # tf.gfile.DeleteRecursively(self.log_dir)
            tf.gfile.MakeDirs(self.log_dir)
        if not tf.gfile.Exists(self.output_model_path):
            # tf.gfile.DeleteRecursively(self.output_model_path)
            tf.gfile.MakeDirs(self.output_model_path)

        # define training step
        global_step = tf.train.get_or_create_global_step()
        inc_step = tf.assign_add(global_step, 1)        # global step

        # build train graph
        train_q_vec = tf.placeholder(tf.int64)      #[b,query_len] , if query_len>seq_len, it will be truncated.
        train_q_vec_len = tf.placeholder(tf.int64)
        train_d_vec = tf.placeholder(tf.int64)
        train_d_vec_len = tf.placeholder(tf.int64)
        train_label = tf.placeholder(tf.int64)

        # print('train_q_vec, train_d_vec', train_q_vec.get_shape(), train_d_vec.get_shape())
        train_qnvec, train_dnvec = self.model.inference(train_q_vec, train_q_vec_len, train_d_vec, train_d_vec_len)
        # ==============================================================================================================
        feature_in = tf.concat([train_qnvec, train_dnvec], 1)
        feature_in = tf.expand_dims(feature_in, 1)
        feature_in = tf.expand_dims(feature_in, -1)
        logits = self.model.binary(feature_in)       # feature_in: [B, 1, 256, 1]
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=train_label, logits=logits) # [batch_size]
        # losses = tf.cast(losses, tf.float32)        # [batch_size]
        loss_sum = tf.reduce_sum(losses)
        # ==============================================================================================================
        # train_loss = self.calc_loss(train_qnvec, train_dnvec)
        train_loss = loss_sum
        avg_train_loss = self.update_loss(train_loss, tf.shape(train_qnvec)[0])
        train_op = self.make_train_op(train_loss)

        # define summary, config and model saver
        summary_op = tf.summary.merge_all()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        saver = tf.train.Saver(max_to_keep=10, name='model_saver')

        with tf.Session(config=config) as session:
            # initialize variables
            summ_writer = tf.summary.FileWriter(self.log_dir, session.graph)
            session.run(tf.local_variables_initializer())
            session.run(tf.global_variables_initializer())
            session.run(tf.tables_initializer())

            # Load pre-trained model
            # ckpt = tf.train.get_checkpoint_state(self.input_previous_model_path)
            ckpt = tf.train.get_checkpoint_state('data/binary_data/checkpoint_train/')
            # ckpt = tf.train.get_checkpoint_state(self.input_previous_model_path)
            if ckpt and ckpt.model_checkpoint_path:
                # saver.restore(session, ckpt.model_checkpoint_path)
                saver.restore(session, 'data/binary_data/checkpoint_train/step14000_precision1.0000_lossavg0.0714')
                # print("Load Model From ", ckpt.model_checkpoint_path)     # the absolute path where the model was initially trained and stored
                print("model loaded ")
            else:
                print("No initial model found.")

            # setup coordinator for thread management
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            log_format_str = "%s: step %d, batch_loss = %.10f, avg_loss = %.10f, %5.1f examples/sec, %.3f sec/batch, %.1f samples processed"

            reset_log_start_time = True
            reset_ckpt_start_time = True
            step = 14000
            # log_writer = open('data/binary_data/log.txt', encoding='utf-8', mode='w')
            log_writer = open('data/binary_data/log.eval.txt', encoding='utf-8', mode='w')
            count_total = 500
            # precision = self.evaluate(step, log_writer, data_processor, count_total, session)
            precision = self.evaluate2(step, log_writer, data_processor, count_total, session)
            print('evaluation count = {0}, precision = {1}'.format(count_total, precision))
            exit()

    @staticmethod
    def run_training_from_param():
        """
        run training from param
        """
        train_file_path = FLAGS.input_training_data_path
        train_tfrecords_path = train_file_path + ".tfrecords"

        eval_file_path = FLAGS.input_validation_data_path
        eval_tfrecords_path = eval_file_path + ".tfrecords"
        data_processor = DataProcessor()
        vocab,revocab = DataProcessor.initialize_vocabulary(FLAGS.vocab_path)
        vocab_size = DataProcessor.get_vocabulary_size(FLAGS.vocab_path)

        max_length = FLAGS.max_length
        data_processor.get_init(train_file_path, eval_file_path, vocab, vocab_size, max_length, revocab)


        # DataProcessor.create_tfrecord(train_file_path, train_tfrecords_path, vocab)
        # DataProcessor.create_tfrecord(eval_file_path, eval_tfrecords_path, vocab, is_training=False)

        with tf.Graph().as_default():
            model = ChineseCdssmModel(vocab_size=len(vocab),
                                      embedding_size=FLAGS.embedding_size,
                                      win_size=FLAGS.win_size,
                                      conv_size=FLAGS.conv_size,
                                      dense_size=FLAGS.dense_size,
                                      share_weight=FLAGS.share_weight)

            trainer = ModelTrainer(model=model,
                                   train_tfrecords=train_tfrecords_path,
                                   eval_tfrecords=eval_tfrecords_path,
                                   log_dir=FLAGS.log_dir,
                                   log_frequency=FLAGS.log_frequency,
                                   checkpoint_frequency=FLAGS.checkpoint_frequency,
                                   input_previous_model_path=FLAGS.input_previous_model_path,
                                   output_model_path=FLAGS.output_model_path,
                                   num_epochs=FLAGS.num_epochs,
                                   train_batch_size=FLAGS.train_batch_size,
                                   eval_batch_size=FLAGS.eval_batch_size,
                                   max_length=FLAGS.max_length,
                                   num_threads=FLAGS.num_threads,
                                   negative_sample=FLAGS.negative_sample,
                                   softmax_gamma=FLAGS.softmax_gamma,
                                   optimizer=FLAGS.optimizer,
                                   learning_rate=FLAGS.learning_rate,
                                   enable_early_stop=FLAGS.enable_early_stop,
                                   early_stop_steps=FLAGS.early_stop_steps)

            trainer._evaluate(data_processor)



if __name__ == '__main__':
    ModelTrainer.run_training_from_param()
    # trainer = ModelTrainer()
    # data_processor = DataProcessor()
    # log_writer = open('data/binary_data/log.eval.txt', encoding='utf-8', mode='w')
    # with tf.Session() as session:
    #     trainer.evaluate(14000, log_writer, data_processor, session)

