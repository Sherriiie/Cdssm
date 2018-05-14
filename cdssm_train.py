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

    def train(self):
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
        inc_step = tf.assign_add(global_step, 1)

        # build train graph
        train_data = DataProcessor.load_training_tfrecords(self.train_tfrecords,
                                                           self.num_epochs,
                                                           self.train_batch_size,
                                                           self.max_length,
                                                           self.num_threads)
        train_q_vec, train_q_vec_len, train_d_vec, train_d_vec_len = train_data
        train_qnvec, train_dnvec = self.model.inference(train_q_vec, train_q_vec_len, train_d_vec, train_d_vec_len)
        train_loss = self.calc_loss(train_qnvec, train_dnvec)
        avg_train_loss = self.update_loss(train_loss, tf.shape(train_qnvec)[0])
        train_op = self.make_train_op(train_loss)

        # build scoring graph
        eval_q_vec = tf.placeholder(tf.int64)
        eval_q_vec_len = tf.placeholder(tf.int64)
        eval_d_vec = tf.placeholder(tf.int64)
        eval_d_vec_len = tf.placeholder(tf.int64)
        eval_qnvec, eval_dnvec = self.model.inference(eval_q_vec, eval_q_vec_len, eval_d_vec, eval_d_vec_len)
        score_op = tf.reduce_sum(tf.multiply(eval_qnvec, eval_dnvec), axis=1)

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
            ckpt = tf.train.get_checkpoint_state(self.input_previous_model_path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(session, ckpt.model_checkpoint_path)
                print("Load Model From ", ckpt.model_checkpoint_path)
            else:
                print("No initial model found.")

            # setup coordinator for thread management
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            log_format_str = "%s: step %d, batch_loss = %.10f, avg_loss = %.10f, %5.1f examples/sec, %.3f sec/batch, %.1f samples processed"
            step = 0

            try:
                reset_log_start_time = True
                reset_ckpt_start_time = True

                while not coord.should_stop():
                    if reset_log_start_time:
                        log_start_time = time.time()
                        reset_log_start_time = False
                    if reset_ckpt_start_time:
                        ckpt_start_time = time.time()
                        reset_ckpt_start_time = False

                    # one step training
                    _, loss, avg_loss, weight, step, summary = session.run([train_op, train_loss, avg_train_loss, self.total_weight, inc_step, summary_op])

                    # log performance
                    if step % self.log_frequency == 0:
                        summ_writer.add_summary(summary, step)
                        current_time = time.time()
                        duration = current_time - log_start_time
                        reset_log_start_time = True

                        examples_per_sec = self.log_frequency * self.train_batch_size / duration
                        sec_per_batch = float(duration / self.log_frequency)
                        print(log_format_str % (datetime.now(), step, loss, avg_loss, examples_per_sec, sec_per_batch, weight))

                    # metrics evaluation and checkpoint model saving
                    if step % self.checkpoint_frequency == 0:
                        current_time = time.time()
                        duration = current_time - ckpt_start_time
                        reset_ckpt_start_time = True
                        trained_samples = self.checkpoint_frequency * self.train_batch_size

                        query_id_list = []
                        label_list = []
                        score_list = []
                        for record in DataProcessor.load_evaluation_tfrecords(self.eval_tfrecords, self.eval_batch_size, self.max_length):
                            query_id, q_vec, q_vec_len, d_vec, d_vec_len, label = record
                            score = session.run(score_op, feed_dict={eval_q_vec: q_vec,
                                                                     eval_q_vec_len: q_vec_len,
                                                                     eval_d_vec: d_vec,
                                                                     eval_d_vec_len: d_vec_len})
                            query_id_list.extend(query_id)
                            label_list.extend(label)
                            score_list.extend(score)

                        query_cnt, auc, precision = ModelEvaluator.evaluate_from_list(query_id_list, label_list, score_list)

                        self.metrics.update(precision, step)
                        eval_format_str = "%s: step %d, trained %d samples in %0.0fs. Top 1 evaluation result: query_cnt = %d, auc = %0.10f, precision_at_full_coverage = %0.10f, improved = %s"
                        impr = self.metrics.improved
                        if impr:
                            # save model to disk if metrics has improvement
                            saver.save(session, self.output_model_path + "/cdssm_model", global_step=step)
                            print(eval_format_str % (datetime.now(), step, trained_samples, duration, query_cnt, auc, precision, impr))
                        else:
                            eval_format_str += ", best_precision_at_full_coverage = %0.10f, best_step = %d"
                            print(eval_format_str % (datetime.now(), step, trained_samples, duration, query_cnt, auc, precision, impr, self.metrics.best_value, self.metrics.best_step))

                        if self.metrics.earlystop:
                            print("\nEarly stop")
                            break

            # this is a workaround to deal with tensorflow 'FIFOQueue is closed and has insufficient elements' exception
            except tf.errors.OutOfRangeError:
                pass

            coord.request_stop()
            coord.join(threads)
            print('Done training')

    @staticmethod
    def run_training_from_param():
        """
        run training from param
        """
        train_file_path = FLAGS.input_training_data_path
        train_tfrecords_path = train_file_path + ".tfrecords"

        eval_file_path = FLAGS.input_validation_data_path
        eval_tfrecords_path = eval_file_path + ".tfrecords"

        vocab = DataProcessor.initialize_vocabulary(FLAGS.vocab_path)
        DataProcessor.create_tfrecord(train_file_path, train_tfrecords_path, vocab)
        DataProcessor.create_tfrecord(eval_file_path, eval_tfrecords_path, vocab, is_training=False)

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

            trainer.train()

if __name__ == '__main__':
    ModelTrainer.run_training_from_param()
