"""
Model Predictor
"""

import time
import tensorflow as tf
from cdssm_model import ChineseCdssmModel
from cdssm_input import DataProcessor
from cdssm_param import FLAGS

class ModelPredictor:
    """
    predict text embedding
    """

    def __init__(self):
        # initialize vocab and model
        self.vocab = DataProcessor.initialize_vocabulary(FLAGS.vocab_path)
        self.vocab_size = len(self.vocab)
        model = ChineseCdssmModel(vocab_size=self.vocab_size,
                                  embedding_size=FLAGS.embedding_size,
                                  win_size=FLAGS.win_size,
                                  conv_size=FLAGS.conv_size,
                                  dense_size=FLAGS.dense_size,
                                  share_weight=FLAGS.share_weight)

        # build predict graph
        self.text_vec = tf.placeholder(tf.int64)
        self.text_vec_len = tf.placeholder(tf.int64)
        self.text_embedding = model.forward_propagation(self.text_vec, self.text_vec_len)

        # initialize session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        saver = tf.train.Saver()
        # saver = tf.train.import_meta_graph('data/finalmodel.ckpt/cdssm_model-114000.meta')
        session = tf.Session(config=config)
        session.run(tf.local_variables_initializer())
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())

        # Load model to evaluate
        ckpt = tf.train.get_checkpoint_state(FLAGS.input_previous_model_path)
        if ckpt and ckpt.model_checkpoint_path:
            # saver.restore(session, ckpt.model_checkpoint_path)
            saver.restore(session, 'data/finalmodel.ckpt/cdssm_model-11000')
            print("Load model from ", ckpt.model_checkpoint_path)
        else:
            raise Exception("No model found in {}".format(ckpt.model_checkpoint_path))

        self.session = session
        self.max_length = FLAGS.max_length
        self.predict("人工智能")

    def batch_predict(self, text_list):
        """
        predict text embeddings for a list of text
        """
        text_id_list = []
        for text in text_list:
            text_id = DataProcessor.vectorize(text, self.vocab, self.vocab_size)
            text_id_list.append(text_id)

        text_vec, text_vec_len = DataProcessor.align_vector_batch(text_id_list, self.max_length)
        text_embedding = self.session.run(self.text_embedding, feed_dict={self.text_vec: text_vec, self.text_vec_len: text_vec_len})
        return text_embedding

    def predict(self, text):
        """
        predict text embedding for single text
        """
        text_vec = DataProcessor.vectorize(text, self.vocab, self.vocab_size)
        text_vec_len = len(text_vec)
        if text_vec_len > self.max_length:
            text_vec_len = self.max_length
            text_vec = text_vec[:self.max_length]

        text_embedding = self.session.run(self.text_embedding, feed_dict={self.text_vec: [text_vec], self.text_vec_len: [text_vec_len]})
        return text_embedding[0]

    @staticmethod
    def test_performance(max_text_count=10000):
        """
        test predicting performance
        """
        predictor = ModelPredictor()

        text_list = []
        with open(FLAGS.input_training_data_path, encoding='utf-8', mode='rt') as reader:
            for line in reader:
                segs = line.strip().split("\t")
                if len(segs) < 2:
                    continue
                query = segs[0]
                text_list.append(query)
                if len(text_list) >= max_text_count:
                    break

        text_count = len(text_list)
        start_time = time.time()
        for text in text_list:
            _ = predictor.predict(text)
        end_time = time.time()
        duration = end_time - start_time
        qps = text_count / duration
        mspq = 1000 * duration / text_count
        print("Processed %d queries in %0.1fs: %0.1f query/s, %0.1f ms/query" % (text_count, duration, qps, mspq))

if __name__ == '__main__':
    # ModelPredictor.test_performance()
    predictor = ModelPredictor()
    print(predictor.predict('人工智能'))
