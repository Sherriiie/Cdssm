"""
DataProcessor
"""

import os
import tensorflow as tf


class DataProcessor:
    """
    process cdssm model training/evaluation data
    """

    @staticmethod
    def get_vocabulary_size(vocab_file_path):
        """
        get vocabulary size
        """
        vocab_size = 0
        with open(vocab_file_path, encoding='utf-8', mode='rt') as vocab_file:
            for line in vocab_file:
                if line.strip():
                    vocab_size += 1
        return vocab_size

    @staticmethod
    def initialize_vocabulary(vocab_file_path):
        """
        load vocabulary from file
        """
        if os.path.exists(vocab_file_path):
            data_list = []
            with open(vocab_file_path, encoding='utf-8', mode='rt') as vocab_file:
                for line in vocab_file:
                    if line.strip():
                        data_list.append(line.strip())
            vocab = dict([(x, y) for (y, x) in enumerate(data_list)])
            return vocab
        else:
            raise ValueError('Vocabulary file {} not found.'.format(vocab_file_path))

    @staticmethod
    def vectorize(text, vocab, default):
        """
        vectorize text to word ids based on vocab
        """
        return [vocab.get(word, default) for word in text.split(' ')]

    @staticmethod
    def create_tfrecord(data_file_path, tfrecords_file_path, vocab, is_training=True):
        """
        vectorize training data and convert to tfrecords
        """
        vocab_size = len(vocab)
        query_to_id = {}
        current_id = 0

        with open(data_file_path, encoding='utf-8', mode='rt') as reader, tf.python_io.TFRecordWriter(tfrecords_file_path) as writer:
            for line in reader:
                segs = line.strip().split("\t")
                if len(segs) < 2:
                    continue

                query = segs[0]
                doc = segs[1]
                query_vec = DataProcessor.vectorize(query, vocab, vocab_size)
                doc_vec = DataProcessor.vectorize(doc, vocab, vocab_size)

                if is_training:
                    example = tf.train.Example(features=tf.train.Features(feature={
                        'query_vec': tf.train.Feature(int64_list=tf.train.Int64List(value=query_vec)),
                        'doc_vec': tf.train.Feature(int64_list=tf.train.Int64List(value=doc_vec)),
                    }))
                else:
                    label = int(segs[2])
                    if not query in query_to_id:
                        query_to_id[query] = current_id
                        current_id += 1
                    query_id = query_to_id[query]
                    example = tf.train.Example(features=tf.train.Features(feature={
                        'query_id': tf.train.Feature(int64_list=tf.train.Int64List(value=[query_id])),
                        'query_vec': tf.train.Feature(int64_list=tf.train.Int64List(value=query_vec)),
                        'doc_vec': tf.train.Feature(int64_list=tf.train.Int64List(value=doc_vec)),
                        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
                    }))

                writer.write(example.SerializeToString())

    @staticmethod
    def load_training_tfrecords(tfrecords_file_path, num_epochs, batch_size, max_length, num_threads):
        """
        load training data from tfrecords
        """
        filename_queue = tf.train.string_input_producer([tfrecords_file_path], num_epochs=num_epochs)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        feature_configs = {
            'query_vec': tf.VarLenFeature(dtype=tf.int64),
            'doc_vec': tf.VarLenFeature(dtype=tf.int64),
        }
        features = tf.parse_single_example(serialized_example, features=feature_configs)
        query_vec, query_vec_length = DataProcessor.parse_feature(features['query_vec'], max_length)
        doc_vec, doc_vec_length = DataProcessor.parse_feature(features['doc_vec'], max_length)
        min_after_dequeue = 10 * batch_size
        capacity = min_after_dequeue + 3 * batch_size
        return tf.train.shuffle_batch([query_vec, query_vec_length, doc_vec, doc_vec_length],
                                      batch_size=batch_size,
                                      capacity=capacity,
                                      min_after_dequeue=min_after_dequeue,
                                      num_threads=num_threads,
                                      allow_smaller_final_batch=True)

    @staticmethod
    def load_evaluation_tfrecords(tfrecords_file_path, batch_size, max_length):
        """
        load evaluation data from tfrecords
        """
        query_id_batch = []
        query_vec_batch = []
        doc_vec_batch = []
        label_batch = []
        cnt = 0

        for serialized_example in tf.python_io.tf_record_iterator(tfrecords_file_path):
            example = tf.train.Example()
            example.ParseFromString(serialized_example)

            query_id = example.features.feature['query_id'].int64_list.value[0]
            query_vec = example.features.feature['query_vec'].int64_list.value
            doc_vec = example.features.feature['doc_vec'].int64_list.value
            label = example.features.feature['label'].int64_list.value[0]

            query_id_batch.append(query_id)
            query_vec_batch.append(query_vec)
            doc_vec_batch.append(doc_vec)
            label_batch.append(label)

            cnt += 1
            if cnt % batch_size == 0:
                query_vec_batch, query_vec_length_batch = DataProcessor.align_vector_batch(query_vec_batch, max_length)
                doc_vec_batch, doc_vec_length_batch = DataProcessor.align_vector_batch(doc_vec_batch, max_length)
                yield query_id_batch, query_vec_batch, query_vec_length_batch, doc_vec_batch, doc_vec_length_batch, label_batch

                query_id_batch = []
                query_vec_batch = []
                doc_vec_batch = []
                label_batch = []

        if query_id_batch:
            query_vec_batch, query_vec_length_batch = DataProcessor.align_vector_batch(query_vec_batch, max_length)
            doc_vec_batch, doc_vec_length_batch = DataProcessor.align_vector_batch(doc_vec_batch, max_length)
            yield query_id_batch, query_vec_batch, query_vec_length_batch, doc_vec_batch, doc_vec_length_batch, label_batch

    @staticmethod
    def align_vector_batch(vector_batch, max_length):
        """
        align vector batch to max length
        """
        result_vector_batch = []
        result_vector_length_batch = []

        batch_max_length = max(len(vector) for vector in vector_batch)
        align_length = min(batch_max_length, max_length)

        for vector in vector_batch:
            result_vector, result_vector_length = DataProcessor.align_vector(vector, align_length)
            result_vector_batch.append(result_vector)
            result_vector_length_batch.append(result_vector_length)

        return result_vector_batch, result_vector_length_batch

    @staticmethod
    def align_vector(vector, max_length):
        """
        align vector to max length
        """
        vector_length = len(vector)
        if vector_length > max_length:
            vector_length = max_length
            vector = vector[:max_length]
        else:
            vector.extend([0 for _ in range(max_length - vector_length)])
        return vector, vector_length

    @staticmethod
    def parse_feature(feature, max_length):
        """
        deserialize feature
        """
        feature_length = tf.minimum(feature.dense_shape[0], tf.constant(max_length, tf.int64))
        feature = tf.sparse_to_dense(sparse_indices=feature.indices[:max_length], output_shape=[max_length],
                                     sparse_values=feature.values[:max_length], default_value=0)
        return feature, feature_length
