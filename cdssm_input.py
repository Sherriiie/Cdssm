"""
DataProcessor
"""

import os
import tensorflow as tf
import random


class DataProcessor:
    """
    process cdssm model training/evaluation data
    """
    def __init__(self):
        self.data_path_random=''
        self.data_path_eval=''
        # self.data_path = data_path
        self.positive_pairs = []
        self.queries = []
        self.queries_vec = []
        self.entities = []
        self.entities_vec = []
        self.entities_neg_vec = []
        self.entities_neg_vec = []
        self.counter = 0  # for train
        self.total_item_count = 0  # positive pairs

        self.eval_positive_pairs = []
        self.eval_queries = []
        self.eval_queries_vec = []
        self.eval_entities = []
        self.eval_entities_vec = []

        self.eval_counter = 0  # for eval
        self.eval_total_item_count = 0
        # no dup in entities_neg
        # print('there are %d entities in neg no dup'%len(self.entities_neg))
        self.entities_neg = []
        self.entities_neg_vec = []
        # write entities_neg to file
        self.flag = True
        self.vocab_size = 0
        self.revocab_dict={}


    def get_init(self, data_path_train, data_path_eval, vocab, vocab_size, max_length,revocab_dict):
        self.data_path_train = data_path_train
        self.data_path_eval = data_path_eval
        # self.data_path = data_path
        self.vocab_size = vocab_size
        self.counter = 0  # for train
        self.total_item_count = 0  # positive pairs
        self.read_train_data(vocab, vocab_size, max_length)  # read data into system
        self.eval_counter = 0  # for eval
        self.eval_total_item_count = 0
        self.read_eval_data(vocab, vocab_size, max_length)
        # no dup in entities_neg
        self.entities_neg = list(set(self.entities))
        print('There are %d entities in neg no dup, writing into file <data/binary_data/entities_evaluation.txt>' % len(self.entities_neg))
        # write entities_neg to file
        self.write_to_file(self.entities_neg)
        self.revocab_dict = revocab_dict

    def write_to_file(self, entities):
        file_write = open('data/binary_data/entities_evaluation.txt', encoding='utf-8', mode='w')
        temp = [ent.replace(' ', '') for ent in entities]
        file_write.write(' '.join(temp))
        file_write.flush()

    def read_train_data(self, vocab, vocab_size, max_length):
        with open(self.data_path_train, encoding='utf-8', mode='r') as file_reader:
            for line in file_reader:
                line = line.strip()
                self.positive_pairs.append(line)
                parts = line.split('\t')
                self.queries.append(parts[0])
                self.entities.append(parts[1])
                query = parts[0]
                ent = parts[1]
                query_vec = DataProcessor.vectorize(query, vocab, vocab_size)
                ent_vec = DataProcessor.vectorize(ent, vocab, vocab_size)
                if (len(query_vec) >= max_length):
                    query_vec = query_vec[0:max_length]
                else:
                    query_vec += [self.vocab_size]*(max_length-len(query_vec))
                if (len(ent_vec) >= max_length):
                    ent_vec = ent_vec[0:max_length]
                else:
                    ent_vec += [self.vocab_size]*(max_length - len(ent_vec))
                self.queries_vec.append(query_vec)
                self.entities_vec.append(ent_vec)
            self.total_item_count = len(self.positive_pairs)
            self.entities_neg = list(set(self.entities))
            for ent in self.entities_neg:
                ent_vec = DataProcessor.vectorize(ent, vocab, vocab_size)
                if (len(ent_vec) >= max_length):
                    ent_vec = ent_vec[0:max_length]
                else:
                    ent_vec += [self.vocab_size]*(max_length - len(ent_vec))
                self.entities_neg_vec.append(ent_vec)
            print('==== read training data into model finished')

    def read_eval_data(self, vocab, vocab_size, max_length):
        with open(self.data_path_eval, encoding='utf-8', mode='r') as file_reader:
            for line in file_reader:
                line = line.strip()
                self.eval_positive_pairs.append(line)
                parts = line.split('\t')
                self.eval_queries.append(parts[0])
                self.eval_entities.append(parts[1])
                eval_query = parts[0]
                eval_ent = parts[1]
                eval_query_vec = DataProcessor.vectorize(eval_query, vocab, vocab_size)
                eval_ent_vec = DataProcessor.vectorize(eval_ent, vocab, vocab_size)
                if (len(eval_query_vec) >= max_length):
                    eval_query_vec = eval_query_vec[0:max_length]
                else:
                    eval_query_vec += [self.vocab_size]*(max_length-len(eval_query_vec))
                if (len(eval_ent_vec) >= max_length):
                    eval_ent_vec = eval_ent_vec[0:max_length]
                else:
                    eval_ent_vec += [self.vocab_size]*(max_length - len(eval_ent_vec))
                self.eval_queries_vec.append(eval_query_vec)
                self.eval_entities_vec.append(eval_ent_vec)
            self.eval_total_item_count = len(self.eval_positive_pairs)
            print('==== read eval data finished, total count = ',  self.eval_total_item_count)

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
            vocab_dict = dict([(x, y) for (y, x) in enumerate(data_list)])           # (char, index)
            revocab_dict = dict([(x, y) for (x, y) in enumerate(data_list)])           # (index, char)
            return vocab_dict, revocab_dict
        else:
            raise ValueError('Vocabulary file {} not found.'.format(vocab_file_path))

    @staticmethod
    def vectorize(text, vocab, default):
        """
        vectorize text to word ids based on vocab
        """
        return [vocab.get(word, default) for word in text.split(' ')]


    def devectorize(self, index, default):
        """
        vectorize text to word ids based on vocab
        """
        return [self.revocab_dict.get(idx, default) for idx in index]

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
        print('query_vec', query_vec.get_shape(), query_vec_length.get_shape())
        print('doc_vec', doc_vec.get_shape())
        min_after_dequeue = 10 * batch_size
        capacity = min_after_dequeue + 3 * batch_size
        return tf.train.shuffle_batch([query_vec, query_vec_length, doc_vec, doc_vec_length],
                                      batch_size=batch_size,
                                      capacity=capacity,
                                      min_after_dequeue=min_after_dequeue,
                                      num_threads=num_threads,
                                      allow_smaller_final_batch=True)

    def next_batch_train_random(self, batch_size):
        # Author: Sherrie
        b_query_vec = []
        b_query_length = []
        b_entity_vec = []
        b_entity_length = []
        b_label = []
        for i in range(batch_size):
            if self.flag == True:
                self.flag = not self.flag
                b_query_vec.append(self.queries_vec[self.counter])
                b_query_length.append(len(self.queries_vec[self.counter]))
                b_entity_vec.append((self.entities_vec[self.counter]))
                b_entity_length.append(len(self.entities_vec[self.counter]))
                b_label.append(1)
                self.counter += 1
                self.counter = self.counter % self.total_item_count
            else:
                self.flag = not self.flag
                b_query_vec.append(self.queries_vec[self.counter])
                b_query_length.append(len(self.queries_vec[self.counter]))
                index = random.randint(0, self.total_item_count - 1)
                while (self.entities_vec[index] == self.entities_vec[self.counter]):
                    index = random.randint(0, self.total_item_count - 1)
                b_entity_vec.append((self.entities_vec[index]))
                b_entity_length.append(len(self.entities_vec[self.counter]))
                b_label.append(0)
        return [b_query_vec, b_query_length, b_entity_vec, b_entity_length, b_label]

    def next_batch_evaluation(self):
        b_query_vec = []
        b_query_length = []
        b_entity_vec = []
        b_entity_length = []
        b_label = []
        entities_neg_vec = self.entities_neg_vec.copy()

        # add the postive sample first, then add all entities as negative, so the total number is +1
        b_query_vec.append(self.eval_queries_vec[self.eval_counter])
        b_query_length.append(len(self.eval_queries_vec[self.eval_counter]))
        b_entity_vec.append(self.eval_entities_vec[self.eval_counter])
        b_entity_length.append(len(self.eval_entities_vec[self.eval_counter]))
        b_label.append(1)
        if (len(entities_neg_vec) != 162):
            print('bug')
        # add all other entities as negative samples
        for i in range(len(entities_neg_vec)):
            b_query_vec.append(self.eval_queries_vec[self.eval_counter])
            b_query_length.append(len(self.eval_queries_vec[self.eval_counter]))
            b_entity_vec.append(entities_neg_vec[i])
            b_entity_length.append(len(entities_neg_vec[i]))
            b_label.append(1)
        # entities_neg.append(self.eval_entities[self.eval_counter])
        self.eval_counter += 1
        self.eval_counter = self.eval_counter % self.eval_total_item_count
        return [b_query_vec, b_query_length, b_entity_vec, b_entity_length, b_label]

    def random_index(self, count):
        index = [i for i in range(self.eval_total_item_count)]
        random.shuffle(index)
        return index[:count]

    def random_batch_evaluation(self, count):
        batches = []
        # get the index of random
        random_index_list = self.random_index(count)
        for i in range(count):
            b_query_vec = []
            b_query_length = []
            b_entity_vec = []
            b_entity_length = []
            b_label = []
            entities_neg_vec = self.entities_neg_vec.copy()

            # add the postive sample first, then add all entities as negative, so the total number is +1
            # random_index = random.randint(0, self.eval_total_item_count-1)          # self.eval_counter
            random_index = random_index_list[i]
            b_query_vec.append(self.eval_queries_vec[random_index])
            b_query_length.append(len(self.eval_queries_vec[random_index]))
            b_entity_vec.append(self.eval_entities_vec[random_index])
            b_entity_length.append(len(self.eval_entities_vec[random_index]))
            b_label.append(1)
            if (len(entities_neg_vec) != 162):
                print('bug')
            # add all other entities as negative samples
            for i in range(len(entities_neg_vec)):
                b_query_vec.append(self.eval_queries_vec[random_index])
                b_query_length.append(len(self.eval_queries_vec[random_index]))
                b_entity_vec.append(entities_neg_vec[i])
                b_entity_length.append(len(entities_neg_vec[i]))
                b_label.append(1)
            batches.append([b_query_vec, b_query_length, b_entity_vec, b_entity_length, b_label])
            # return [b_query_vec, b_query_length, b_entity_vec, b_entity_length, b_label]
        return batches

    # no dup
    def random_batch_evaluation2(self, count):
        batches = []
        qa = []
        # get the index of random
        random_index_list = self.random_index(count)
        for i in range(count):
            b_query_vec = []
            b_query_length = []
            b_entity_vec = []
            b_entity_length = []
            b_label = []
            entities_neg_vec = self.entities_neg_vec.copy()

            # add the postive sample first, then add all entities as negative, so the total number is +1
            # random_index = random.randint(0, self.eval_total_item_count-1)          # self.eval_counter
            random_index = random_index_list[i]
            # b_query_vec.append(self.eval_queries_vec[random_index])
            # b_query_length.append(len(self.eval_queries_vec[random_index]))
            # b_entity_vec.append(self.eval_entities_vec[random_index])
            # b_entity_length.append(len(self.eval_entities_vec[random_index]))
            # b_label.append(1)
            if (len(entities_neg_vec) != 162):
                print('bug')
            # add all other entities as negative samples
            for i in range(len(entities_neg_vec)):
                b_query_vec.append(self.eval_queries_vec[random_index])
                b_query_length.append(len(self.eval_queries_vec[random_index]))
                b_entity_vec.append(entities_neg_vec[i])
                b_entity_length.append(len(entities_neg_vec[i]))
                b_label.append(1)
            batches.append([b_query_vec, b_query_length, b_entity_vec, b_entity_length, b_label])
            qa.append([self.eval_queries_vec[random_index], self.eval_entities_vec[random_index]])
            # return [b_query_vec, b_query_length, b_entity_vec, b_entity_length, b_label]
        return batches, qa

    # get entity name by index
    def get_entity_by_index(self, entity_index):
        if 0 not in entity_index:
            return [self.entities_neg[idx-1] for idx in entity_index]

    def get_entity_by_index2(self, entity_index):
        # if 0 not in entity_index:
        return [self.entities_neg[idx] for idx in entity_index]

    def get_char_by_index(self, index):
        return [ self.initialize_vocabulary() for idx in index]

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
        print("parse_feature: feature.shape()", feature.get_shape())
        feature = tf.sparse_to_dense(sparse_indices=feature.indices[:max_length], output_shape=[max_length],
                                     sparse_values=feature.values[:max_length], default_value=0)
        return feature, feature_length
