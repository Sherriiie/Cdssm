"""
Tensorflow FLAGS param
"""

import tensorflow as tf
import os

root_dir = os.path.abspath('.') + '\\data'
# root_dir = '/home/work/xueyun/BinaryClassificationStackCdssm/data'
print('root data directory: ', root_dir)

tf.app.flags.DEFINE_string('input_training_data_path', root_dir + '/train.txt', 'training data path')
tf.app.flags.DEFINE_string('input_validation_data_path', root_dir + '/evaluation.txt', 'validation data path')
tf.app.flags.DEFINE_string('input_previous_model_path', root_dir + '\\binary_data\\checkpoint_train', 'path of previous model')
tf.app.flags.DEFINE_string('output_model_path', root_dir + '/binary_data/checkpoint_train', 'path to save model')
tf.app.flags.DEFINE_string('log_dir', root_dir + '/log_folder', 'folder to save checkpoints')
tf.app.flags.DEFINE_string('vocab_path', root_dir + '/vocabulary.txt', 'path of vocab dict')

tf.app.flags.DEFINE_integer('embedding_size', 1024, 'the dense embedding layer size')        # if = -1, then one-hot embedding
tf.app.flags.DEFINE_integer('win_size', 3, 'window size of convolution')
tf.app.flags.DEFINE_integer('conv_size', 1024, 'the convolution and max pooling layer size(kernal numbers)')
tf.app.flags.DEFINE_integer('dense_size', 128, 'the fully connect dense layer size')
tf.app.flags.DEFINE_bool('share_weight', True, 'whether to share weight between query and doc network')
tf.app.flags.DEFINE_integer('log_frequency', 100, 'log frequency')
tf.app.flags.DEFINE_integer('checkpoint_frequency', 500, 'steps to save checkpoint') #1000,500
tf.app.flags.DEFINE_integer('num_epochs', 1000, 'num of epochs to train')  #30
tf.app.flags.DEFINE_integer('train_batch_size', 163, 'batch size of training procedure')    #256  163=entities' number plus 1
tf.app.flags.DEFINE_integer('eval_batch_size', 1024, 'batch size when evaluation')
tf.app.flags.DEFINE_integer('max_length', 250, 'query will be truncated if token count is larger than max_length')#256
tf.app.flags.DEFINE_integer('num_threads', 2, 'read thread for reading training data')
tf.app.flags.DEFINE_integer('negative_sample', 50, 'negative sample count')
tf.app.flags.DEFINE_float('softmax_gamma', 10.0, 'softmax gamma for loss function')
tf.app.flags.DEFINE_string('optimizer', 'adam', 'which optimizer to use')
tf.app.flags.DEFINE_float('learning_rate', 0.0001, 'learning rate to train the model')
tf.app.flags.DEFINE_bool('enable_early_stop', False, 'whether to use early stop')
tf.app.flags.DEFINE_integer('early_stop_steps', 30, 'How many bad checks to trigger early stop')

FLAGS = tf.app.flags.FLAGS


if __name__ == '__main__':
    root_dir = os.path.abspath('.') + '\data'
    print('root directory1: ', root_dir)


# tf.app.flags.DEFINE_string('vocab_path', '/home/work/fantasy/data/BusinessAIChina/Customers/CMCC/ChineseCdssm/CharVersion/vocab_merged.txt', 'path of vocab dict')
# tf.app.flags.DEFINE_string('input_training_data_path', '/home/work/fantasy/data/BusinessAIChina/Customers/CMCC/ChineseCdssm/CharVersion/source_shuffle.tsv', 'training data path')
# tf.app.flags.DEFINE_string('input_validation_data_path', '/home/work/fantasy/data/BusinessAIChina/Customers/CMCC/ChineseCdssm/CharVersion/label.5.evaluation_.tsv', 'validation data path')
# tf.app.flags.DEFINE_string('input_previous_model_path', '/home/work/fantasy/data/BusinessAIChina/Customers/CMCC/ChineseCdssm/CharVersion/finalmodel.ckpt', 'path of previous model')
# tf.app.flags.DEFINE_string('output_model_path', '/home/work/fantasy/data/BusinessAIChina/Customers/CMCC/ChineseCdssm/CharVersion/finalmodel.ckpt', 'path to save model')
# tf.app.flags.DEFINE_string('log_dir', '/home/work/fantasy/data/BusinessAIChina/Customers/CMCC/ChineseCdssm/CharVersion/log_folder', 'folder to save checkpoints')
#
# tf.app.flags.DEFINE_integer('embedding_size', 512, 'the dense embedding layer size')
# tf.app.flags.DEFINE_integer('win_size', 3, 'window size of convolution')
# tf.app.flags.DEFINE_integer('conv_size', 512, 'the convolution and max pooling layer size(kernal numbers)')
# tf.app.flags.DEFINE_integer('dense_size', 64, 'the fully connect dense layer size')
# tf.app.flags.DEFINE_bool('share_weight', True, 'whether to share weight between query and doc network')
# tf.app.flags.DEFINE_integer('log_frequency', 100, 'log frequency')
# tf.app.flags.DEFINE_integer('checkpoint_frequency', 1000, 'steps to save checkpoint')
# tf.app.flags.DEFINE_integer('num_epochs', 30, 'num of epochs to train')
# tf.app.flags.DEFINE_integer('train_batch_size', 256, 'batch size of training procedure')
# tf.app.flags.DEFINE_integer('eval_batch_size', 1024, 'batch size when evaluation')
# tf.app.flags.DEFINE_integer('max_length', 128, 'query will be truncated if token count is larger than max_length')
# tf.app.flags.DEFINE_integer('num_threads', 2, 'read thread for reading training data')
# tf.app.flags.DEFINE_integer('negative_sample', 50, 'negative sample count')
# tf.app.flags.DEFINE_float('softmax_gamma', 10.0, 'softmax gamma for loss function')
# tf.app.flags.DEFINE_string('optimizer', 'adam', 'which optimizer to use')
# tf.app.flags.DEFINE_float('learning_rate', 0.001, 'learning rate to train the model')
# tf.app.flags.DEFINE_bool('enable_early_stop', False, 'whether to use early stop')
# tf.app.flags.DEFINE_integer('early_stop_steps', 30, 'How many bad checks to trigger early stop')
#
# FLAGS = tf.app.flags.FLAGS
