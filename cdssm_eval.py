"""
Model Evaluator
"""

import tensorflow as tf
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from cdssm_model import ChineseCdssmModel
from cdssm_input import DataProcessor
from cdssm_param import FLAGS

class PairScoring:
    """
    structure to store {query, label, score} pair data
    """
    def __init__(self, query, label, score):
        self.query = query
        self.label = label
        self.score = score

class PrecisionCoveragePoint:
    """
    structure to store p-c point data
    """
    def __init__(self, threshold, coverage, precision):
        self.threshold = threshold
        self.coverage = coverage
        self.precision = precision

class ModelEvaluator:
    """
    evaluator for top-1 intent evaluation
    """
    @staticmethod
    def evaluate_from_list(query_id, label, score, output_scoring_file_path=None, draw_curve=False):
        """
        evaluate metrics from list
        """
        pair_scorings = ModelEvaluator.get_pair_scorings_from_list(query_id, label, score)
        ModelEvaluator.dump_pair_scorings_to_file(pair_scorings, output_scoring_file_path)
        return ModelEvaluator.evaluate_pair_scorings(pair_scorings, draw_curve)

    @staticmethod
    def evaluate_from_param(output_scoring_file_path=None, draw_curve=False):
        """
        evaluate metrics from param
        """
        pair_scorings = ModelEvaluator.get_pair_scorings_from_param()
        ModelEvaluator.dump_pair_scorings_to_file(pair_scorings, output_scoring_file_path)
        return ModelEvaluator.evaluate_pair_scorings(pair_scorings, draw_curve)

    @staticmethod
    def evaluate_from_file(input_scoring_file_path, draw_curve=False):
        """
        evaluate metrics from file with scoring
        """
        pair_scorings = ModelEvaluator.get_pair_scorings_from_file(input_scoring_file_path)
        return ModelEvaluator.evaluate_pair_scorings(pair_scorings, draw_curve)

    @staticmethod
    def get_pair_scorings_from_list(query_id, label, score):
        """
        get pair scorings from list
        """
        pair_scorings = []
        assert len(query_id) == len(label) == len(score)
        for i, qid in enumerate(query_id):
            pair_scoring = PairScoring(qid, label[i], score[i])
            pair_scorings.append(pair_scoring)
        return pair_scorings

    @staticmethod
    def get_pair_scorings_from_param():
        """
        get pair scorings from param
        """
        # create evaluation tfrecords
        eval_file_path = FLAGS.input_validation_data_path
        eval_tfrecords_path = eval_file_path + ".tfrecords"
        vocab = DataProcessor.initialize_vocabulary(FLAGS.vocab_path)
        DataProcessor.create_tfrecord(eval_file_path, eval_tfrecords_path, vocab, is_training=False)

        # define model
        model = ChineseCdssmModel(vocab_size=len(vocab),
                                  embedding_size=FLAGS.embedding_size,
                                  win_size=FLAGS.win_size,
                                  conv_size=FLAGS.conv_size,
                                  dense_size=FLAGS.dense_size,
                                  share_weight=FLAGS.share_weight)

        # build scoring graph
        eval_q_vec = tf.placeholder(tf.int64)
        eval_q_vec_len = tf.placeholder(tf.int64)
        eval_d_vec = tf.placeholder(tf.int64)
        eval_d_vec_len = tf.placeholder(tf.int64)
        eval_qnvec, eval_dnvec = model.inference(eval_q_vec, eval_q_vec_len, eval_d_vec, eval_d_vec_len)
        score_op = tf.reduce_sum(tf.multiply(eval_qnvec, eval_dnvec), axis=1)

        query_id_list = []
        label_list = []
        score_list = []

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        saver = tf.train.Saver()

        with tf.Session(config=config) as session:

            session.run(tf.local_variables_initializer())
            session.run(tf.global_variables_initializer())
            session.run(tf.tables_initializer())

            # Load model to evaluate
            ckpt = tf.train.get_checkpoint_state(FLAGS.input_previous_model_path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(session, ckpt.model_checkpoint_path)
                print("Load model from ", ckpt.model_checkpoint_path)
            else:
                print("No model found in ", ckpt.model_checkpoint_path)
                return

            for record in DataProcessor.load_evaluation_tfrecords(eval_tfrecords_path, FLAGS.eval_batch_size, FLAGS.max_length):
                query_id, q_vec, q_vec_len, d_vec, d_vec_len, label = record
                score = session.run(score_op, feed_dict={eval_q_vec: q_vec,
                                                         eval_q_vec_len: q_vec_len,
                                                         eval_d_vec: d_vec,
                                                         eval_d_vec_len: d_vec_len})
                query_id_list.extend(query_id)
                label_list.extend(label)
                score_list.extend(score)

        return ModelEvaluator.get_pair_scorings_from_list(query_id_list, label_list, score_list)

    @staticmethod
    def get_pair_scorings_from_file(input_scoring_file_path):
        """
        get pair scorings from file
        """
        pair_scorings = []

        if input_scoring_file_path:
            with open(input_scoring_file_path, encoding="utf-8") as f_open:
                for line in f_open:
                    segs = line.strip().split("\t")
                    if len(segs) == 3:
                        query = segs[0]
                        label = int(segs[1])
                        score = float(segs[2])
                        pair_scoring = PairScoring(query, label, score)
                        pair_scorings.append(pair_scoring)

        return pair_scorings

    @staticmethod
    def dump_pair_scorings_to_file(pair_scorings, output_scoring_file_path):
        """
        dump pair scorings to file
        """
        if output_scoring_file_path:
            with open(output_scoring_file_path, 'w', encoding="utf-8") as f_open:
                for _, pair_scoring in enumerate(pair_scorings):
                    f_open.write("%s\t%d\t%f\n" % (pair_scoring.query, pair_scoring.label, pair_scoring.score))

    @staticmethod
    def evaluate_pair_scorings(pair_scorings, draw_curve=False):
        """
        evaluate pair scorings
        """
        query_cnt, precision_coverage_points, precisions, coverages = ModelEvaluator.get_top_1_points(pair_scorings)
        auc = metrics.auc(coverages, precisions)
        precision = ModelEvaluator.get_precision_at_full_coverage(precision_coverage_points)
        if draw_curve:
            ModelEvaluator.draw_precision_coverage_curve(precisions, coverages)
        return query_cnt, auc, precision

    @staticmethod
    def get_top_1_points(pair_scorings):
        """
        get top 1 precision coverage points
        """
        pair_scorings = ModelEvaluator.get_top_1_scorings(pair_scorings)
        query_cnt = len(pair_scorings)
        precision_coverage_points = ModelEvaluator.get_precision_coverage_points(pair_scorings)
        precisions, coverages = ModelEvaluator.get_precision_coverage_nparray(precision_coverage_points)
        return query_cnt, precision_coverage_points, precisions, coverages

    @staticmethod
    def get_top_1_scorings(pair_scorings):
        """
        get top 1 scorings
        """
        query_dict = {}

        for _, pair_scoring in enumerate(pair_scorings):
            if pair_scoring.query not in query_dict or pair_scoring.score > query_dict[pair_scoring.query].score:
                query_dict[pair_scoring.query] = pair_scoring

        return list(query_dict.values())

    @staticmethod
    def get_precision_coverage_points(pair_scorings):
        """
        get precision coverage points
        """
        points = []
        total_cnt = len(pair_scorings)
        pair_scorings.sort(key=lambda x: x.score, reverse=True)
        trigger_cnt = 0
        positive_cnt = 0

        for i in range(0, total_cnt):
            pair_scoring = pair_scorings[i]
            trigger_cnt += 1
            positive_cnt += pair_scoring.label
            threshold = pair_scoring.score
            coverage = trigger_cnt / total_cnt
            precision = positive_cnt / trigger_cnt
            point = PrecisionCoveragePoint(threshold, coverage, precision)
            points.append(point)

        return points

    @staticmethod
    def get_precision_coverage_nparray(precision_coverage_points):
        """
        get precision coverage numpy array
        """
        precisions = [point.precision for point in precision_coverage_points]
        precisions = np.array(precisions)
        coverages = [point.coverage for point in precision_coverage_points]
        coverages = np.array(coverages)
        return precisions, coverages

    @staticmethod
    def get_precision_at_full_coverage(precision_coverage_points):
        """
        get precision at 100% coverage
        """
        point = next((m for m in precision_coverage_points if m.coverage >= 1.0), None)
        if point:
            return point.precision
        else:
            return 0.0

    @staticmethod
    def draw_precision_coverage_curve(precisions, coverages):
        """
        draw 2D p-c curve
        """
        plt.plot(coverages, precisions, '-')
        plt.xlabel("Coverage")
        plt.ylabel("Top-1 Precision")
        plt.title("Chinese CDSSM Model Evaluation")
        plt.grid(True)
        plt.axis([0, 1, 0, 1])
        plt.show()

    @staticmethod
    def evaluate_side_by_side(input_scoring_file_path_1, legend_1, input_scoring_file_path_2, legend_2):
        """
        evaluate top-1 metrics side by side from files
        """
        line_1 = ModelEvaluator.draw_side_by_side_curve(input_scoring_file_path_1, legend_1, 'b-')
        line_2 = ModelEvaluator.draw_side_by_side_curve(input_scoring_file_path_2, legend_2, 'r-')
        plt.legend(handles=[line_1, line_2])
        plt.xlabel("Coverage")
        plt.ylabel("Top-1 Precision")
        plt.title("Chinese CDSSM Model Evaluation")
        plt.grid(True)
        plt.axis([0, 1, 0, 1])
        plt.show()

    @staticmethod
    def draw_side_by_side_curve(input_scoring_file_path, legend, mark='-'):
        """
        draw side by side curves in the same graph
        """
        pair_scorings = ModelEvaluator.get_pair_scorings_from_file(input_scoring_file_path)
        _, precision_coverage_points, precisions, coverages = ModelEvaluator.get_top_1_points(pair_scorings)
        auc = metrics.auc(coverages, precisions)
        precision = ModelEvaluator.get_precision_at_full_coverage(precision_coverage_points)
        legend += (": AUC=%0.4f, Prec@1Cov=%0.4f" % (auc, precision))
        line, = plt.plot(coverages, precisions, mark, label=legend)
        return line

if __name__ == '__main__':
    RESULTS = ModelEvaluator.evaluate_from_param(draw_curve=False)
    print("Top 1 evaluation result: query_cnt = %d, auc = %0.10f, precision_at_full_coverage = %0.10f" % (RESULTS[0], RESULTS[1], RESULTS[2]))
