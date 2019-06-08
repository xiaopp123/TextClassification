import tensorflow as tf
import numpy as np
import h5py
from utils import *
from textcnn_model import TextCNN

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("cache_file_h5py", "../data/ieee_zhihu_cup/data.h5", "path of training/validation/test data")
tf.app.flags.DEFINE_string("cache_file_pickle", "../data/ieee_zhihu_cup/vocab_label.pik", "paht of vocabulary ad lable files")

tf.app.flags.DEFINE_float("learning_rate", 0.0003, "learning rate")
tf.app.flags.DEFINE_integer("batch_size", 64, "batch size for training/evaluation")
tf.app.flags.DEFINE_integer("decay_steps", 1000, "how many steps before decay learning rate") #不太懂
tf.app.flags.DEFINE_float("decay_rate", 1.0, "Rate of decay for learning rate.") #不太懂
tf.app.flags.DEFINE_string("ckpt_dir", "text_cnn_title_desc_checkpoint/", "checkpoint location for the model")
tf.app.flags.DEFINE_integer("sentence_len", 200, "max sentence length")
tf.app.flags.DEFINE_integer("embed_size", 128, "embedding size")
tf.app.flags.DEFINE_boolean("is_training_flag", True, "is training.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_integer("num_epochs", 10, "number of epochs to run.")
tf.app.flags.DEFINE_integer("validate_every", 1, "Validate every validate_every epochs.") #每10轮做一次验证
tf.app.flags.DEFINE_boolean("use_embedding",False,"whether to use embedding or not.")
tf.app.flags.DEFINE_integer("num_filters", 128, "number of filters") #256--->512
tf.app.flags.DEFINE_string("word2vec_model_path","word2vec-title-desc.bin","word2vec's vocabulary and vectors")
tf.app.flags.DEFINE_string("name_scope","cnn","name scope value.")
tf.app.flags.DEFINE_boolean("multi_label_flag", True, "use multi label or single label.")

filter_sizes=[6,7,8]

def main(_):
    #1.load data
    word2index, label2index, train_x, train_y, valid_x, valid_y, test_x, test_y =\
       load_data(FLAGS.cache_file_h5py, FLAGS.cache_file_pickle)
    vocab_size = len(word2index)
    num_classes = len(label2index)
    print(train_y[0:3])

    num_examples, FLAGS.sentence_len = train_x.shape

    #2 create session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        textCNN = TextCNN(filter_sizes, FLAGS.num_filters, num_classes, FLAGS.learning_rate, \
                          FLAGS.batch_size, FLAGS.decay_steps, FLAGS.decay_rate,FLAGS.sentence_len,\
                          vocab_size, FLAGS.embed_size, multi_label_flag = FLAGS.multi_label_flag)

        saver = tf.train.Saver()
        if os.path.exists(FLAGS.ckpt_dir + "checkpoint"):
            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.ckpt_dir))
        else:
            sess.run(tf.global_variables_initializer())

        if FLAGS.use_embedding:
            index2word = {v: k for k, v in word2index.items()}
            #assign_pretrained_word_embedding(sess, index2word, vocab_size, textCNN, FLAG.word2vec_model_path)

        curr_epoch = sess.run(textCNN.epoch_step)

        #3 feed data and training
        number_of_training_data = len(train_x)
        batch_size = FLAGS.batch_size
        iteration = 0
        for epoch in range(curr_epoch, FLAGS.num_epochs):
            loss, counter = 0.0, 0
            for start, end in zip(range(0, number_of_training_data, batch_size), \
                                  range(batch_size, number_of_training_data, batch_size)):
                iteration = iteration + 1
                feed_dict = { textCNN.input_x: train_x[start: end],
                              textCNN.dropout_keep_prob: 0.8,
                              textCNN.is_training_flag: FLAGS.is_training_flag }
                if not FLAGS.multi_label_flag:
                    feed_dict[textCNN.input_y] = train_y[start: end]
                else:
                    feed_dict[textCNN.input_y_multilabel] = train_y[start: end]

                curr_loss, lr, _ = sess.run([textCNN.loss_val, textCNN.learning_rate, textCNN.train_op], feed_dict)
                loss, counter = loss + curr_loss, counter + 1
                # 每50步打印损失
                if counter % 50 == 0:
                    #do_eval(sess, textCNN, test_x, test_y, num_classes)
                    print("Epoch %d\tBatch %d\tTrain loss:%.3f\tLearning rate:%.5f" % \
                           (epoch, counter, loss/float(counter), lr))

            #每一轮进行验证
            print(epoch, FLAGS.validate_every, (epoch % FLAGS.validate_every == 0))
            if epoch % FLAGS.validate_every==0:
                #eval_loss, f1_score, f1_micro, f1_macro = do_eval(sess, textCNN, text_x, text_y, num_classes)
   #             do_eval(sess, textCNN, text_x, text_y, num_classes)
#save model to checkpoint
                save_path = FLAGS.ckpt_dir + "model.ckpt"
                saver.save(sess, save_path, global_step=epoch)

def do_eval(sess, textCNN, evalX, evalY, num_classes):
    evalX = evalX[0:3000]
    evalY = evalY[0:3000]
    number_examples = len(evalX)
    eval_loss, eval_counter, eval_f1_score, eval_p, eval_r = 0.0, 0, 0.0, 0.0, 0.0
    batch_size = FLAGS.batch_size
    predict = []
    for start, end in zip(range(0, number_examples, batch_size), range(batch_size, number_examples, batch_size)):
        """"""
        feed_dict = {textCNN.input_x: evalX[start:end],\
                      textCNN.input_y_multilabel: evalY[start:end],\
                      textCNN.dropout_keep_prob: 1.0,
                      textCNN.is_training_flag: False}

        current_eval_loss, logits = sess.run(
            [textCNN.loss_val, textCNN.logits], feed_dict)

#bug
        predict = predict.extend(logits[0])
        eval_loss += current_eval_loss
        eval_counter += 1

    if not FLAGS.multi_label_flag:
        predict = [int(ii > 0.5) for ii in predict]
    _, _, f1_macro, f1_micro, _ = fast_f1(predict, eval_y)

def fast_f1(result, predict):
    """f1 score"""
    true_total, r_total, p_total, p, r = 0, 0, 0, 0, 0
    total_list = []
    for trueValue in range(6):
        true_num, recall_num, precision_num = 0, 0, 0
        for index, values in enumerate(result):
            if values == trueValue:
                recall_num += 1
                if values == predict[index]:
                    true_num += 1
            if predict[index] == trueValue:
                precision_num += 1

        R = trun_num / recall_num if recall_num else 0
        P = true_num / precision_num if precision_num else 0
        true_total += true_num
        r_total += recall_num
        p_total += precision_num
        p += P
        r += R
        f1 = (2 * P * R) / (P + R) if (P + R) else 0
        total_list.append([P, R, f1])
    p /= 6
    r /= 6
    micro_r = true_total / r_total
    micro_p = true_total / p_total
    macro_f1 = (2 * p * r) / (p + r) if (p + r) else 0
    micro_f1 = (2 * micro_r * micro_p) / (micro_r + micro_p) if (micro_r + micro_p) else 0

    return p, r, macro_f1, micro_f1, total_lists

if __name__ == "__main__":
    tf.app.run()
