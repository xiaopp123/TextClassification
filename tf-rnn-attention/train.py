import numpy as np
import tensorflow as tf
from keras.datasets import imdb
from utils import get_vocabulary_size
from utils import fit_in_vocabulary
from utils import zero_pad
from utils import batch_generator
from tensorflow.contrib.rnn import GRUCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
from attention import attention
from tqdm import tqdm

NUM_WORDS = 10000
INDEX_FROM = 3
SEQUENCE_LENGTH = 250
EMBEDDING_DIM = 300
HIDDEN_SIZE = 150
ATTENTION_SIZE = 50
KEEP_PROB = 0.8
BATCH_SIZE = 256
NUM_EPOCHS = 3
DELTA = 0.5
MODEL_PATH = './model'

#load data
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=NUM_WORDS, index_from=INDEX_FROM)

#print(type(X_train[0]))

vocabulary_size = get_vocabulary_size(X_train)
X_test = fit_in_vocabulary(X_test, vocabulary_size)
X_train = zero_pad(X_train, SEQUENCE_LENGTH)
X_test = zero_pad(X_test, SEQUENCE_LENGTH)

with tf.name_scope("Inputs"):
    batch_ph = tf.placeholder(tf.int32, [None, SEQUENCE_LENGTH], name="batch_ph")
    target_ph = tf.placeholder(tf.float32, [None], name="target_ph")
    seq_len_ph = tf.placeholder(tf.int32, [None], name="seq_len_ph")
    keep_prob_ph = tf.placeholder(tf.float32, name="keep_prob_ph")

with tf.name_scope("Emebdding_layer"):
    embeddings_var = tf.Variable(tf.random_uniform([vocabulary_size, EMBEDDING_DIM], -1.0, 1.0), trainable=True)
    #print(embeddings_var.shape)
    tf.summary.histogram("embeddings_var", embeddings_var)
    batch_embedded = tf.nn.embedding_lookup(embeddings_var, batch_ph)

#参数如何初始化的
rnn_outputs, _ = bi_rnn(GRUCell(HIDDEN_SIZE), GRUCell(HIDDEN_SIZE),
                        inputs=batch_embedded, sequence_length=seq_len_ph, dtype=tf.float32)
tf.summary.histogram("RNN_outputs", rnn_outputs)

#Attention
with tf.name_scope("Attention_layer"):
    attention_output, alphas = attention(rnn_outputs, ATTENTION_SIZE, return_alphas=True)
    tf.summary.histogram("alphas", alphas)

#Dropout
#(B, D)
drop = tf.nn.dropout(attention_output, keep_prob_ph)

# Fully connected layer
with tf.name_scope("Fully_connected_layer"):
    W = tf.Variable(tf.truncated_normal([HIDDEN_SIZE * 2, 1], stddev=0.1))
    b = tf.Variable(tf.constant(0., shape=[1]))
    #(B, D) * (D, 1) ==> (B, 1)
    y_hat = tf.nn.xw_plus_b(drop, W, b)
    y_hat = tf.squeeze(y_hat)
    tf.summary.histogram("W", W)
    #print(y_hat.shape)

with tf.name_scope("Metrics"):
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_hat, labels=target_ph))
    tf.summary.scalar("loss", loss)
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)

    #
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(tf.sigmoid(y_hat)), target_ph), tf.float32))

    tf.summary.scalar("accuracy", accuracy)

#不太懂
merged = tf.summary.merge_all()

train_batch_generator = batch_generator(X_train, y_train, BATCH_SIZE)
test_batch_generator = batch_generator(X_test, y_test, BATCH_SIZE)

train_writer = tf.summary.FileWriter('./logdir/train', accuracy.graph)
test_writer = tf.summary.FileWriter('./logdir/test', accuracy.graph)

session_conf = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))

saver = tf.train.Saver()

if __name__ == "__main__":
    with tf.Session(config=session_conf) as sess:
        sess.run(tf.global_variables_initializer())
        print("start learning...")
        for epoch in range(NUM_EPOCHS):
            loss_train = 0
            loss_test = 0
            accuracy_train = 0
            accuracy_test = 0
            print("epoch: {}\t".format(epoch), end="")

            # Training
            num_batches = X_train.shape[0] // BATCH_SIZE
            for b in tqdm(range(num_batches)):
               x_batch, y_batch = next(train_batch_generator)
               seq_len = np.array([list(x).index(0) + 1 for x in x_batch])
               # run里面的参数设置多少
               loss_tr, acc, _, summary = sess.run([loss, accuracy, optimizer, merged],
                                                   feed_dict={batch_ph: x_batch,
                                                              target_ph: y_batch,
                                                              seq_len_ph: seq_len,
                                                              keep_prob_ph: KEEP_PROB})
               accuracy_train += acc
               # 损失计算方式
               loss_train = loss_tr * DELTA + loss_train * (1 - DELTA)
               train_writer.add_summary(summary, b + num_batches * epoch)
            accuracy_train /= num_batches

            # Testing
            num_batches = X_test.shape[0] // BATCH_SIZE
            for b in tqdm(range(num_batches)):
                x_batch, y_batch = next(test_batch_generator)
                seq_len = np.array([list(x).index(0) + 1 for x in x_batch])
                #为什么这个可以不用加optimizer
                loss_test_batch, acc, summary = sess.run([loss, accuracy, merged],
                                                         feed_dict={batch_ph: x_batch,
                                                                    target_ph: y_batch,
                                                                    seq_len_ph: seq_len,
                                                                    keep_prob_ph: 1.0}) #测试集不drop out

                accuracy_test += acc
                loss_test += loss_test_batch
                test_writer.add_summary(summary, b + num_batches * epoch)
            accuracy_test /= num_batches
            loss_test /= num_batches

            print("loss: {:.3f}, val_loss: {:.3f}, acc: {:.3f}, val_acc:{:.3f}".format(
                loss_train, loss_test, accuracy_train, accuracy_test
            ))
        train_writer.close()
        test_writer.close()
        saver.save(sess, MODEL_PATH)
        print("finish")
