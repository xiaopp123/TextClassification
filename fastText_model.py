#encoding=utf-8
import tensorflow as tf
import numpy as np

class FastText(object):
    def __init__(self, label_size, learning_rate, batch_size, decay_steps, decay_rate, num_sampled, sentence_len, vocab_size, embed_size, is_training):
        """init all hyperparameter here"""
        #set hyperparameter
        self.label_size = label_size
        self.batch_size = batch_size
        self.num_sampled = num_sampled
        self.sentence_len = sentence_len
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.is_training = is_training
        self.learning_rate = learning_rate

        #add placeholder
        self.sentence = tf.placeholder(tf.int32, [None, self.sentence_len], name="sentence")
        self.labels = tf.placeholder(tf.int32, [None], name="labels")

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
        self.decay_steps, self.decay_rate = decay_steps, decay_rate

        #init weight
        self.instantiate_weights()

        #inference
        self.logits = self.inference()

        if not is_training:
            return
        #loss
        self.loss_val = self.loss()
        self.train_op = self.train()
        self.predictions = tf.argmax(self.logits, axis=1, name="predictions")
        correct_prediction = tf.equal(tf.cast(self.predictions, tf.int32), self.labels)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")


    def instantiate_weights(self):
        """"""""
        self.Embedding = tf.get_variable("Embedding", [self.vocab_size, self.embed_size])
        self.W = tf.get_variable("W", [self.embed_size, self.label_size])
        self.b = tf.get_variable("b", [self.label_size])

    def inference(self):
        # 1.get embedding of words in the sentence
        sentence_embeddings = tf.nn.embedding_lookup(self.Embedding, self.sentence)
        
        # 2.average vectors, to get representation of the sentence
        self.sentence_embeddings = tf.reduce_mean(sentence_embeddings, axis=1)

        # 3.linear classifier layer
        logits = tf.matmul(self.sentence_embeddings, self.W) + self.b
        return logits

    def loss(self, l2_lambda=0.01):
        if self.is_training:
            labels = tf.reshape(self.labels, [-1])
            labels = tf.expand_dims(labels, 1)
            loss = tf.reduce_mean(
                tf.nn.nce_loss(weights=tf.transpose(self.W),
                               biases=self.b,
                               labels=labels,
                               inputs=self.sentence_embeddings,
                               num_sampled=self.num_sampled,
                               num_classes=self.label_size,
                               partition_strategy="div"))
        else:
            labels_one_hot = tf.one_hot(self.labels, self.label_size)
            loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_one_hot, logits=self.logits)
            loss = tf.reduce_sum(loss, axis=1)

        l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
        return loss

    def train(self):
        """based on the loss, use SGD to update parameter"""
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps, self.decay_rate, staircase=True)
        train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step, learning_rate=learning_rate, optimizer="Adam")
        return train_op

 
def test():
    num_classes = 19
    learning_rate = 0.01
    batch_size = 8
    decay_steps = 1000
    decay_rate = 0.9
    sequence_length = 5
    vocab_size = 10000
    embed_size = 100
    is_training = True
    dropout_keep_prob = 1
    fastText = FastText(num_classes, learning_rate, batch_size, decay_steps, decay_rate, 5, sequence_length, vocab_size, embed_size, is_training)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(100):
            input_x = np.zeros((batch_size, sequence_length), dtype=np.int32)
            input_y = np.array([1, 0, 1, 1, 1, 2, 1, 1], dtype=np.int32)
            loss, acc, predict, _ = sess.run([fastText.loss_val, fastText.accuracy, fastText.predictions, fastText.train_op],
                                             feed_dict={fastText.sentence: input_x,
                                                        fastText.labels: input_y
                                             })
            print("loss: {}, acc: {}, label: {}".format(loss, acc, predict))
test()
print("ended...")
