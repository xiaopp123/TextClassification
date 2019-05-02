from train import *

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, MODEL_PATH)

    x_batch_test, y_batch_test = X_test[:1], y_test[:1]
    #print(x_batch_test)
    #print(y_batch_test)
    seq_len_test = np.array([list(x).index(0) + 1 for x in x_batch_test])
    alphas_test = sess.run([alphas], feed_dict={batch_ph: x_batch_test,
                                                target_ph: y_batch_test,
                                                seq_len_ph: seq_len_test,
                                                keep_prob_ph: 1.0 })
alphas_values = alphas_test[0][0]

word_index = imdb.get_word_index()
word_index = {word: index + INDEX_FROM for word, index in word_index.items()}
word_index[":PAD:"] = 0
word_index[":START:"] = 1
word_index[":UNK:"] = 2

index_word = {index: word for word, index in word_index.items()}

words = list(map(index_word.get, x_batch_test[0]))

#save visualization as html

with open("visualization.html", "w") as html_file:
    for word, alpha in zip(words, alphas_values / alphas_values.max()):
        if word == ":START:":
            continue
        elif word == ":PAD:":
            break
        html_file.write('<font style="background: rgba(255, 255, 0, %f)">%s</font>\n' % (alpha, word))
