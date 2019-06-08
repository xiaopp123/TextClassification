import os
import h5py
import pickle

def load_data(cache_file_h5py, cache_file_pickle):
    if not os.path.exists(cache_file_h5py) or not os.path.exists(cache_file_pickle):
        raise RuntimeError("")
    f_data = h5py.File(cache_file_h5py, "r")
    train_x = f_data['train_X']
    train_y = f_data['train_Y']
    valid_x = f_data['vaild_X'] # np.array(
    valid_y = f_data['valid_Y'] # np.array(
    test_x = f_data['test_X'] # np.array(
    test_y = f_data['test_Y'] # np.array(

    word2index, label2index = None, None
    with open(cache_file_pickle, "rb") as data_f_pickle:
        word2index, label2index = pickle.load(data_f_pickle)
    
    return word2index, label2index, train_x, train_y, valid_x, valid_y, test_x, test_y

def assign_pretrained_word_embedding(sess, vocabulary_index2word, vocab_size, textCNN, word2vec_model_path):
    import word2vec # we put import here so that many people who do not use word2vec do not need to install this package. you can move import to the beginning of this file.
    print("using pre-trained word emebedding.started.word2vec_model_path:",word2vec_model_path)
    word2vec_model = word2vec.load(word2vec_model_path, kind='bin')
    word2vec_dict = {}
    for word, vector in zip(word2vec_model.vocab, word2vec_model.vectors):
        word2vec_dict[word] = vector
    word_embedding_2dlist = [[]] * vocab_size  # create an empty word_embedding list.
    word_embedding_2dlist[0] = np.zeros(FLAGS.embed_size)  # assign empty for first word:'PAD'
    bound = np.sqrt(6.0) / np.sqrt(vocab_size)  # bound for random variables.
    count_exist = 0;
    count_not_exist = 0
    for i in range(2, vocab_size):  # loop each word. notice that the first two words are pad and unknown token
        word = vocabulary_index2word[i]  # get a word
        embedding = None
        try:
            embedding = word2vec_dict[word]  # try to get vector:it is an array.
        except Exception:
            embedding = None
        if embedding is not None:  # the 'word' exist a embedding
            word_embedding_2dlist[i] = embedding;
            count_exist = count_exist + 1  # assign array to this word.
        else:  # no embedding for this word
            word_embedding_2dlist[i] = np.random.uniform(-bound, bound, FLAGS.embed_size);
            count_not_exist = count_not_exist + 1  # init a random value for the word.
    word_embedding_final = np.array(word_embedding_2dlist)  # covert to 2d array.
    word_embedding = tf.constant(word_embedding_final, dtype=tf.float32)  # convert to tensor
    t_assign_embedding = tf.assign(textCNN.Embedding,word_embedding)  # assign this value to our embedding variables of our model.
    sess.run(t_assign_embedding);
    print("word. exists embedding:", count_exist, " ;word not exist embedding:", count_not_exist)
    print("using pre-trained word emebedding.ended...")


