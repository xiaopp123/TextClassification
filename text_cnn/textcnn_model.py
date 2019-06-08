import tensorflow as tf

class TextCNN(object):
    def __init__(self, filter_sizes, num_filters, num_classes, learning_rate, batch_size,\
                 decay_steps, decay_rate, sequence_length, vocab_size, embed_size, \
                 initializer=tf.random_normal_initializer(stddev=0.1),\
                 multi_label_flag=False,\
                 clip_gradients=5.0,\
                 decay_rate_big=0.50):

        # set hyperparamter
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.learning_rate = tf.Variable(learning_rate, trainable=False, name="learning_rate")#ADD learning_rate
        self.learning_rate_decay_half_op = tf.assign(self.learning_rate, self.learning_rate * decay_rate_big)
        self.filter_sizes = filter_sizes # it is a list of int. e.g. [3,4,5]
        self.num_filters = num_filters
        self.initializer = initializer
        self.num_filters_total = self.num_filters * len(filter_sizes) #how many filters totally.
        self.multi_label_flag = multi_label_flag
        self.clip_gradients = clip_gradients
        self.is_training_flag = tf.placeholder(tf.bool, name="is_training_flag")

        #add placeholder (X, label)
        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")
        #self.input_y = tf.placeholder(tf.int32, [None], name="input_y")
        self.input_y_multilabel = tf.placeholder(tf.float32, [None, self.num_classes], name="input_y_multilable")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.iter = tf.placeholder(tf.int32) #作用,没用到这个变量
        self.tst = tf.placeholder(tf.bool) #没用到这个变量
        self.use_mulitple_layer_cnn = False

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
        self.b1 = tf.Variable(tf.ones([self.num_filters]) / 10) #没用？
        self.b2 = tf.Variable(tf.ones([self.num_filters]) / 10) #没用?
        self.decay_steps, self.decay_rate = decay_steps, decay_rate
        
        #初始化变量权重
        self.instantiate_weights()
        #[none, label_size]
        self.logits = self.inference()

        self.possibility = tf.nn.sigmoid(self.logits)

        if multi_label_flag:
            print("going to use multi label loss")
            self.loss_val = self.loss_multilabel()
        else:
            self.loss_val = self.loss()

        self.train_op = self.train()
        

    def instantiate_weights(self):
        """define all weight here"""
        with tf.name_scope("embedding"):
            self.Embedding = tf.get_variable("embedding", shape=[self.vocab_size, self.embed_size],\
                                             initializer=self.initializer)
            self.W_projection = tf.get_variable("W_projection", shape=[self.num_filters_total, self.num_classes],\
                                                initializer=self.initializer)
            self.b_projection = tf.get_variable("b_projection", shape=[self.num_classes])

    def inference(self):
        """main computation graph here:
           1.embedding
           2.CONV-BN-RELU-MAX_POOLING
           3.linear classifier
        """
        #1.=====>get embedding of words in the sentence
        self.embedded_words = tf.nn.embedding_lookup(self.Embedding, self.input_x)
        #卷积操作，所以要变成4维
        self.sentence_embeddings_expanded = tf.expand_dims(self.embedded_words, -1)
        #2.====>loop each filter size.
        h = self.cnn_single_layer() #[batch, total_filter]
        #3.====>use linear layer 
        with tf.name_scope("output"):
            logits = tf.matmul(h, self.W_projection) + self.b_projection

        return logits

    def train(self):
        """"""
        #需要了解learning rate的改变方式
        #指数衰减法
        #每经过decay_steps后更新learning_rate一次，如果staircase为True，否则每步都更新
        #learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step,\
                                                   self.decay_steps, self.decay_rate, staircase=True)
        self.learning_rate = learning_rate
        optimizer = tf.train.AdamOptimizer(learning_rate)
        #compute_gradients:返回(gradient, variable)对的list
        gradients, variables = zip(*optimizer.compute_gradients(self.loss_val))
        #clip_by_value将梯度限制在一个范围
        #clip_by_global_norm(t_list, clip_norm)在截取是给每一个变量一个缩放因子 clip_norm是截取比率,
        #对于第i个变量更新方式为t_list[i] * clip_norm / max(global_norm, clip_norm)
        #gloabl_norm是所有梯度的平方和,如果clip_norm > global_norm就不截取
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):  #ADD 2018.06.01
            train_op = optimizer.apply_gradients(zip(gradients, variables))
        
        return train_op

    def cnn_single_layer(self):
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.variable_scope("convolution-pooling-%s" % filter_size):
                #=====>a.create filter
                filter = tf.get_variable("filter-%s" % filter_size, \
                                         [filter_size, self.embed_size, 1, self.num_filters],\
                                         initializer=self.initializer)
                #b====>b.conv operation
                #conv = [batch, sequence_length - filter_size + 1, 1, num_filters]
                # padding="valid"是不补0， out_height = ceil((in_height - filter_height + 1) / stride[1])
                conv = tf.nn.conv2d(self.sentence_embeddings_expanded, filter, strides=[1, 1, 1, 1], \
                                    padding="VALID", name="conv")
                # bacth normal 在训练集和测试集中的方式不同
                conv = tf.contrib.layers.batch_norm(conv, is_training=self.is_training_flag, scope='cnn_bn_')

                #=======>c.apply nolinearity
                b = tf.get_variable("b-%s" % filter_size, [self.num_filters])
                h = tf.nn.relu(tf.nn.bias_add(conv, b), "relu") #h=[batch, sequence_length-filter_size, 1, num_filters]

                #======>d.max pooling
                #pool = [batch, 1, 1, num_filters]
                pooled = tf.nn.max_pool(h, ksize=[1, self.sequence_length - filter_size + 1, 1, 1],\
                                        strides=[1, 1, 1, 1], padding="VALID", name="pool")

                pooled_outputs.append(pooled)
        #===>combine all pooled features, and flatten the feature.output 
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, self.num_filters_total]) #shape = [None, num_filters_total]

        #====>add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, keep_prob=self.dropout_keep_prob) #[None, num_filters_total]

        h = tf.layers.dense(self.h_drop, self.num_filters_total, activation=tf.nn.tanh, use_bias=True)

        return h

    def loss_multilabel(self, l2_lambda=0.0001):
        with tf.name_scope("loss"):
            #多个标签问题使用sigmoid_cross_entropy
            #https://zhuanlan.zhihu.com/p/33560183
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y_multilabel, logits=self.logits)
            losses = tf.reduce_sum(losses, axis=1)
            loss = tf.reduce_sum(losses)
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if "bias" not in v.name]) * l2_lambda
            loss =loss + l2_losses

        return loss
    
    def loss(self, l2_lambda=0.0001):
        with tf.name_scope("loss"):
            #input: "logits": [batch_size, num_classes], labels: [batch_size]
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)
            loss = tf.reduce_mean(losses)
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss = loss + l2_losses

        return loss


