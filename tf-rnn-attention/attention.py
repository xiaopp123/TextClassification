import tensorflow as tf

def attention(inputs, attention_size, time_major=False, return_alphas=False):
    """
    inputs:
    attention_size:
    time_major:
    return_alphas:
    Returns:
        In case of RNN: [batch_size, cell.output_size]
        In case of Bidirectional RNN: [batch_size, cell_fw.output_size + cell_bw.output_size]
    """
    if isinstance(inputs, tuple):
        inputs = tf.concat(inputs, 2)

    if time_major:
        #[B, T, D] ==> [T, B, D]
        inputs = tf.transpose(inputs, [1, 0, 2])

    # D value
    hidden_size = inputs.shape[2].value

    w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    #不太理解这种attentnion的意义
    with tf.name_scope("v"):
        # the shape of 'v' is (B, T, D) * (D, A) = (B, T, A)
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

    vu = tf.tensordot(v, u_omega, axes=1, name='vu') # (B, T) shape
    alphas = tf.nn.softmax(vu, name="alphas") # (B, T) shape

    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

    if not return_alphas:
        return output
    else:
        return output, alphas

