# coding: utf-8

import tensorflow as tf
# from SRU_tensorflow import SRUCell
from SRU import SRUCell


# from SRU_Model.sru import SRUCell


def attention(inputs, attention_size, time_major=False, return_alphas=False):
    """
    Attention mechanism layer which reduces RNN/Bi-RNN outputs with Attention vector.
    The idea was proposed in the article by Z. Yang et al., "Hierarchical Attention Networks
     for Document Classification", 2016: http://www.aclweb.org/anthology/N16-1174.
    Variables notation is also inherited from the article

    Args:
        inputs: The Attention inputs.
            Matches outputs of RNN/Bi-RNN layer (not final state):
                In case of RNN, this must be RNN outputs `Tensor`:
                    If time_major == False (default), this must be a tensor of shape:
                        `[batch_size, max_time, cell.output_size]`.
                    If time_major == True, this must be a tensor of shape:
                        `[max_time, batch_size, cell.output_size]`.
                In case of Bidirectional RNN, this must be a tuple (outputs_fw, outputs_bw) containing the forward and
                the backward RNN outputs `Tensor`.
                    If time_major == False (default),
                        outputs_fw is a `Tensor` shaped:
                        `[batch_size, max_time, cell_fw.output_size]`
                        and outputs_bw is a `Tensor` shaped:
                        `[batch_size, max_time, cell_bw.output_size]`.
                    If time_major == True,
                        outputs_fw is a `Tensor` shaped:
                        `[max_time, batch_size, cell_fw.output_size]`
                        and outputs_bw is a `Tensor` shaped:
                        `[max_time, batch_size, cell_bw.output_size]`.
        attention_size: Linear size of the Attention weights.
        time_major: The shape format of the `inputs` Tensors.
            If true, these `Tensors` must be shaped `[max_time, batch_size, depth]`.
            If false, these `Tensors` must be shaped `[batch_size, max_time, depth]`.
            Using `time_major = True` is a bit more efficient because it avoids
            transposes at the beginning and end of the RNN calculation.  However,
            most TensorFlow data is batch-major, so by default this function
            accepts input and emits output in batch-major form.
        return_alphas: Whether to return attention coefficients variable along with layer's output.
            Used for visualization purpose.
    Returns:
        The Attention output `Tensor`.
        In case of RNN, this will be a `Tensor` shaped:
            `[batch_size, cell.output_size]`.
        In case of Bidirectional RNN, this will be a `Tensor` shaped:
            `[batch_size, cell_fw.output_size + cell_bw.output_size]`.
    """

    if isinstance(inputs, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        inputs = tf.concat(inputs, 2)

    if time_major:
        # (T,B,D) => (B,T,D)
        inputs = tf.array_ops.transpose(inputs, [1, 0, 2])

    hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer

    # Trainable parameters
    w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    with tf.name_scope('v'):
        # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
        #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

    # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
    vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
    alphas = tf.nn.softmax(vu, name='alphas')  # (B,T) shape

    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

    if not return_alphas:
        return output
    else:
        return output, alphas


class TCNNConfig(object):
    """CNN配置参数"""

    embedding_dim = 64  # 词向量维度
    seq_length = 600  # 序列长度
    num_classes = 10  # 类别数
    num_filters = 256  # 卷积核数目
    kernel_size = [3, 4]  # 卷积核尺寸
    vocab_size = 5000  # 词汇表达小

    hidden_dim = 128  # 全连接层神经元

    dropout_keep_prob = 0.5  # dropout保留比例
    learning_rate = 1e-3  # 学习率

    batch_size = 64  # 每批训练大小
    num_epochs = 10  # 总迭代轮次

    print_per_batch = 100  # 每多少轮输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard
    attention_size = 64
    lstm_size = 128
    keep_prob = 0.5  # LSTM的dropout概率


class TextCNN(object):
    """文本分类，CNN模型"""

    def __init__(self, config):
        self.config = config

        # 三个待输入的数据
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.learning_rate = self.config.learning_rate
        self.cnn()

    def BiRNN(self, x, name):

        # x = tf.transpose(x, [1, 0, 2])
        # x = tf.reshape(x, [-1, x.shape[-1]])
        # x = tf.split(x, x.shape[-2])
        # 解决变量作用域问题。
        ## 模型中多次服用次函数，故加上变量作用域，以此来区别。
        with tf.variable_scope(name):


            lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(self.config.lstm_size, forget_bias=1.0)
            lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(self.config.lstm_size, forget_bias=1.0)
            lstm_fw_cell = tf.nn.rnn_cell.GRUCell(self.config.lstm_size)
            lstm_bw_cell = tf.nn.rnn_cell.GRUCell(self.config.lstm_size)
            lstm_fw_cell = SRUCell(self.config.lstm_size)
            lstm_bw_cell = SRUCell(self.config.lstm_size)
            # dropout
            lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_fw_cell, input_keep_prob=1.0,
                                                         output_keep_prob=self.config.keep_prob)
            lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_bw_cell, input_keep_prob=1.0,
                                                         output_keep_prob=self.config.keep_prob)
            (outputs, _) = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)
        # 将两个LSTM的输出合并
        output_fw, output_bw = outputs
        # gmp = _outputs[:, -1, :]  # 取最后一个时序输出作为结果
        output = tf.concat([output_fw[:, -1, :] ,output_bw[:, -1, :] ],axis=-1)
        #output = tf.concat([output_fw, output_bw], axis=-1)
        return output

    def cnn(self):

        def lstm_cell():  # lstm核
            return tf.nn.rnn_cell.BasicLSTMCell(self.config.hidden_dim, state_is_tuple=True)

        def gru_cell():  # gru核
            return tf.nn.rnn_cell.GRUCell(self.config.hidden_dim)

        def dropout():  # 为每一个rnn核后面加一个dropout层
            if (self.config.rnn == 'lstm'):
                cell = lstm_cell()
            else:
                cell = gru_cell()
            return tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.config.keep_prob)

        # 定义Bidirectional LSTM网络的生成函数


        """BCL模型"""
        # 词向量映射
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)
        #with tf.name_scope("bilstm"):
            #output_lstm = self.BiRNN(embedding_inputs, 'bilstm_')
            # conv1 = tf.layers.conv1d(output_lstm, self.config.num_filters, self.config.kernel_size, name='conv1')
        with tf.name_scope("convolution_lstm"):
            # CNN layer
            conv = tf.layers.conv1d(embedding_inputs, self.config.num_filters, self.config.kernel_size[1], name='conv',kernel_regularizer=tf.keras.regularizers.l2(1e-6))
            # global max pooling layer
            gmp = tf.reduce_max(conv, reduction_indices=[1], name='gmp')
            # CNN parallel
            '''conv = tf.layers.conv1d(embedding_inputs, self.config.num_filters, 2, name='conv',
                                    kernel_regularizer=tf.keras.regularizers.l2(1e-5))
            conv1 = tf.layers.conv1d(embedding_inputs, self.config.num_filters, 3, name='conv1',
                                    kernel_regularizer=tf.keras.regularizers.l2(1e-5))
            conv2 = tf.layers.conv1d(embedding_inputs, self.config.num_filters, 4, name='conv2',
                                    kernel_regularizer=tf.keras.regularizers.l2(1e-5))
            gmp = tf.reduce_max(conv, reduction_indices=[1], name='gmp')
            gmp1 = tf.reduce_max(conv1, reduction_indices=[1], name='gmp1')
            gmp2 = tf.reduce_max(conv2, reduction_indices=[1], name='gmp2')
            gmp = tf.concat([gmp,gmp1,gmp2],axis=-1)'''
            # lstm
            #lstm = tf.nn.rnn_cell.BasicLSTMCell(self.config.hidden_dim, state_is_tuple=True)
            #sru = SRUCell(self.config.lstm_size)
            #outputs, states = tf.nn.static_rnn(sru, embedding_inputs, dtype=tf.float32)
            # Bi-LSTM
            #_outputs = self.BiRNN(conv, 'conv_lstm')
            #outputs, _ = tf.nn.dynamic_rnn(cell=sru, inputs=embedding_inputs, dtype=tf.float32)
            # 多层rnn网络

            '''cells = [dropout() for _ in range(self.config.num_layers)]
            rnn_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
            _outputs, _ = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=conv, dtype=tf.float32)'''
            # gmp = _outputs[:, -1, :]  # 取最后一个时序输出作为结果
            #conv1 = tf.layers.conv1d(embedding_inputs, self.config.num_filters, self.config.kernel_size[1],name='conv1', kernel_regularizer=tf.keras.regularizers.l2(1e-5))
            #_outputs1 = self.BiRNN(conv1, 'conv_lstm1')
            # _outputs1 = self.BiRNN(_outputs1, 'conv_lstm2')
        #with tf.name_scope("Merge"):
            #_outputs = tf.concat([_outputs, _outputs1], axis=-2)
            #_outputs = _outputs[:, -1, :]
        # Attention layer
        #with tf.name_scope('Attention_layer'):
        #    attention_output, alphas = attention(conv, self.config.embedding_dim, return_alphas=True)
        #    tf.summary.histogram('alphas', alphas)

        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            fc = tf.layers.dense(gmp, self.config.hidden_dim, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)

            # 分类器
            self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别

        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy, name='loss')
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            # self.optim = tf.train.AdadeltaOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)
        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='acc')
