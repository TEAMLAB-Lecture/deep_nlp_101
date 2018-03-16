import tensorflow as tf
import os
from config import FLAGS
import time


class CharCnn(object):
    def __init__(self, sequence_length, num_char, batch_size, iteration, init_lr, n_class,
                 embedding_size, num_filter, filter_size, hidden_unit, step_size, decay):
        self.sequence_length = sequence_length
        self.num_char = num_char
        self.batch_size = batch_size
        self.iteration = iteration
        self.init_lr = init_lr
        self.n_class = n_class
        self.embedding_size = embedding_size
        self.num_filters_per_size = num_filter
        self.filter_sizes = filter_size
        self.hidden_unit = hidden_unit
        self.STEP_SIZE = step_size
        self.DECAY = decay

    def calc_acc(self, sess, x, y):
        nbatches = int(len(x) / self.batch_size)
        acc = 0
        for i in range(nbatches):
            acc += sess.run(self.accuracy, feed_dict={
                self.x_data: x[i * self.batch_size:(i + 1) * self.batch_size],
                self.y_data: y[i * self.batch_size:(i + 1) * self.batch_size],
                self.keep_prob: 1.0})
        return acc / nbatches

    def model(self):

        self.x_data = tf.placeholder(tf.int32, [None, self.sequence_length])
        self.y_data = tf.placeholder(tf.float32, [None, self.n_class])

        self.w_init = tf.random_normal_initializer(stddev=0.05)
        self.keep_prob = tf.placeholder(tf.float32)

        embedding_w = tf.Variable(
            tf.random_normal([self.num_char, self.embedding_size], stddev=0.05),
            name='embed_w')
        embedded_chars = tf.nn.embedding_lookup(embedding_w, self.x_data)
        self.embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)

        with tf.name_scope("layer-01"):
            filter_shape = [self.filter_sizes[0], self.embedding_size, 1, self.num_filters_per_size]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.05), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[self.num_filters_per_size]), name="b")
            conv = tf.nn.conv2d(self.embedded_chars_expanded, W, strides=[1, 1, 1, 1], padding="VALID", name="conv1")
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, 3, 1, 1],
                strides=[1, 3, 1, 1],
                padding='VALID',
                name="pool1")

        with tf.name_scope("layer-02"):
            filter_shape = [self.filter_sizes[1], 1, self.num_filters_per_size, self.num_filters_per_size]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.05), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[self.num_filters_per_size]), name="b")
            conv = tf.nn.conv2d(pooled, W, strides=[1, 1, 1, 1], padding="VALID", name="conv2")
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, 3, 1, 1],
                strides=[1, 3, 1, 1],
                padding='VALID',
                name="pool2")

        with tf.name_scope("layer-03"):
            filter_shape = [self.filter_sizes[2], 1, self.num_filters_per_size, self.num_filters_per_size]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.05), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[self.num_filters_per_size]), name="b")
            conv = tf.nn.conv2d(pooled, W, strides=[1, 1, 1, 1], padding="VALID", name="conv3")
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

        with tf.name_scope("layer-04"):
            filter_shape = [self.filter_sizes[3], 1, self.num_filters_per_size, self.num_filters_per_size]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.05), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[self.num_filters_per_size]), name="b")
            conv = tf.nn.conv2d(h, W, strides=[1, 1, 1, 1], padding="VALID", name="conv4")
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

        with tf.name_scope("layer-05"):
            filter_shape = [self.filter_sizes[4], 1, self.num_filters_per_size, self.num_filters_per_size]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.05), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[self.num_filters_per_size]), name="b")
            conv = tf.nn.conv2d(h, W, strides=[1, 1, 1, 1], padding="VALID", name="conv5")
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

        with tf.name_scope("layer-06"):
            filter_shape = [self.filter_sizes[5], 1, self.num_filters_per_size, self.num_filters_per_size]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.05), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[self.num_filters_per_size]), name="b")
            conv = tf.nn.conv2d(h, W, strides=[1, 1, 1, 1], padding="VALID", name="conv6")
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, 3, 1, 1],
                strides=[1, 3, 1, 1],
                padding='VALID',
                name="pool6")
            import numpy as np
            print(np.shape(pooled))

        num_features_total = 7 * self.num_filters_per_size
        h_pool_flat = tf.reshape(pooled, [-1, num_features_total])
        drop1 = tf.nn.dropout(h_pool_flat, self.keep_prob)

        with tf.name_scope("fc-1"):
            W = tf.Variable(tf.truncated_normal([num_features_total, self.hidden_unit], stddev=0.05), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[self.hidden_unit]), name="b")
            fc_1_output = tf.nn.relu(tf.nn.xw_plus_b(drop1, W, b), name="fc-1-out")
            drop2 = tf.nn.dropout(fc_1_output, self.keep_prob)

        with tf.name_scope("fc-2"):
            W = tf.Variable(tf.truncated_normal([self.hidden_unit, self.hidden_unit], stddev=0.05), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[self.hidden_unit]), name="b")
            fc_2_output = tf.nn.relu(tf.nn.xw_plus_b(drop2, W, b), name="fc-2-out")

        with tf.name_scope("fc-3"):
            W = tf.Variable(tf.truncated_normal([self.hidden_unit, self.n_class], stddev=0.05), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[self.n_class]), name="b")
            logits = tf.nn.xw_plus_b(fc_2_output, W, b, name="output")

        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.y_data)
            self.loss = tf.reduce_mean(losses)

        with tf.name_scope("accuracy"):
            predictions = tf.argmax(logits, 1, name="predictions")
            correct_predictions = tf.equal(predictions, tf.argmax(self.y_data, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        return self.loss

    def train(self, x_data, y_data):

        loss = self.model()

        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(self.init_lr, global_step,
                                                   self.STEP_SIZE, self.DECAY,
                                                   staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        gradients = optimizer.compute_gradients(loss)

        train_op = optimizer.apply_gradients(gradients, global_step=global_step)

        saver = tf.train.Saver()
        start_time = time.time()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for epoch in range(self.iteration):
                avg_cost = 0
                total_batch = int(len(x_data)/self.batch_size)
                for step in range(total_batch):
                    feed_dict = {self.x_data: x_data[step*self.batch_size:(step+1)*self.batch_size],
                                                   self.y_data: y_data[step*self.batch_size:(step+1)*self.batch_size],
                                                   self.keep_prob: 0.5}
                    sess.run(train_op, feed_dict=feed_dict)
                    loss_ = sess.run(loss, feed_dict=feed_dict)
                    avg_cost += loss_

                if epoch % 10 == 0:
                    feed_dict = {self.x_data: x_data[step * self.batch_size:(step + 1) * self.batch_size],
                                 self.y_data: y_data[step * self.batch_size:(step + 1) * self.batch_size],
                                 self.keep_prob: 1.0}
                    print("Epoch %d  Loss : %f" % (epoch, avg_cost/total_batch))
                    print("Epoch %d  batch train acc : %f" % (epoch, sess.run(self.accuracy, feed_dict)))

            if not os.path.exists(FLAGS.model_save):
                os.makedirs(FLAGS.model_save)

            saver.save(sess, FLAGS.checkpoint)
            print("Learning finish!")
            print("Duration time: %f" % (time.time()-start_time))

    def test(self, x_test, y_test):

        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, FLAGS.checkpoint)

            test_acc = self.calc_acc(sess, x_test, y_test)
            print('Test accuracy  : {:.5f}'.format(test_acc))







