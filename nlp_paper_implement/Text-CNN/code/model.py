import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
from sklearn import preprocessing
from konlpy.tag import Twitter
pos_tagger = Twitter()

class KoreanTextCnn():
    
    def __init__(self,channel_shape):
        
        self.TRAIN_DIRECTORY = '../data/ratings_train.txt'
        self.TEST_DIRECTORY = '../data/ratings_test.txt'
        self.EMBED_SIZE = 300
        self.MAX_LENGTH_SENTENCE = 0
        self.FILTER_SIZE = [2,3,4,5]
        self.NUM_CLASSES = 2
        self.WINDOW_SIZE = 5
        self.EXTRACT_TAG = ['Noun','Verb','Adjective']
        self.STOPWORD_KOREAN = set([value.strip() for value in open("../data/stop_word_korea.txt","r").readlines()])
        
        self.train_data, self.test_data, self.channel_shape = self.read_data(self.TRAIN_DIRECTORY,self.TEST_DIRECTORY,channel_shape) 
        self.one_dimension_data = []
        self.reverse_dictionary = []
        self.dictionary = {}
        self.word2vec_center_word = []
        self.word2vec_target_word = []
        self.train_docs = []
        self.test_docs = []
        self.train_y_data = []
        self.test_y_data = []
        self.train_doc_to_image = []
        self.test_doc_to_image = []
        
    def read_data(self,train_directory,test_directory,channel_shape):
        
        train_select_number = 10000
        test_select_number = 10000
        
        with open(train_directory, 'r') as f:
            data = [line.split('\t') for line in f.read().splitlines()]
            data = data[1:]
        train_data = data[:train_select_number]
        
        with open(test_directory, 'r') as f:
            data = [line.split('\t') for line in f.read().splitlines()]
            data = data[1:]
        test_data = data[:test_select_number]
        
        return train_data, test_data, channel_shape
        
    def dictionary_for_static(self):
        
        self.one_dimension_data = ['/'.join(t_w) for t_i in  self.train_data for t_w in pos_tagger.pos(t_i[1],norm=True, stem=True) \
                                    if t_w[1] in self.EXTRACT_TAG and t_w[0] not in self.STOPWORD_KOREAN and len(t_w[0]) != 1]
        self.reverse_dictionary = [i for i in list(set(self.one_dimension_data))] 
        self.dictionary = {w: i for i, w in enumerate(self.reverse_dictionary)} 
        
    def static_word2vec_dataset(self):
        
         for index,center in enumerate(self.reverse_dictionary):
            for window in range(-self.WINDOW_SIZE//2,(self.WINDOW_SIZE//2)+1):
                if window == 0:
                    continue
                elif index+window < 0 or index+window >= len(self.reverse_dictionary):
                    continue
                else:
                    self.word2vec_center_word.append(self.dictionary[self.reverse_dictionary[index]])
                    self.word2vec_target_word.append([self.dictionary[self.reverse_dictionary[index+window]]])
                    
    def datas_to_docs(self):
        lb = preprocessing.LabelBinarizer()
        self.train_docs = [['/'.join(t_w) for t_w in pos_tagger.pos(t_i[1],norm=True, stem=True) \
                           if t_w[1] in self.EXTRACT_TAG and t_w[0] not in self.STOPWORD_KOREAN and len(t_w[0]) != 1] \
                           for t_i in self.train_data]
        self.test_docs = [['/'.join(t_w) for t_w in pos_tagger.pos(t_i[1],norm=True, stem=True) \
                           if t_w[1] in self.EXTRACT_TAG and t_w[0] not in self.STOPWORD_KOREAN and len(t_w[0]) != 1] \
                           for t_i in self.test_data]
        train_data_y = [int(t_i[2]) for t_i in self.train_data]
        test_data_y = [int(t_i[2]) for t_i in self.test_data]
        lb.fit(list(range(self.NUM_CLASSES+1)))
        self.train_y_data = lb.transform(train_data_y)
        self.train_y_data = self.train_y_data[:,:2]
        self.test_y_data = lb.transform(test_data_y)
        self.test_y_data = self.test_y_data[:,:2]
        
    def static_make_word2vec(self):
        
        VOCAB_SIZE = len(self.reverse_dictionary)
        BATCH_SIZE = 128
        LEARNING_RATE = 1.0
        NUM_TRAIN_STEPS = 400
        SKIP_STEP = 100
        EMBED_SIZE = 300
        NUM_SAMPLED = 64
        BATCH_COUNT = len(self.word2vec_center_word)//BATCH_SIZE
        center_words = tf.placeholder(tf.int32, shape=[BATCH_SIZE])
        target_words = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 1])
        embed_matrix = tf.Variable(tf.random_uniform([VOCAB_SIZE, EMBED_SIZE], -1.0, 1.0))

        embed = tf.nn.embedding_lookup(embed_matrix, center_words, name='embed')

        nce_weight = tf.Variable(tf.truncated_normal([VOCAB_SIZE, EMBED_SIZE],
                                                    stddev=1.0 / (EMBED_SIZE ** 0.5)), 
                                                    name='nce_weight')
        nce_bias = tf.Variable(tf.zeros([VOCAB_SIZE]), name='nce_bias')

        loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight, 
                                            biases=nce_bias, 
                                            labels=target_words, 
                                            inputs=embed, 
                                            num_sampled=NUM_SAMPLED, 
                                            num_classes=VOCAB_SIZE), name='loss')
        optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
             
            for iterate in range(NUM_TRAIN_STEPS):
                total_loss = 0.0
                self.word2vec_center_word, self.word2vec_target_word = shuffle(self.word2vec_center_word, self.word2vec_target_word, random_state=0)
                for count in range(BATCH_COUNT):
                    centers, targets =self.word2vec_center_word[count*BATCH_SIZE:(count+1)*BATCH_SIZE],self.word2vec_target_word[count*BATCH_SIZE:(count+1)*BATCH_SIZE]
                    loss_batch, _ = sess.run([loss, optimizer], 
                                            feed_dict={center_words: centers, target_words: targets})
                    total_loss += loss_batch
               
                if (iterate + 1) % SKIP_STEP == 0:
                    print("iterate : ",iterate)
                    print("loss : ",total_loss)
                    total_loss = 0.0
            self.static_word2vec = sess.run(embed_matrix)
            sess.close()

    def static_docs_to_index(self):
        
        VOCAB_SIZE = len(self.reverse_dictionary)
        self.MAX_LENGTH_SENTENCE = max([len(doc) for doc in self.train_docs])
        
            
        for index in range(len(self.train_docs)):
            self.train_doc_to_image.append([])
            for t_i in self.train_docs[index]:
                if  t_i in self.reverse_dictionary:
                    dic_i = self.reverse_dictionary.index(t_i)
                    self.train_doc_to_image[index].append(dic_i)
                else:
                    self.train_doc_to_image[index].append(-1)

            for cnt in range(len(self.train_doc_to_image[index]),self.MAX_LENGTH_SENTENCE):
                self.train_doc_to_image[index].append(-1)
            self.train_doc_to_image[index] = np.array(self.train_doc_to_image[index])
        self.train_doc_to_image = np.array(self.train_doc_to_image)  
        
        
        for index in range(len(self.test_docs)):
            self.test_doc_to_image.append([])
            for t_i in self.test_docs[index]:
                if  t_i in self.reverse_dictionary:
                    dic_i = self.reverse_dictionary.index(t_i)
                    self.test_doc_to_image[index].append(dic_i)
                else:
                    self.test_doc_to_image[index].append(-1)
                    
            for cnt in range(len(self.test_doc_to_image[index]),self.MAX_LENGTH_SENTENCE):
                self.test_doc_to_image[index].append(-1)
            self.test_doc_to_image[index] = np.array(self.test_doc_to_image[index])
        self.test_doc_to_image = np.array(self.test_doc_to_image)  

    def static_cnn_method(self):
        LEARNING_RATE = 0.01
        pooled_outputs = []
        multi = False
        
        X = tf.placeholder(tf.int32, [None, self.MAX_LENGTH_SENTENCE])
        Y = tf.placeholder(tf.float32, [None, self.NUM_CLASSES])
        keep_prob = tf.placeholder(tf.float32)
        
        num_filters = 1
        if self.channel_shape == 'rand':
            num_channel = 1
            word2vec = tf.Variable(tf.random_uniform([len(self.reverse_dictionary), self.EMBED_SIZE], -1.0, 1.0))
            embedded_chars = tf.nn.embedding_lookup(word2vec, X)
            embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)
        elif self.channel_shape == 'static':
            num_channel = 1
            word2vec = tf.Variable(self.static_word2vec,trainable=False)
            embedded_chars = tf.nn.embedding_lookup(word2vec, X)
            embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)
        elif self.channel_shape == 'nonstatic':
            num_channel = 1
            word2vec = tf.Variable(self.static_word2vec,trainable=True)
            embedded_chars = tf.nn.embedding_lookup(word2vec, X)
            embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)
        else:
            num_channel = 2
            multi = True
            word2vec = tf.Variable(self.static_word2vec,trainable=False)
            embedded_chars = tf.nn.embedding_lookup(word2vec, X)
            embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)
            
            multi_word2vec = tf.Variable(self.static_word2vec,trainable=True)
            embedded_chars_multi = tf.nn.embedding_lookup(multi_word2vec, X)
            embedded_chars_expanded_multi = tf.expand_dims(embedded_chars_multi, -1)
        for filter_size in self.FILTER_SIZE:
            
            W = tf.Variable(tf.random_normal([filter_size, self.EMBED_SIZE, 1, 1], stddev=0.01))
            L = tf.nn.conv2d(embedded_chars_expanded, W, strides=[1, 1, 1, 1], padding='VALID')
            L = tf.nn.relu(L)
            L = tf.nn.max_pool(L, ksize=[1, self.MAX_LENGTH_SENTENCE-filter_size+1, 1, 1], strides=[1, 1, 1, 1], padding='VALID')
            pooled_outputs.append(L)
            if multi:
                L = tf.nn.conv2d(embedded_chars_expanded_multi, W, strides=[1, 1, 1, 1], padding='VALID')
                L = tf.nn.relu(L)
                L = tf.nn.max_pool(L, ksize=[1, self.MAX_LENGTH_SENTENCE-filter_size+1, 1, 1], strides=[1, 1, 1, 1], padding='VALID')
                pooled_outputs.append(L)
        h_pool = tf.concat(pooled_outputs,3)
        h_pool = tf.reshape(h_pool,[-1,len(self.FILTER_SIZE)*num_channel*num_filters])
        
        h_drop = tf.nn.dropout(h_pool, keep_prob)
        
        w_2 = tf.get_variable("w_2",
                shape=[len(self.FILTER_SIZE)*num_channel*num_filters, self.NUM_CLASSES],
                initializer=tf.contrib.layers.xavier_initializer())
        b_2 = tf.Variable(tf.constant(0.1, shape=[self.NUM_CLASSES]),name='b')
        output_layer = tf.nn.tanh(tf.matmul(h_drop,w_2) + b_2)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=Y))
        correct_prediction = tf.equal(tf.argmax(output_layer,1), tf.argmax(Y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            
            for iterate in range(100000):
                self.train_doc_to_image, self.train_y_data = shuffle(self.train_doc_to_image, self.train_y_data, random_state=0)
                result = sess.run(word2vec)
                losses, _ = sess.run([loss, optimizer], feed_dict={X: self.train_doc_to_image, Y: self.train_y_data, keep_prob:0.5})
                if iterate % 10 == 0:
                    print("\n")
                    print("iterate : ",iterate)
                    print("loss : ",sess.run(loss,feed_dict={X: self.train_doc_to_image, Y: self.train_y_data, keep_prob:1.0}))
                    print("train accuray : ",sess.run(accuracy,feed_dict={X: self.train_doc_to_image, Y: self.train_y_data, keep_prob:1.0}))
                    print("test accuray : ",sess.run(accuracy,feed_dict={X: self.test_doc_to_image, Y: self.test_y_data, keep_prob:1.0}))            
        sess.close()
