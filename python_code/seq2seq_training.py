import gensim
import tensorflow as tf
import numpy as np
from konlpy.tag import Mecab
import pickle


class dialouge_training_lstm:
    def __init__(self, x_data, y_data, **model_info):
        '''
        data를 입력 받고 한국어 형태소 분석 엔진을 로드한다.
        :param x_data: string을 원소로 갖는 리스트 (list)
        :param y_data: string을 원소로 갖는 리스트 (list)
        '''
        self.ko_engine = Mecab()
        self.model_path = model_info['model_path']
        self.w2v_path = model_info['w2v_path']
        self.w2v_model = gensim.models.Word2Vec.load(self.w2v_path)
        self.x_data = x_data
        self.y_data = y_data
        x_len_list = np.array([len(self.ko_engine.pos(data_element)) for data_element in self.x_data])
        self.x_data = list(np.array(self.x_data)[x_len_list < 100])
        self.y_data = list(np.array(self.y_data)[x_len_list < 100])
        y_len_list = np.array([len(self.ko_engine.pos(data_element)) for data_element in self.y_data])
        self.x_data = list(np.array(self.x_data)[y_len_list < 100])
        self.y_data = list(np.array(self.y_data)[y_len_list < 100])

        # x_data, y_data를 pos로 쪼개서 저장
        self.enc_source = [[x[0] for x in self.ko_engine.pos(data_element)] for data_element in self.x_data]
        self.dec_source = [[y[0] for y in self.ko_engine.pos(data_element)] for data_element in self.y_data]

    def translate_to_word(self, outputs) :
        eos_index = self.w2v_model.wv["<EOS>"]
        trans_temp = [[self.w2v_model.wv.most_similar(positive=[vec], topn=1)[0][0] for vec in data_element if not np.array_equal(eos_index, vec)] for data_element in outputs]
        trans_result = [" ".join(data_element) for data_element in trans_temp]
        return trans_result


    def update_word2vec(self):
        response = input("word2vec 모델을 학습시킬 경우 기존에 학습된 모델의 결과가 달라집니다. 계속 하시겠습니까? [y/n] : ")
        enc_set = set([tag for data_element in self.enc_source for tag in data_element])
        dec_set = set([tag for data_element in self.dec_source for tag in data_element])
        words_temp = enc_set.union(dec_set)
        new_words = (words_temp - set(self.w2v_model.wv.vocab))
        new_words = [list(new_words)]*self.w2v_model.min_count
        self.w2v_model.build_vocab(new_words, update=True)
        self.w2v_model.train((self.enc_source + self.dec_source[-1])*self.w2v_model.min_count,
                             total_examples=self.w2v_model.corpus_count, epochs=self.w2v_model.iter)
        if response == "y":
            with open(self.w2v_path, mode="wb") as fp:
                self.w2v_model.save(fp)

    def data_preprocess(self):
        '''
        string의 배열을 각각 입력 받아 기본적인 데이터 전처리(idx2word, word2idx 생성하고 word vector를 idx vector로 치환)한다.
        '''
        # pos로 나뉘어진 정보를 encoder, decoder에 맞게 training에 사용할 input/output으로 나눔
        self.enc_input = [data_element for data_element in self.enc_source]
        self.enc_output = [data_element for data_element in self.dec_source]
        self.dec_input = [['<GO>'] + data_element for data_element in self.dec_source]
        self.dec_output = [data_element + ['<EOS>'] for data_element in self.dec_source]

        # 나뉘어진 인코더를 이용해 그에 맞는 sequence_length를 구함. dynamic_rnn의 sequence_length에 사용할 인자
        self.enc_input_seqlen = [len(sentence) for sentence in self.enc_input]
        self.enc_output_seqlen = [len(sentence) for sentence in self.enc_output]
        self.dec_seqlen = [len(sentence) for sentence in self.dec_input]
        # padding의 기준으로 삼을 max_seqlen
        # self.max_enc_seqlen = max(max(self.enc_input_seqlen), max(self.enc_output_seqlen))
        # self.max_dec_seqlen = max(self.dec_seqlen)
        self.max_seqlen = 100  # max(self.max_enc_seqlen, self.max_dec_seqlen)

        # decoder에 pad값 넣어줌
        self.enc_input = [data_element + ['<PAD>'] * (self.max_seqlen - len(data_element)) for data_element in
                          self.enc_input]
        self.enc_output = [data_element + ['<PAD>'] * (self.max_seqlen - len(data_element)) for data_element in
                           self.enc_output]

        # decoder에 pad값 넣어줌
        self.dec_input = [data_element + ['<PAD>'] * (self.max_seqlen - len(data_element)) for data_element in
                          self.dec_input]
        self.dec_output = [data_element + ['<PAD>'] * (self.max_seqlen - len(data_element)) for data_element in
                           self.dec_output]

        self.update_word2vec()

    def set_parameter(self, num_layer=3, learning_rate=0.03, epoch=3000):
        '''
        :param num_layer: lstm layer의 깊이 (integer) 
        :param learning_rate: 학습 속도 (float)
        :param epoch: iteration, 학습 반복수 (integer)
        '''
        self.data_preprocess()

        # input/output 데이터에 관한 parameter
        self.dim = 500  # len(self.word2idx)
        self.enc_batch_size = len(self.enc_source)
        self.dec_batch_size = len(self.dec_source)
        self.batch_size = len(self.dec_source)

        # 그래프 구성에 필요한 parameter
        self.num_layer = num_layer
        self.learning_rate = learning_rate
        self.epoch = epoch

    def get_new_data(self, x_data, y_data, **model_info):
        '''
        새로운 데이터를 받을 경우, 해당 데이터로 치환한다. parameter값을 재설정하지 않으면 이전 값을 그대로 사용한다.
        :param x_data: string을 원소로 갖는 리스트 (list)
        :param y_data: string을 원소로 갖는 리스트 (list) 
        '''
        self.model_path = model_info['model_path']
        self.w2v_path = model_info['w2v_path']
        self.w2v_model = model  # gensim.models.Word2Vec.load(model_info['w2v_path'])

        self.x_data = x_data
        self.y_data = y_data
        x_len_list = np.array([len(self.ko_engine.pos(data_element)) for data_element in self.x_data])
        self.x_data = list(np.array(self.x_data)[x_len_list < 100])
        self.y_data = list(np.array(self.y_data)[x_len_list < 100])
        y_len_list = np.array([len(self.ko_engine.pos(data_element)) for data_element in self.y_data])
        self.x_data = list(np.array(self.x_data)[y_len_list < 100])
        self.y_data = list(np.array(self.y_data)[y_len_list < 100])

        # x_data, y_data를 pos로 쪼개서 저장
        self.enc_source = [[x[0] for x in self.ko_engine.pos(data_element)] for data_element in self.x_data]
        self.dec_source = [[y[0] for y in self.ko_engine.pos(data_element)] for data_element in self.y_data]

        self.data_preprocess()

    def get_encoder_data(self):
        '''
        encoder에 필요한 data 생성
        :return1: input data for encoder, list of list which have vectorized word as elements (list)
        :return2: output data for encoder, list of list which have vectorized word as elements (list)
        '''
        dataX = []
        dataY = []

        # encoder에 사용할 X, Y
        for i in range(0, self.enc_batch_size):
            x_words = self.enc_input[i]
            y_words = self.enc_output[i]

            x = [self.w2v_model.wv[word] for word in x_words]
            y = [self.w2v_model.wv[word] for word in y_words]

            dataX.append(x)
            dataY.append(y)

        return dataX, dataY

    def get_decoder_data(self):
        '''
        decoder에 필요한 data 생성
        :return1: input data for decoder, list of list which have vectorized word as elements (list)
        :return2: output data for decoder, list of list which have vectorized word as elements (list)
        '''
        dataX = []
        dataY = []

        # decoder에 사용할 X, Y
        for i in range(0, self.dec_batch_size):
            x_words = self.dec_input[i]
            y_words = self.dec_output[i]

            x = [self.w2v_model.wv[word] for word in x_words]
            y = [self.w2v_model.wv[word] for word in y_words]

            dataX.append(x)
            dataY.append(y)

        return dataX, dataY

    def softmax_and_reshape(self, outputs, name, batch_len, tf_name):

        # fully-connected layers
        outputs = tf.reshape(outputs, [batch_len, self.max_seqlen, self.dim])
        X_for_softmax = tf.reshape(outputs, [-1, self.dim])

        with tf.variable_scope(name):
            softmax_w = tf.Variable(tf.random_normal([self.dim, self.dim]), name="softmax_w")
            softmax_b = tf.Variable(tf.random_normal([self.dim]), name="softmax_b")
            outputs = tf.matmul(X_for_softmax, softmax_w) + softmax_b

        outputs = tf.reshape(outputs, [batch_len, self.max_seqlen, self.dim], name=tf_name)

        return outputs

    def train_seq2seq(self):
        # self.set_parameter()
        batch_len = tf.placeholder(tf.int32, name="batch_len")

        #######encoder#######
        with tf.variable_scope('encode'):
            enc_X = tf.placeholder(tf.float32, [None, None, self.dim], name="enc_X")
            enc_Y = tf.placeholder(tf.float32, [None, None, self.dim], name="enc_Y")
            enc_seqlen = tf.placeholder(tf.int32, name="enc_seqlen")

            enc_cell = tf.nn.rnn_cell.LSTMCell(self.dim, state_is_tuple=True)
            enc_multi_cell = tf.nn.rnn_cell.MultiRNNCell([enc_cell] * self.num_layer, state_is_tuple=True)

            enc_outputs, enc_states = tf.nn.dynamic_rnn(enc_multi_cell,
                                                        enc_X,
                                                        sequence_length=enc_seqlen,
                                                        dtype=tf.float32)

            enc_prediction = self.softmax_and_reshape(enc_outputs, "encode", batch_len, tf_name="enc_prediction")
            # enc_outputs = self.softmax_and_reshape(enc_outputs, "encode", batch_len)
            # enc_prediction = tf.argmax(enc_outputs, axis=2, name="enc_prediction")

            enc_weights = tf.ones([self.enc_batch_size, self.max_seqlen])  # seqlen을 None으로 하는 방법을 최종으로
            #enc_sequence_loss = tf.losses.cosine_distance(predictions=enc_prediction, labels=enc_Y, dim=1)
            #enc_sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=(enc_prediction-enc_Y), targets=tf.zeros([self.batch_size, 100], dtype=tf.int32), weights=enc_weights)
            ######
            h = 1e-5
            enc_prediction_reshape = tf.reshape(enc_prediction, [-1, self.dim])
            enc_Y_reshape = tf.reshape(enc_Y, [-1, self.dim])
            eP_norm = tf.sqrt(tf.reduce_sum((enc_prediction_reshape * enc_prediction_reshape), axis=1))
            eY_norm = tf.sqrt(tf.reduce_sum((enc_Y_reshape * enc_Y_reshape), axis=1))
            e_denom = eP_norm * eY_norm + h
            e_num = tf.reduce_sum((enc_prediction_reshape * enc_Y_reshape), axis=1)
            e_cosine_similarity = e_num / e_denom
            e_cosine_distance = 1 - e_cosine_similarity
            e_cosine_distance_reshape = tf.reshape(e_cosine_distance, [self.enc_batch_size, self.max_seqlen])
            e_cosine_distance_reshape = e_cosine_distance_reshape * enc_weights
            enc_sequence_loss = tf.reduce_mean(e_cosine_distance_reshape, axis=1)
            ######
            enc_mean_loss = tf.reduce_mean(enc_sequence_loss)
            enc_train = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(enc_mean_loss)

        #######decoder#######
        with tf.variable_scope('decode'):
            dec_X = tf.placeholder(tf.float32, [None, None, self.dim], name="dec_X")
            dec_Y = tf.placeholder(tf.float32, [None, None, self.dim], name="dec_Y")
            dec_seqlen = tf.placeholder(tf.int32, name="dec_seqlen")

            dec_cell = tf.nn.rnn_cell.LSTMCell(self.dim, state_is_tuple=True)
            dec_multi_cell = tf.nn.rnn_cell.MultiRNNCell([dec_cell] * self.num_layer, state_is_tuple=True)

            dec_outputs, dec_states = tf.nn.dynamic_rnn(dec_multi_cell,
                                                        dec_X,
                                                        sequence_length=dec_seqlen,
                                                        initial_state=enc_states,
                                                        dtype=tf.float32)

            dec_prediction = self.softmax_and_reshape(dec_outputs, "decode", batch_len, tf_name="dec_prediction")
            # dec_outputs = self.softmax_and_reshape(dec_outputs, "decode", batch_len)
            # dec_prediction = tf.argmax(dec_outputs, axis=2, name="dec_prediction")

            dec_weights = tf.ones([self.dec_batch_size, self.max_seqlen])  # seqlen을 None으로 하는 방법을 최종으로
            #dec_sequence_loss = tf.losses.cosine_distance(predictions=dec_prediction, labels=dec_Y, dim=0)
            #dec_sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=(dec_prediction-dec_Y), targets=tf.zeros([self.batch_size, 100], dtype=tf.int32), weights=dec_weights)
            #########
            h = 1e-5
            dec_prediction_reshape = tf.reshape(dec_prediction, [-1, self.dim])
            dec_Y_reshape = tf.reshape(dec_Y, [-1, self.dim])
            dP_norm = tf.sqrt(tf.reduce_sum((dec_prediction_reshape * dec_prediction_reshape), axis=1))
            dY_norm = tf.sqrt(tf.reduce_sum((dec_Y_reshape * dec_Y_reshape), axis=1))
            d_denom = dP_norm * dY_norm + h
            d_num = tf.reduce_sum((dec_prediction_reshape * dec_Y_reshape), axis=1)
            d_cosine_similarity = d_num / d_denom
            d_cosine_distance = 1 - d_cosine_similarity
            d_cosine_distance_reshape = tf.reshape(d_cosine_distance, [self.dec_batch_size, self.max_seqlen])
            d_cosine_distance_reshape = d_cosine_distance_reshape * dec_weights
            dec_sequence_loss = tf.reduce_mean(d_cosine_distance_reshape, axis=1)
            ########
            dec_mean_loss = tf.reduce_mean(dec_sequence_loss)
            dec_train = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(dec_mean_loss)

        self.enc_prediction = enc_prediction
        self.dec_prediction = dec_prediction
        self.enc_X = enc_X
        self.enc_Y = enc_Y
        self.dec_X = dec_X
        self.dex_Y = dec_Y

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            enc_dataX, enc_dataY = self.get_encoder_data()
            dec_dataX, dec_dataY = self.get_decoder_data()
            go_index = [self.w2v_model.wv["<GO>"]]

            for ep in range(self.epoch):
                e_loss, _ = sess.run([enc_mean_loss, enc_train],
                         feed_dict={enc_X: enc_dataX, enc_Y: enc_dataY, enc_seqlen: self.enc_input_seqlen,
                                    batch_len: self.batch_size})
                d_loss, _ = sess.run([dec_mean_loss, dec_train],
                         feed_dict={enc_X: enc_dataX, enc_Y: enc_dataY, enc_seqlen: self.enc_input_seqlen,
                                    batch_len: self.batch_size,
                                    dec_X: dec_dataX, dec_Y: dec_dataY, dec_seqlen: self.dec_seqlen})

                enc_result = sess.run(enc_prediction, feed_dict={enc_X: enc_dataX, enc_seqlen: self.enc_input_seqlen,
                                                                 batch_len: self.batch_size})
                test_dec_input = [list(go_index) + list(enc_output[:-1]) for enc_output in enc_result]
                dec_result = sess.run(dec_prediction, feed_dict={enc_X: enc_dataX, enc_seqlen: self.enc_input_seqlen,
                                                                 batch_len: self.batch_size,
                                                                 dec_X: dec_dataX, dec_seqlen: self.dec_seqlen})
                # result는 index scalar의 vector(list)를 원소로 갖는 리스트

                #if ((ep/self.epoch)*100)%5==0 :
                if ep%50 == 0:
                    print("ep:",ep, "]]]]]\t\te_loss:",e_loss, "\td_loss:", d_loss, sep="")
                    #print("\t\tencP: ",self.translate_to_word(enc_result), "\tdecP: ",self.translate_to_word(dec_result))


            saver = tf.train.Saver()
            saver.save(sess, self.model_path)
'''
tf.reset_default_graph()

X = []
Y = []


with open("training_data/beauty_inside/beauty_inside.txt", mode="rb") as fp :
    X, Y, _, _ = pickle.load(fp)

temp = dialouge_training_lstm([X[3]], [Y[3]], model_path="./test", w2v_path="word2vec/word2vec")
temp.set_parameter(learning_rate=0.1, epoch=1000, num_layer=2)
temp.train_seq2seq()
'''

new_or_old = input("새로운 모델에 학습시키겠습니까? [y/n]\n(기존 모델에 이어서 학습시키려면 n 입력) :\n")
data_path = input("학습시킬 데이터가 위치한 path를 입력해주세요. (pickle.dump로 저장된 txt 파일) : \n")
with open(data_path, mode="rb") as fp :
    X, Y, _, _ = pickle.load(fp)

learning_rate, epoch, num_layer = input("학습 parameter를 설정해주세요. (learning_rate / epoch / num_layer) :\n").split()

if new_or_old == "y" :
    model_save_path = input("새로운 모델을 저장할 path를 입력해주세요 (eg. ./total) :\n")
    word2vec_path = input("word2vec 모델이 위치한 path를 입력해주세요 :\n")
    print("learning_rate :", learning_rate, "\tepoch :", epoch, "\tnum_layer :", num_layer)
    temp = dialouge_training_lstm(X, Y, model_path=model_save_path, w2v_path=word2vec_path)
    temp.set_parameter(learning_rate=float(learning_rate), epoch=int(epoch), num_layer=int(num_layer))
    temp.train_seq2seq()
elif new_or_old == "n" :
    model_save_path = input("기존 모델이 위치한 path를 입력해주세요 (eg. ./model/beauty_inside/beauty_inside) :\n")
    word2vec_path = input("word2vec 모델이 위치한 path를 입력해주세요 :\n")
    print("learning_rate :", learning_rate, "\tepoch :", epoch, "\tnum_layer :", num_layer)
