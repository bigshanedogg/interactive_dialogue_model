import gensim
import tensorflow as tf
import numpy as np
from konlpy.tag import Mecab
import pickle


class dialogue_generator:
    def __init__(self, **model_info):
        # 모델의 디렉토리와 사용할 모델명
        self.ko_engine = Mecab()
        self.model_path = model_info['model_path']
        self.model_name = model_info['model_name']
        self.max_seqlen = 100
        self.w2v_model = gensim.models.Word2Vec.load(model_info['w2v_path'])

    def translate_to_word(self, outputs):
        eos_index = self.w2v_model.wv["<EOS>"]
        trans_temp = [[self.w2v_model.wv.most_similar(positive=[vec], topn=1)[0][0] for vec in data_element if
                       not np.array_equal(eos_index, vec)] for data_element in outputs]
        trans_result = [" ".join(data_element) for data_element in trans_temp]
        return trans_result

    def transform_data(self, mention):
        '''
        encoder에 필요한 data 생성
        :return1: input data for encoder, list of list which have vectorized word as elements (list)
        :return1: input data for decoder, list of list which have vectorized word as elements (list)
        '''
        enc_len = np.array([len(self.ko_engine.pos(sentence)) for sentence in mention])
        mention = list(np.array(mention)[enc_len < 100])
        enc_temp = [sentence for sentence in mention]
        enc_source = [[x[0] for x in self.ko_engine.pos(data_element)] for data_element in enc_temp]
        enc_source = [data_element + ['<PAD>'] * (self.max_seqlen - len(data_element)) for data_element in enc_source]
        enc_input = [data_element for data_element in enc_source]

        encX = []

        for i in range(0, len(enc_input)):
            enc_words = enc_input[i]
            x = [self.w2v_model.wv[word] for word in enc_words]
            encX.append(x)

        encX_len = [len(data_element) for data_element in encX]
        return encX, encX_len

    def predict(self, mention):
        encX, encX_len = self.transform_data(mention)
        go_index = [self.w2v_model.wv["<GO>"]]
        batch_size = len(encX)

        # 텐서플로우 그래프와 텐서 로드
        tf.reset_default_graph()
        with tf.Session() as sess:
            # 그래프와 메타 데이터 로드
            saver = tf.train.import_meta_graph(self.model_path + self.model_name + ".meta")
            saver.restore(sess, tf.train.latest_checkpoint(self.model_path))
            graph = tf.get_default_graph()

            # load tensor to use running model
            batch_len = graph.get_tensor_by_name("batch_len:0")
            eX = graph.get_tensor_by_name("encode/enc_X:0")
            eP = graph.get_tensor_by_name("encode/enc_prediction:0")  # 테스트하고 삭제
            eSeq = graph.get_tensor_by_name("encode/enc_seqlen:0")
            dX = graph.get_tensor_by_name("decode/dec_X:0")
            dP = graph.get_tensor_by_name("decode/dec_prediction:0")
            dSeq = graph.get_tensor_by_name("decode/dec_seqlen:0")

            # running model
            e_out = sess.run(eP, feed_dict={eX: encX, eSeq: encX_len, batch_len: batch_size})
            dec_input = [list(go_index) + list(enc_output[:-1]) for enc_output in e_out]
            # dec_input = [go_index + list(np.squeeze(enc_output)[:-1]) for enc_output in e_out]

            d_out = sess.run(dP, feed_dict={eX: encX, eSeq: encX_len, batch_len: batch_size,
                                            dX: dec_input, dSeq: self.max_seqlen})

            # translate prediction of index to natural language
            prediction_result = self.translate_to_word(d_out)
            return d_out

X = []
Y = []
with open("training_data/beauty_inside/beauty_inside.txt", mode="rb") as fp :
    X, Y, _, _ = pickle.load(fp)

test = dialogue_generator(model_path="./m20d500_loss0.09/", model_name="m20d500", w2v_path="word2vec/w2v_m20d500_bi")

for i in range(0, 30) :
    print("#"*40)
    print("input:",X[i])
    print("output:",Y[i])
    prediction = test.translate_to_word(test.predict([X[i]]))
    print("prediction:", prediction)