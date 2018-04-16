from keras.layers import Input, LSTM, Embedding, Dense, concatenate, Lambda, GRU, Activation, Dot, RepeatVector, \
    multiply, average, add
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras import backend as tf
import numpy as np

vocab_size = 400000
video_len = 5000
video_dim = 512
qa_len = 50
subtitle_len = 200
embedding_dim = 100

path = r'data\glove.6B.100d.txt'

K = 2
memory_len = 100

qa_pairs = []
video_features = []
subtitles = []
labels = []


def read_glove_vecs():
    with open(path, 'r+', encoding="utf8", errors='ignore') as f:
        lines = [line.rstrip().split(' ') for line in f.readlines()]
    word_to_index = {}
    word_to_vec_map = {}
    for i, line in enumerate(lines, 0):
        word_to_index[line[0]] = i
        word_to_vec_map[line[0]] = np.array([float(number) for number in line[1:]])
    return word_to_index, word_to_vec_map


def pretrained_embedding_layer(word_to_vec_map, word_to_index, length, embedding_dim):
    """
    Creates a Keras Embedding() layer and loads in pre-trained GloVe 50-dimensional vectors.

    Arguments:
    word_to_vec_map -- dictionary mapping words to their GloVe vector representation.
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    embedding_layer -- pretrained layer Keras instance
    """

    vocab_len = len(word_to_index) + 1  # adding 1 to fit Keras embedding (requirement)

    emb_matrix = np.zeros((vocab_len, embedding_dim))

    for word, index in word_to_index.items():
        emb_matrix[index, :] = word_to_vec_map[word]

    embedding_layer = Embedding(input_dim=vocab_len, output_dim=embedding_dim, trainable=False, input_length=length)

    embedding_layer.build((None,))

    embedding_layer.set_weights([emb_matrix])

    return embedding_layer


def apply_weights(x, length):
    v = x[0]
    alphas = x[1]
    return tf.sum(v * tf.tile(tf.expand_dims(alphas, axis=2), [1, 1, length]), axis=1)


class Attention:
    def __init__(self, vector_len, dimension):
        def connection(x, i):
            return x[:, i, :]

        v = Input(shape=(vector_len, dimension))
        m = Input(shape=(memory_len,))
        alphas = []
        weight_input = Dense(1, activation='tanh')
        weight_memory = Dense(1, activation='tanh')
        weight_hidden = Dense(1, activation='softmax')
        multiplication = multiply
        extraction = Lambda(connection, arguments={'i': 0})
        for i in range(vector_len):
            if i % 1000 == 0:
                print(i)
            extraction.arguments = {'i': i}
            vi = extraction(v)
            hidden = multiplication([weight_input(vi), weight_memory(m)])
            alpha = weight_hidden(hidden)
            alphas.append(alpha)
        alphas = concatenate(alphas)
        attented_v = Lambda(apply_weights, arguments={'length': dimension})([v, alphas])
        self.model = Model([v, m], outputs=attented_v)


def connection(x, i):
    return x[:, i, :]


# Shared layers
extraction = Lambda(connection, arguments={'i': 0})
pair_average = Lambda(lambda x: tf.mean(x, axis=1))
word_to_index, word_to_vec_map = read_glove_vecs()
qa_encoder = pretrained_embedding_layer(word_to_vec_map, word_to_index, qa_len, embedding_dim)
sub_encoder = pretrained_embedding_layer(word_to_vec_map, word_to_index, subtitle_len, embedding_dim)
p0 = Dense(memory_len, activation='tanh')
video_shaper = Dense(memory_len, activation='tanh')
V_Att = Attention(video_len, video_dim).model
print('finished V_Att')
S_Att = Attention(subtitle_len, embedding_dim).model
print('finished S_Att')
Q_Att = Attention(qa_len, embedding_dim).model
print('finished Q_Att')

# video_lstm = LSTM(256, return_sequences=True)
# text_lstm = LSTM(256, return_sequences=True)
video_lstm = GRU(units=embedding_dim, return_sequences=True)
qa_lstm = GRU(units=embedding_dim, return_sequences=True)
sub_lstm = GRU(units=embedding_dim, return_sequences=True)


class ScoreModel:
    def __init__(self):
        vi = Input(shape=(video_len, video_dim))
        qa = Input(shape=(qa_len,))
        subt = Input(shape=(subtitle_len,))

        vp = video_lstm(vi)  # the output will be a vector
        qa1 = qa_encoder(qa)
        qa1 = qa_lstm(qa1)
        sub1 = sub_encoder(subt)
        sub1 = sub_lstm(sub1)

        v0 = p0(pair_average(vp))
        u0 = pair_average(sub1)
        q0 = pair_average(qa1)

        v = v0
        u = u0
        q = q0

        uv = multiply([v, u])
        uq = multiply([u, q])
        vq = multiply([v, q])

        m = average([uv, uq, vq])

        for _ in range(K):
            v = video_shaper(V_Att([vi, m]))
            u = S_Att([sub1, m])
            q = Q_Att([qa1, m])

            # Multi-modal Fusion
            uv = multiply([v, u])
            uq = multiply([u, q])
            vq = multiply([v, q])

            m = add([m, average([uv, uq, vq])])

        x = Dense(100, activation='relu')(m)
        x = Dense(50, activation='relu')(x)
        score = Dense(1)(x)
        self.model = Model(inputs=[vi, qa, subt], outputs=score)
        print('finished score_model')


def DAN():
    score_model = ScoreModel().model

    vi = Input(shape=(video_len, video_dim))
    subt = Input(shape=(subtitle_len,))
    qai = Input(shape=(5, qa_len))

    qa = []
    for i in range(5):
        extraction.arguments = {'i': i}
        qa.append(extraction(qai))
    scores = []
    for i in range(5):
        score = score_model([vi, qa[i], subt])
        scores.append(score)
    prediction = Activation('softmax')(concatenate(scores))
    return Model([vi, qai, subt], outputs=prediction)


model = DAN()
print('finished overall')

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
# model.summary()
# model.save('./model/dan.h5')

import sys
from data_generator import DataGenerator
sys.path.insert(0, r'C:\Users\WUSHI\github\MovieQA_benchmark')
import data_loader

mqa = data_loader.DataLoader()
vl_qa, training_qas = mqa.get_video_list('train', 'qa_clips')
training_qa = training_qas[:1024]
training_generator = DataGenerator(training_qa, batch_size=16, vocab_size=vocab_size, video_len=video_len, subtitle_len=subtitle_len, qa_len=qa_len)
validation_qa = training_qas[4096:4224]
validation_generator = DataGenerator(validation_qa, batch_size=16, vocab_size=vocab_size, video_len=video_len, subtitle_len=subtitle_len, qa_len=qa_len)
print("starting training")
model.fit_generator(generator=training_generator, validation_data=validation_generator, epochs=10,
                    use_multiprocessing=False, workers=1)
