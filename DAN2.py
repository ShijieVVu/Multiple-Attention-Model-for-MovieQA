from keras.layers import Input, LSTM, Embedding, Dense, concatenate, Lambda, GRU, Activation, Dot, RepeatVector, \
    multiply, average, add, merge, Bidirectional
from keras.models import Model, load_model
from keras.optimizers import Adam, SGD
from keras import backend as tf
from keras.callbacks import ModelCheckpoint
import numpy as np

video_len = 5000
video_dim = 512

vocab_size = 5000
qa_len = 50
subtitle_len = 150
embedding_dim = 10

embedding_path = '/media/shijie/Users/WUSHI/github/Multiple-Attention-Model-for-MovieQA/data/glove.6B.100d.txt'

K = 3
hidden_size = 10
memory_size = 2 * hidden_size

def read_glove_vecs():
    with open(embedding_path, 'r+', encoding="utf8", errors='ignore') as f:
        lines = [line.rstrip().split(' ') for line in f.readlines()]
    word_to_index = {}
    word_to_vec_map = {}
    for i, line in enumerate(lines, 0):
        word_to_index[line[0]] = i
        word_to_vec_map[line[0]] = np.array([float(number) for number in line[1:]])
    return word_to_index, word_to_vec_map


def pretrained_embedding_layer(word_to_vec_map, word_to_index, length, embedding_dim):
    vocab_len = len(word_to_index) + 1  # adding 1 to fit Keras embedding (requirement)
    emb_matrix = np.zeros((vocab_len, embedding_dim))
    for word, index in word_to_index.items():
        emb_matrix[index, :] = word_to_vec_map[word]
    embedding_layer = Embedding(input_dim=vocab_len, output_dim=embedding_dim, trainable=True)
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
        m = Input(shape=(memory_size,))
        weight_input = Dense(hidden_size, activation='tanh')
        weight_memory = Dense(hidden_size, activation='tanh')
        weight_hidden = Dense(1)
        multiplication = multiply
        extraction = Lambda(connection, arguments={'i': 0})
        alphas = []
        for i in range(vector_len):
            if i % 1000 == 0:
                print(i)
            extraction.arguments = {'i': i}
            vi = extraction(v)
            hidden = multiplication([weight_input(vi), weight_memory(m)])
            alpha = weight_hidden(hidden)
            alphas.append(alpha)
        alphas = concatenate(alphas)
        # attented_v = merge([v, alphas], output_shape=vector_len, mode='mul')
        attented_v = Lambda(apply_weights, arguments={'length': dimension})([v, alphas])
        self.model = Model([v, m], outputs=attented_v)


def connection(x, i):
    return x[:, i, :]


# Shared layers
extraction = Lambda(connection, arguments={'i': 0})
pair_average = Lambda(lambda x: tf.mean(x, axis=1))
word_to_index, word_to_vec_map = read_glove_vecs()
embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim)
# embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index, qa_len, embedding_dim)
qa_encoder = embedding_layer
sub_encoder = embedding_layer

p0 = Dense(memory_size, activation='tanh')
video_shaper = Dense(memory_size, activation='tanh')
# V_Att = Attention(video_len, video_dim).model
# print('finished V_Att')
S_Att = Attention(subtitle_len, memory_size).model
print('finished S_Att')
Q_Att = Attention(qa_len, memory_size).model
print('finished Q_Att')

# video_lstm = LSTM(256, return_sequences=True)
# text_lstm = LSTM(256, return_sequences=True)
# video_lstm = GRU(units=embedding_dim, return_sequences=True)
lstm_layer = Bidirectional(LSTM(units=hidden_size, return_sequences=True), merge_mode='concat')
qa_lstm = lstm_layer
sub_lstm = lstm_layer


class ScoreModel:
    def __init__(self):
        # vi = Input(shape=(video_len, video_dim))
        qa = Input(shape=(qa_len,))
        subt = Input(shape=(subtitle_len,))

        # vp = video_lstm(vi)  # the output will be a vector
        qa1 = qa_encoder(qa)
        qa1 = qa_lstm(qa1)
        sub1 = sub_encoder(subt)
        sub1 = sub_lstm(sub1)

        # v0 = p0(pair_average(vp))
        u0 = pair_average(sub1)
        q0 = pair_average(qa1)

        # v = v0
        u = u0
        q = q0

        # uv = multiply([v, u])
        uq = multiply([u, q])
        # vq = multiply([v, q])

        # m = average([uv, uq, vq])
        m = uq

        for _ in range(K):
            # v = video_shaper(V_Att([vi, m]))
            u = S_Att([sub1, m])
            q = Q_Att([qa1, m])

            # Multi-modal Fusion
            # uv = multiply([v, u])
            uq = multiply([u, q])
            # vq = multiply([v, q])

            # m = add([m, average([uv, uq, vq])])
            m = add([m, uq])

        score = Dense(1)(m)
        self.model = Model(inputs=[qa, subt], outputs=score)
        print('finished score_model')
        self.model.summary()


def DAN():
    score_model = ScoreModel().model

    # vi = Input(shape=(video_len, video_dim))
    subt = Input(shape=(subtitle_len,))
    qai = Input(shape=(5, qa_len))

    qa = []
    for i in range(5):
        extraction.arguments = {'i': i}
        qa.append(extraction(qai))
    scores = []
    for i in range(5):
        score = score_model([qa[i], subt])
        scores.append(score)
    prediction = Activation('softmax')(concatenate(scores))
    return Model([qai, subt], outputs=prediction)


model = DAN()
print('finished overall')

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
# model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.1, momentum=0.9, decay=0.0005, nesterov=False, clipvalue=0.1), metrics=['accuracy'])
model.summary()
model.save('./model/dan.h5')

import sys
from data_generator import DataGenerator
sys.path.insert(0, '/media/shijie/Users/WUSHI/github/MovieQA_benchmark')
import data_loader

batch_size = 64
epochs = 50
workers = 2

mqa = data_loader.DataLoader()
training_qas = mqa.get_video_list('train', 'qa_clips')[1]
validation_qas = mqa.get_video_list('val', 'qa_clips')[1]
nb_train_samples = len(training_qas)
nb_validation_samples = len(validation_qas)
training_generator = DataGenerator(training_qas, batch_size=batch_size, vocab_size=vocab_size, video_len=video_len, subtitle_len=subtitle_len, qa_len=qa_len)
validation_generator = DataGenerator(validation_qas, batch_size=batch_size, vocab_size=vocab_size, video_len=video_len, subtitle_len=subtitle_len, qa_len=qa_len)
check_point = ModelCheckpoint('./model/dan2.{epoch:02d}-{val_acc:.4f}.h5', save_best_only=True)

print("starting training")
model.fit_generator(training_generator,
                    steps_per_epoch=nb_train_samples // batch_size,
                    epochs=epochs,
                    callbacks=[check_point],
                    validation_data=validation_generator,
                    validation_steps=nb_validation_samples // batch_size,
                    workers=workers,
                    use_multiprocessing=True)
