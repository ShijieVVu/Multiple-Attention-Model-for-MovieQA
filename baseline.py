from keras.layers import Input, LSTM, Embedding, Dense, concatenate, Lambda, GRU, Activation
from keras.models import Model
from keras.optimizers import Adam
import numpy as np
from data_generator import DataGenerator

vocab_size = 400000
video_len = 10000
qa_len = 50
subtitle_len = 500
embedding_dim = 100

path = r'data\glove.6B.100d.txt'


def read_glove_vecs():
    with open(path, 'r+', encoding="utf8", errors='ignore') as f:
        lines = [line.rstrip().split(' ') for line in f.readlines()]
    word_to_index = {}
    word_to_vec_map = {}
    for i, line in enumerate(lines, 0):
        word_to_index[line[0]] = i
        word_to_vec_map[line[0]] = np.array([float(number) for number in line[1:]])
    return word_to_index, word_to_vec_map


def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    """
    Creates a Keras Embedding() layer and loads in pre-trained GloVe 50-dimensional vectors.

    Arguments:
    word_to_vec_map -- dictionary mapping words to their GloVe vector representation.
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    embedding_layer -- pretrained layer Keras instance
    """

    vocab_len = len(word_to_index) + 1  # adding 1 to fit Keras embedding (requirement)
    emb_dim = word_to_vec_map["cucumber"].shape[0]  # define dimensionality of your GloVe word vectors (= 50)

    emb_matrix = np.zeros((vocab_len, emb_dim))

    for word, index in word_to_index.items():
        emb_matrix[index, :] = word_to_vec_map[word]

    embedding_layer = Embedding(vocab_len, emb_dim, trainable=False, input_length=qa_len)

    embedding_layer.build((None,))

    embedding_layer.set_weights([emb_matrix])

    return embedding_layer


word_to_index, word_to_vec_map = read_glove_vecs()
encoder = pretrained_embedding_layer(word_to_vec_map, word_to_index)

# video_lstm = LSTM(256)
# text_lstm = LSTM(256)
video_lstm = GRU(embedding_dim)
text_lstm = GRU(embedding_dim)

vi = Input(shape=(video_len, 512))
subt = Input(shape=(subtitle_len,))
qa = Input(shape=(qa_len,))

vp = video_lstm(vi)  # the output will be a vector
qa1 = encoder(qa)
qa1 = text_lstm(qa1)
sub1 = encoder(subt)
sub1 = text_lstm(sub1)

merged = concatenate([vp, qa1, sub1])
x = Dense(100, activation='relu')(merged)
x = Dense(50, activation='relu')(x)
score = Dense(1)(x)
score_model = Model(inputs=[vi, qa, subt], outputs=score)
score_model.summary()


def baseModel():
    vi = Input(shape=(video_len, 512))
    subt = Input(shape=(subtitle_len,))
    qai = Input(shape=(5, qa_len))

    qas = [Lambda(lambda x: x[:, i, :], output_shape=(qa_len,))(qai) for i in range(5)]
    scores = []
    for i in range(5):
        score = score_model([vi, qas[i], subt])
        scores.append(score)
    prediction = Activation('softmax')(concatenate(scores))
    return Model([vi, qai, subt], outputs=prediction)


model = baseModel()

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
model.summary()
import sys

sys.path.insert(0, r'C:\Users\WUSHI\github\MovieQA_benchmark')
import data_loader

mqa = data_loader.DataLoader()
vl_qa, training_qas = mqa.get_video_list('train', 'qa_clips')
training_qa = training_qas[:4096]
training_generator = DataGenerator(training_qa, batch_size=16)
validation_qa = training_qas[4096:4224]
validation_generator = DataGenerator(validation_qa, batch_size=16)
model.fit_generator(generator=training_generator, validation_data=validation_generator, epochs=10,
                    use_multiprocessing=False, workers=1)
