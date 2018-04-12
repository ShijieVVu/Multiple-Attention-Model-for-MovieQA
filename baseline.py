from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import Input, LSTM, Embedding, Dense, concatenate, Lambda, GRU
from keras.models import Model, Sequential
from keras.layers import TimeDistributed
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import text_to_word_sequence, one_hot
import numpy as np
from os import path

from pickle import load, dump
from my_classes import DataGenerator

vocab_size = 50000
video_len = 10000
qa_len = 40
subtitle_len = 500


def baseModel():
    encoder = Embedding(input_dim=vocab_size, output_dim=256, input_length=qa_len)
    # video_lstm = LSTM(256)
    # text_lstm = LSTM(256)
    video_lstm = GRU(256)
    text_lstm = GRU(256)

    vi = Input(shape=(video_len, 512))
    vp = video_lstm(vi)  # the output will be a vector
    qa = Input(shape=(qa_len,))
    qa1 = encoder(qa)
    qa1 = text_lstm(qa1)
    subt = Input(shape=(subtitle_len,))
    sub1 = encoder(subt)
    sub1 = text_lstm(sub1)

    merged = concatenate([vp, qa1, sub1])
    x = Dense(500, activation='relu')(merged)
    x = Dense(500, activation='relu')(x)
    score = Dense(1)(x)
    score_model = Model(inputs=[vi, qa, subt], outputs=score)

    # Next, let's define a language model to encode the question into a vector.
    # Each question will be at most 100 word long,
    # and we will index words as integers from 1 to 9999.
    qai = Input(shape=(5, qa_len))
    qas = [Lambda(lambda x: x[:, i, :], output_shape=(qa_len,))(qai) for i in range(5)]
    scores = []
    for i in range(5):
        score = score_model([vi, qas[i], subt])
        scores.append(score)
    prediction = Dense(5, activation='softmax')(concatenate(scores))
    return Model([vi, qai, subt], outputs=prediction)

model = baseModel()

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
# model.fit([videos, question_answers, subtitles], [labels], epochs=1, batch_size=10)
import sys
sys.path.insert(0, r'C:\Users\WUSHI\github\MovieQA_benchmark')
import data_loader
mqa = data_loader.DataLoader()
vl_qa, training_qas = mqa.get_video_list('train', 'qa_clips')
training_qa = training_qas[:2000]
training_generator = DataGenerator(training_qa, batch_size=8)
validation_qa = training_qas[2000:2032]
validation_generator = DataGenerator(validation_qa, batch_size=8)
model.fit_generator(generator=training_generator, validation_data=validation_generator, use_multiprocessing=False, workers=1)
