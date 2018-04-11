from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import Input, LSTM, Embedding, Dense, concatenate, Lambda
from keras.models import Model, Sequential
from keras.layers import TimeDistributed
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import text_to_word_sequence, one_hot
import numpy as np
from os import path

from pickle import load, dump

vocab_size = 50000
video_len = 10000
qa_len = 40
subtitle_len = 500


def baseModel():
    encoder = Embedding(input_dim=vocab_size, output_dim=256, input_length=qa_len)
    video_lstm = LSTM(256)
    text_lstm = LSTM(256)

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


def load_data(vocab_size=50000, video_len=10000, qa_len=40, subtitle_len=200):
    """
    Return the training video frames, questions and answers based on qa index.
    :return:
    """
    if path.isfile(r"D:\train_data.p"):
        return load(open(r'D:\train_data.p', 'rb'))
    video_base = r'D:\data_processed'
    subtt_base = r'D:\subtt'
    import sys
    sys.path.insert(0, r'C:\Users\WUSHI\github\MovieQA_benchmark')
    import data_loader
    mqa = data_loader.DataLoader()
    vl_qa, qas = mqa.get_video_list('train', 'qa_clips')

    qas = qas[:3]

    videos = []
    subtitles = []
    for i, qa in enumerate(qas, 0):
        if i % 100 == 0:
            print(i)
        video = []
        subtt = []
        for clip in qa.video_clips:
            abs_path = r'{}\{}features.p'.format(video_base, clip)
            frame_feature = load(open(abs_path, 'rb'))
            video.extend(frame_feature.reshape(-1, 512))
            abs_path = r'{}\{}.p'.format(subtt_base, clip)
            lines = load(open(abs_path, 'rb'))
            lines = [item for sublist in [one_hot(line, vocab_size) for line in lines] for item in sublist]
            subtt.extend(lines)
        video = np.expand_dims(np.array(video), axis=0)
        video = pad_sequences(video, maxlen=video_len)
        videos.append(video.squeeze(axis=0))
        tmp = np.zeros((1, subtitle_len))
        subtt = np.array(subtt).reshape(1, -1)
        tmp[:, -subtt.shape[1]:] = subtt[:, :subtitle_len]
        subtitles.append(tmp)
    videos = np.array(videos)
    subtitles = np.squeeze(np.array(subtitles), axis=1)

    labels = []
    question_answers = []
    for qa in qas:
        tmp = np.zeros(5)
        tmp[qa.correct_index] = 1
        labels.append(tmp)
        question = qa.question
        qas2 = []
        for answer in qa.answers:
            tmp = np.zeros((1, qa_len))
            c = np.array(one_hot(question + answer, vocab_size)).reshape(1, -1)
            tmp[:, -c.shape[1]:] = c
            qas2.append(tmp)
        np.array(qas2)
        question_answers.append(qas2)
    question_answers = np.array(question_answers).squeeze(2)
    labels = np.array(labels).reshape(-1, 5)

    dump((videos, subtitles, question_answers, labels), open(r"D:\train_data.p", 'wb'))
    return videos, subtitles, question_answers, labels


videos, subtitles, question_answers, labels = load_data(vocab_size=vocab_size, video_len=video_len, qa_len=qa_len,
                                                        subtitle_len=subtitle_len)
model = baseModel()

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
model.fit([videos, question_answers, subtitles], [labels], epochs=1, batch_size=16)
