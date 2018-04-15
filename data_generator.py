import numpy as np
from keras.utils import Sequence
from pickle import load
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences


class DataGenerator(Sequence):
    def __init__(self, qa_list, batch_size=32, shuffle=True, vocab_size=50000, video_len=10000, subtitle_len=500,
                 qa_len=50):
        'Initialization'
        self.batch_size = batch_size
        self.qa_list = qa_list
        self.shuffle = shuffle
        self.vocab_size = vocab_size
        self.video_len = video_len
        self.subtitle_len = subtitle_len
        self.qa_len = qa_len
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.qa_list) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        qa_list_tmp = [self.qa_list[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(qa_list_tmp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.qa_list))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, qa_list_tmp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        video_base = r'D:\data_processed'
        subtt_base = r'D:\subtt'

        videos = []
        subtitles = []
        for i, qa in enumerate(qa_list_tmp, 0):
            video = []
            subtt = []
            for clip in qa.video_clips:
                abs_path = r'{}\{}features.p'.format(video_base, clip)
                frame_feature = load(open(abs_path, 'rb'))
                video.extend(frame_feature.reshape(-1, 512))
                abs_path = r'{}\{}.p'.format(subtt_base, clip)
                lines = load(open(abs_path, 'rb'))
                lines = [item for sublist in [one_hot(line, self.vocab_size) for line in lines] for item in sublist]
                subtt.extend(lines)
            video = np.expand_dims(np.array(video), axis=0)
            video = pad_sequences(video, maxlen=self.video_len)
            assert video.shape == (1, self.video_len, 512), "Video shape:{}".format(video.shape)
            videos.append(video.squeeze(axis=0))
            tmp = np.zeros((1, self.subtitle_len))
            if subtt != []:
                subtt = np.array(subtt).reshape(1, -1)
                tmp[:, -subtt.shape[1]:] = subtt[:, :self.subtitle_len]
            assert tmp.shape == (1, self.subtitle_len)
            subtitles.append(tmp)
        videos = np.array(videos)
        subtitles = np.squeeze(np.array(subtitles), axis=1)

        labels = []
        question_answers = []
        for qa in qa_list_tmp:
            tmp = np.zeros(5)
            tmp[qa.correct_index] = 1
            labels.append(tmp)
            question = qa.question
            qas2 = []
            for answer in qa.answers:
                tmp = np.zeros((1, self.qa_len))
                c = np.array(one_hot(question + answer, self.vocab_size)).reshape(1, -1)
                tmp[:, -c.shape[1]:] = c[:, :self.qa_len]
                qas2.append(tmp)
            np.array(qas2)
            question_answers.append(qas2)
        question_answers = np.array(question_answers).squeeze(2)
        labels = np.array(labels).reshape(-1, 5)
        return [videos, question_answers, subtitles], labels
