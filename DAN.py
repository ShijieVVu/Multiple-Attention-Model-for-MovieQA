# Path to your downloaded MovieQA directory
MOVIEQA_PATH = "/Users/shijiewu/github/MovieQA_benchmark"
# Path to your video feature output directory
OUTPUT_PATH = ""

from keras import Model
from keras.layers import Input, Embedding, LSTM, Bidirectional, RepeatVector, Dense, Dot, average, multiply, concatenate
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import one_hot

K = 2
vocab_size = 10000
output_dim = 300
max_len = 1000
memory_len = text_len = 300
image_len = 512

qa_pairs = []
video_features = []
subtitles = []
labels = []

# Map word to indices
word2index = {}


def V_Att(vn, mv):
    """
    Perform visual attention: Output a context vector computed as tanh of the output of a dense layer with input of the
    dot product of the attention weights "alphas" and vn.

    :param vn: The visual context vector, numpy array of shape (N, image_len)
    :param mv: The memory unit, numpy array of shape (image_len, )
    :return: vk: context vector, numpy array of shape (memory_len, )
    """
    num_cases = vn.shape[0]
    mv = RepeatVector(num_cases)(mv)
    alphas = []
    WvVn = Dense(1, activation='tanh')
    Wvmmv = Dense(1, activation='tanh')
    WvhHvn = Dense(1, activation='softmax')
    for i in range(num_cases):
        wv = WvVn(vn)
        wvm = Wvmmv(mv)
        hvn = Dot(wv, wvm)
        avn = WvhHvn(hvn)
        alphas.append(avn)
    vk = Dense(memory_len, activation='tanh')(multiply([alphas, vn]))
    return vk


def T_Att(ut, mu):
    """
    Perform textual attention by focusing on specific words in the input sequence every step.
    Output a context vector computed as dot product of attention weights "alphas" and ut

    :param ut: The textual context vector, numpy array of shape(T, text_len)
    :param mu: The memory unit, numpy array of shape (text_len, )
    :return:
    """
    num_words = ut.shape[0]
    mu = RepeatVector(num_words)(mu)
    alphas = []
    WvVn = Dense(1, activation='tanh')
    Wvmmv = Dense(1, activation='tanh')
    WvhHvn = Dense(1, activation='softmax')
    for i in range(num_words):
        wv = WvVn(ut)
        wvm = Wvmmv(mu)
        hvn = Dot(wv, wvm)
        avn = WvhHvn(hvn)
        alphas.append(avn)
    uk = multiply([alphas, ut])
    return uk


def score_model(vn, text):
    xt = Embedding(vocab_size, output_dim=output_dim, input_length=max_len)(text)
    ut = Bidirectional(LSTM(memory_len))(xt)

    v0 = Dense(memory_len, activation='tanh')(average(vn))
    u0 = average(ut)

    m = multiply([v0, u0])

    for _ in range(K):
        # Visual Attention
        vk = V_Att(vn, m)

        # Textual Attention
        uk = T_Att(ut, m)

        m = m + multiply([vk, uk])

    score = Dense(1, activation='softmax')(m)
    return score


def model(n_v, n_w):
    questions = Input()
    answers = Input()
    # Image representation
    vn = Input()

    # Text representation
    texts = concatenate([RepeatVector(5)(questions), answers])
    scores = []
    for i in range(5):
        score = score_model(vn, texts[i])
        scores.append(score)

    prediction = Dense(1, activation='softmax')(scores)
    return Model([vn, questions, answers], outputs=prediction)


encoded_docs = [one_hot(d, vocab_size) for d in subtitles]
padded_subtitles = pad_sequences(encoded_docs, maxlen=max_len, padding='pre')

model = model(512, 128)
opt = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy'])
model.fit([video_features, padded_subtitles], labels, epochs=1, batch_size=32)
