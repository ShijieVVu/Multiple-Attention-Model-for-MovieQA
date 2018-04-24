from pickle import dump,load
import os

base = '/media/shijie/Users/WUSHI/github/Multiple-Attention-Model-for-MovieQA/data/subtt'
# base = '/home/shijie/Downloads/subtt'

word_bag = set([])
word2index = {}
index2word = {}
index = 0
#directory = os.fsencode(base)
for file in os.scandir(base):
    filename = os.fsdecode(file)
    path = os.path.join(base, filename)
    lines = load(open(path, 'rb'))
    for line in lines:
        for word in line.split():
            word = word.lower()
            if word not in word_bag:
                word_bag.add(word)
                word2index[word] = index
                index2word[index] = word
                index += 1

# print(word2index)
# print(index2word)
word2index['<UNK>'] = index
index2word[index] = '<UNK>'
print('length is {}'.format(index + 1))
            
dump(word2index, open("./data/word2index.p", 'wb'))
dump(index2word, open("./data/index2word.p", 'wb'))
