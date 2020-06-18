"""Base vocabulary map & EOS & UNK (PAD) token"""

# Author: Ziqi Yuan <1564123490@qq.com>


class Vocab:
    def __init__(self, pretrain_embedding_path):
        self.vocab = []
        self.dict = []
        with open(pretrain_embedding_path) as file:
            for index, line in enumerate(file):
                word, embedding = line.split()[0], line.split()[1:]
                embedding = [float(v) for v in embedding]
                self.vocab.append(word)
                self.dict.append(embedding)
        self.unk_idx = len(self.vocab)
        self.vocab.append('<unk>')
        self.dict.append([0 for i in range(100)])
        self.vocab_size = len(self.vocab)
        self.word2idx = {word: index for index, word in enumerate(self.vocab)}
        self.idx2word = {index: word for index, word in enumerate(self.vocab)}


# Note : to run this module(python file alone use '../../dataset/glove.6B.100d.txt' instead)
gloveVocabulary = Vocab("../dataset/glove.6B.100d.txt")

if __name__ == '__main__':

    test_vocab = Vocab('../../dataset/glove.6B.100d.txt')
    # print(test_vocab.vocab[37], test_vocab.vocab[14])
    print(test_vocab.dict[40000])
    # print(np.array())
    # print(test_vocab.idx2word[test_vocab.vocab_size - 1])

