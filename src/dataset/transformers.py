"""transformers' used to transform names into sequences of tokens"""

# Author: Ziqi Yuan <1564123490@qq.com>

import torch
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from torchvision.transforms import Compose

from src.utils.vocab import gloveVocabulary


class SentenceToIndex:
    def __init__(self, word_to_idx, tokenizer, unk_idx):
        self.word_to_idx = word_to_idx
        self.tokenizer = tokenizer
        self.unk_idx = unk_idx

    def __call__(self, string: str):
        word_sequence = self.tokenizer(string.lower())
        return np.array([self.word_to_idx.get(word, self.unk_idx) for word in word_sequence])


class PadToSize:
    def __init__(self, size, value, append_length):
        self.size = size
        self.value = value
        self.append_length = append_length

    def __call__(self, index_array):
        if len(index_array) < self.size:
            padded = np.pad(index_array, [0, self.size - len(index_array)], mode='constant', constant_values=self.value)
        else:
            padded = index_array[:self.size]

        if self.append_length:
            padded = np.hstack([padded, len(index_array)])
        return padded


class ToTensor:
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, x: np.ndarray):
        return torch.tensor(x)


class PadToSentenceLength:
    def __init__(self, max_sen, max_word, value):
        self.max_word = max_word
        self.value = value
        self.max_sen = max_sen

    def __call__(self, index_array):

        if len(index_array) < self.max_sen:
            extended_sentences = [[self.value for _ in range(self.max_word)] for _ in range(self.max_sen - len(index_array))]
            padded = np.vstack((index_array, extended_sentences))
        else:
            padded = index_array[:self.max_sen]
        return padded


class DocumentToPaddedIndex:
    def __init__(self, max_sen, max_word):
        self.transformer = Compose((
            SentenceToIndex(gloveVocabulary.word2idx, word_tokenize, gloveVocabulary.unk_idx),
            PadToSize(max_word, gloveVocabulary.unk_idx, False)
        ))
        self.PadSentence = PadToSentenceLength(max_sen, max_word, gloveVocabulary.unk_idx)

    def __call__(self, x: str):
        sentences = sent_tokenize(x)
        return torch.tensor(self.PadSentence(np.array([self.transformer(sentence) for sentence in sentences])))


if __name__ == "__main__":
    transformer_x = DocumentToPaddedIndex(4, 16)
    print(transformer_x("My dog loves the dyno stix."
                                " They stopped selling them at Petsmart so now I have to buy them online."))
