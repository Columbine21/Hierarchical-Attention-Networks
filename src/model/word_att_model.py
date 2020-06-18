"""
@author: Jason Yuan <1564123490@qq.com>
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.utils import matrix_mul, element_wise_mul
from src.utils.vocab import gloveVocabulary
import pandas as pd
import numpy as np
import csv


class WordAttNet(nn.Module):
    def __init__(self, word2vec_path, hidden_size=100):
        super(WordAttNet, self).__init__()

        # dict = pd.read_csv(filepath_or_buffer=word2vec_path, header=None, sep=" ", quoting=csv.QUOTE_NONE).values[:, 1:]
        dict_len, embed_size = np.array(gloveVocabulary.dict).shape
        print("********************************")
        print(dict_len, embed_size)
        # dict_len += 1
        # unknown_word = np.zeros((1, embed_size))
        # dict = torch.from_numpy(np.concatenate([unknown_word, dict], axis=0).astype(np.float))

        self.word_weight = nn.Parameter(torch.zeros(2 * hidden_size, 2 * hidden_size), requires_grad=True)
        self.word_bias = nn.Parameter(torch.zeros(1, 2 * hidden_size), requires_grad=True)
        self.context_weight = nn.Parameter(torch.zeros(2 * hidden_size, 1), requires_grad=True)

        self.lookup = nn.Embedding(num_embeddings=dict_len, embedding_dim=embed_size).\
            from_pretrained(torch.tensor(np.array(gloveVocabulary.dict)))
        self.gru = nn.GRU(embed_size, hidden_size, bidirectional=True)
        self._create_weights(mean=0.0, std=0.05)

    def _create_weights(self, mean=0.0, std=0.05):

        self.word_weight.data.normal_(mean, std)
        self.context_weight.data.normal_(mean, std)

    def forward(self, input, hidden_state):
        output = self.lookup(input)
        f_output, h_output = self.gru(output.float(), hidden_state)  # feature output and hidden state output
        output = matrix_mul(f_output, self.word_weight, self.word_bias)
        output = matrix_mul(output, self.context_weight).permute(1, 0)
        # print(output.shape)
        output = F.softmax(output, dim=1)
        output = element_wise_mul(f_output, output.permute(1, 0))

        return output, h_output


if __name__ == "__main__":
    abc = WordAttNet("../data/glove.6B.50d.txt")
