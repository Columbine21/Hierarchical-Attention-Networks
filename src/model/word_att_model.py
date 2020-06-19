"""
@author: Jason Yuan <1564123490@qq.com>
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.utils import matrix_mul, element_wise_mul
from src.utils.vocab import gloveVocabulary
import numpy as np


class WordAttNet(nn.Module):
    def __init__(self, word2vec_path, hidden_size=100):
        super(WordAttNet, self).__init__()

        dict_len, embed_size = np.array(gloveVocabulary.dict).shape
        print("********************************")
        print(dict_len, embed_size)

        self.word_weight = nn.Parameter(torch.Tensor(2 * hidden_size, 2 * hidden_size))
        self.word_bias = nn.Parameter(torch.Tensor(1, 2 * hidden_size))
        self.context_weight = nn.Parameter(torch.Tensor(2 * hidden_size, 1))

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
        output = F.softmax(output, dim=1)
        output = element_wise_mul(f_output, output.permute(1, 0))

        return output, h_output


if __name__ == "__main__":
    abc = WordAttNet("../data/glove.6B.50d.txt")
