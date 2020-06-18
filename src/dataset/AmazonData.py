"""Amazon Reviews Dataset"""

# Author: Ziqi Yuan <1564123490@qq.com>

import pandas as pd
from typing import Tuple, List, Dict
from torch.utils.data.dataset import Dataset
from src.dataset.transformers import DocumentToPaddedIndex
from torch.utils.data import DataLoader
import numpy as np


class AmazonReviews(Dataset):
    def __init__(self, path: str, max_length_sen=8, max_length_word=24):
        super(AmazonReviews, self).__init__()
        self.data = pd.read_csv(path)
        self.max_length_sen = max_length_sen
        self.max_length_word = max_length_word

        self.transformer = DocumentToPaddedIndex(self.max_length_sen, self.max_length_word)

        self.num_classes = len(set(list(self.data['Score'])))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentences = self.data.iloc[index]['Text']
        input = self.transformer(sentences)
        target = self.data.iloc[index]['Score'] - 1
        return input, target


if __name__ == '__main__':
    Dataset = AmazonReviews("../../dataset/AmazonReviews/train.csv")

