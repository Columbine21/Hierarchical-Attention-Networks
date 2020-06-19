import pandas as pd
import heapq

import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn import metrics
from poutyne.framework import Model, Callback
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.distributions import Categorical



from src.dataset.AmazonData import AmazonReviews
from src.model import HierAttNet


class ClipGradient(Callback):
    def __init__(self, module: HierAttNet, clip):
        super().__init__()
        self.clip = clip
        self.module = module

    def on_batch_end(self, batch, logs):
        torch.nn.utils.clip_grad_value_(self.module.parameters(), self.clip)


class GenerateCallback(Callback):
    def __init__(self, net: HierAttNet, device, n=10, every=10):
        super().__init__()
        self.net = net
        self.device = device
        self.n = n
        self.every = every

    def on_batch_begin(self, batch_number, logs):
        self.net._init_hidden_state()
    # def on_epoch_begin(self, epoch_number, logs):


    def on_epoch_end(self, epoch, logs):
        if epoch % self.every == 0:
            self.net.train(False)
            # Todo: test on dev dataset.

            self.net.train(True)


def get_evaluation(y_true, y_prob, list_metrics):
    y_pred = np.argmax(y_prob, -1)
    print(y_pred, y_true)
    output = {}
    if 'accuracy' in list_metrics:
        output['accuracy'] = metrics.accuracy_score(y_true, y_pred)
    if 'loss' in list_metrics:
        try:
            output['loss'] = metrics.log_loss(y_true, y_prob)
        except ValueError:
            output['loss'] = -1
    if 'confusion_matrix' in list_metrics:
        output['confusion_matrix'] = str(metrics.confusion_matrix(y_true, y_pred))
    return output


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = AmazonReviews("../dataset/AmazonReviews/train.csv")
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    net = HierAttNet(100, 100, 32, 5, None, 8, 24)
    # def count_parameters(model):
    #     for name, param in model.named_parameters():
    #         print(name)
    #         print(param)
    #     return sum(p.numel() for p in model.parameters() if p.requires_grad)
    #
    #
    # print(f'The model has {count_parameters(net):,} trainable parameters')

    optimizer = Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=1e-3)
    criterion = CrossEntropyLoss()

    model = Model(net, optimizer, criterion, batch_metrics=['accuracy']).to(device)
    history = model.fit_generator(loader, epochs=300, validation_steps=0,
                                  callbacks=[ClipGradient(net, 2), GenerateCallback(net, device)])
    df = pd.DataFrame(history).set_index('epoch')
    df.plot(subplots=True)
    plt.show()
