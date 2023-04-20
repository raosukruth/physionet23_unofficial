import torch
from torch import nn as nn
import numpy as np
import torch.nn.functional as fn

class MLP(object):
    def __init__(self, D_in, D_out, hidden_layer_sizes=(1000,), activation=nn.ReLU,
                 solver=torch.optim.Adam, loss_fn=nn.CrossEntropyLoss(),
                 learning_rate=1e-4, max_iter=200):
        self.D_in = D_in
        self.D_out = D_out
        model = nn.Sequential()
        for i in range(len(hidden_layer_sizes)):
            actf = activation
            if i == 0:
                num_in = D_in
            else:
                num_in = hidden_layer_sizes[i - 1]
            num_out = hidden_layer_sizes[i]
            model.append(nn.Linear(num_in, num_out))
            model.append(activation())
        model.append(nn.Linear(hidden_layer_sizes[-1], D_out))
        model.append(nn.Softmax(dim=1))
        print("Model: ", model)

        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = solver(model.parameters(), lr=learning_rate)
        self.lr = learning_rate
        self.epochs = max_iter

    def is_np_ndarray(self, X):
        nd_dummy = np.ndarray(10)
        return type(X) == type(nd_dummy)

    def to_torch(self, m, dtype):
        return torch.from_numpy(m).to(dtype)

    def one_hot_encoding(self, m, num_classes):
        return fn.one_hot(m, num_classes=num_classes)

    def fit(self, X, y):
        y = y.ravel()
        self.model.train(True)
        if self.is_np_ndarray(X):
            X = self.to_torch(X, torch.float32)
        if self.is_np_ndarray(y):
            y = self.to_torch(y, torch.int64)
        y = self.one_hot_encoding(y, 2)
        y = y.to(torch.float32)
        for _ in range(self.epochs):
            y_pred = self.model(X)
            loss = self.loss_fn(y_pred, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.model.train(False)
        return self

    def predict_proba(self, X):
        self.model.train(False)
        if self.is_np_ndarray(X):
            X = self.to_torch(X, torch.float32)
        return self.model(X).detach().numpy()

    def predict(self, X):
        output = self.predict_proba(X)
        y_pred = (np.argmax(output, axis=1), np.argmin(output, axis=1))
        return np.ravel(y_pred)


