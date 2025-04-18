# 8c-v0-beta/train/trainer.py - Pętla treningowa

from optim.sgd import SGD
from train.losses import MSELoss

class Trainer:
    def __init__(self, model, lr=0.01):
        self.model = model
        self.loss_fn = MSELoss()
        self.optimizer = SGD(self._gather_params(model), lr=lr)

    def _gather_params(self, model):
        params = []
        for layer in getattr(model, 'layers', []):
            for attr in ['weight', 'bias']:
                if hasattr(layer, attr):
                    params.append(getattr(layer, attr))
        return params

    def train_step(self, x, y_true):
        y_pred = self.model(x)
        loss = self.loss_fn(y_pred, y_true)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss