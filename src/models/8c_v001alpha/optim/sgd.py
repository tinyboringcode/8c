# 8c-v0-beta/optim/sgd.py - Prosty optymalizator SGD

class SGD:
    """
    Stochastic Gradient Descent (SGD) optimizer.
    """
    def __init__(self, parameters, lr=0.01):
        self.parameters = parameters  # lista Tensorów
        self.lr = lr

    def step(self):
        for p in self.parameters:
            if p.grad is not None:
                for i in range(len(p.data.data)):
                    p.data.data[i] -= self.lr * p.grad.data[i]

    def zero_grad(self):
        for p in self.parameters:
            p.grad = None
