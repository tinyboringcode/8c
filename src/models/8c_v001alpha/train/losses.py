# 8c-v0-beta/train/losses.py - Funkcje straty
from core.tensor import Tensor
from graph.node import Sub, Pow, Mean, Node

class MSELoss:
    """
    Mean Squared Error Loss: L = mean((y_pred - y_true)^2)
    """
    def __call__(self, y_pred, y_true):
        diff = Tensor(Sub.forward(y_pred.data, y_true.data))
        node1 = Node(Sub, [y_pred, y_true])
        diff.set_creator(node1)

        sq = Tensor(Pow.forward(diff, 2))
        node2 = Node(Pow, [diff])
        sq.set_creator(node2)

        out = Tensor(Mean.forward(sq.data))
        node3 = Node(Mean, [sq])
        out.set_creator(node3)

        return out
