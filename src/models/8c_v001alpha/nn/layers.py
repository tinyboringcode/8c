# 8c-v0-beta/nn/layers.py - Warstwy sieci neuronowych

from core.tensor import Tensor
from graph.node import MatMul, Add, ReLU
from graph.node import Node
import random

class Linear:
    """
    Warstwa liniowa: y = xW + b
    """
    def __init__(self, in_features, out_features):
        self.weight = Tensor(
            [[random.uniform(-1, 1) for _ in range(out_features)] for _ in range(in_features)],
            requires_grad=True
        )
        self.bias = Tensor([0.0 for _ in range(out_features)], requires_grad=True)

    def __call__(self, x):
        out = Tensor(MatMul.forward(x.data, self.weight.data))
        node1 = Node(MatMul, [x, self.weight])
        out.set_creator(node1)

        out_b = Tensor(Add.forward(out.data, self.bias.data))
        node2 = Node(Add, [out, self.bias])
        out_b.set_creator(node2)
        return out_b


class ReLU_:
    """
    ReLU aktywacja: max(0, x)
    """
    def __call__(self, x):
        out = Tensor(ReLU.forward(x.data))
        node = Node(ReLU, [x])
        out.set_creator(node)
        return out


class Sequential:
    """
    Sekwencyjne łączenie warstw
    """
    def __init__(self, *layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


"""
models = Sequential(
    Linear(2, 4),
    ReLU_(),
    Linear(4, 1)
)

x = Tensor([[1.0, 2.0]], requires_grad=True)
y = models(x)
y.backward()

print("x.grad:", x.grad)

"""