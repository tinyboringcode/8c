# 8c-v0-beta/graph/node.py - Węzły grafu obliczeń
from core.array import Array

class Node:
    """
    Podstawowa reprezentacja węzła grafu obliczeń.
    Każdy Node odpowiada jednej operacji i przechowuje informacje o:
    - jego wejściach
    - funkcji forward i backward
    - wyjściowym tensorze
    """
    def __init__(self, op, inputs):
        self.op = op              # operacja np. Add, Mul, Pow, MatMul, ReLU
        self.inputs = inputs      # lista tensorów-wejść
        self.output = None        # wynik operacji

    def forward(self):
        self.output = self.op.forward(*self.inputs)
        return self.output

    def backward(self, grad):
        grads = self.op.backward(grad, *self.inputs)
        for inp, g in zip(self.inputs, grads):
            if inp.requires_grad:
                if inp.grad is None:
                    inp.grad = g
                else:
                    inp.grad = inp.grad + g


class Op:
    """
    Klasa bazowa dla operacji w grafie.
    Każda operacja musi zdefiniować metodę forward() i backward().
    """
    @staticmethod
    def forward(*args):
        raise NotImplementedError

    @staticmethod
    def backward(out_grad, *args):
        raise NotImplementedError


class Add(Op):
    @staticmethod
    def forward(a, b):
        return a + b

    @staticmethod
    def backward(out_grad, a, b):
        return [out_grad, out_grad]


class Mul(Op):
    @staticmethod
    def forward(a, b):
        return a * b

    @staticmethod
    def backward(out_grad, a, b):
        return [out_grad * b, out_grad * a]


class Pow(Op):
    @staticmethod
    def forward(a, p):
        return a ** p

    @staticmethod
    def backward(out_grad, a):
        return [out_grad * a * 2]  # domyślnie p = 2


class MatMul(Op):
    @staticmethod
    def forward(a, b):
        return a.matmul(b)

    @staticmethod
    def backward(out_grad, a, b):
        return [out_grad.matmul(b.transpose()), a.transpose().matmul(out_grad)]


class ReLU(Op):
    @staticmethod
    def forward(x):
        return x._elementwise_op(0, lambda a, _: max(0, a))

    @staticmethod
    def backward(out_grad, x):
        relu_grad = x._elementwise_op(0, lambda a, _: 1.0 if a > 0 else 0.0)
        return [out_grad * relu_grad]


class Mean(Op):
    @staticmethod
    def forward(x):
        return Array([x.mean()])

    @staticmethod
    def backward(out_grad, x):
        grad = Array([out_grad.data[0] / len(x.data)] * len(x.data))
        return [grad]


class Sum(Op):
    @staticmethod
    def forward(x):
        return Array([x.sum()])

    @staticmethod
    def backward(out_grad, x):
        grad = Array([out_grad.data[0]] * len(x.data))
        return [grad]


class Transpose(Op):
    @staticmethod
    def forward(x):
        rows, cols = x.shape
        reshaped = [[x[i * cols + j] for i in range(rows)] for j in range(cols)]
        return Array(reshaped)

    @staticmethod
    def backward(out_grad, x):
        return [Transpose.forward(out_grad)]
