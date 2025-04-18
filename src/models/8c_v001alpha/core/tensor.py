# 8c-v0-beta/core/tensor.py - Tensor z autograd i grafem operacji

from .array import Array
from .graph.node import Node, Add, Mul


class Tensor:
    """
    Tensor - podstawowa jednostka obliczeniowa z autograd i grafem zależności.
    Obsługuje operacje matematyczne oraz propagację gradientów przez graf.
    """

    def __init__(self, data, requires_grad=False, dtype=float):
        self.data = Array(data, dtype=dtype)
        self.requires_grad = requires_grad
        self.grad = None
        self.creator = None  # Node, który utworzył ten tensor

    def set_creator(self, node):
        self.creator = node

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(Add.forward(self.data, other.data))
        if self.requires_grad or other.requires_grad:
            node = Node(Add, [self, other])
            out.set_creator(node)
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(Mul.forward(self.data, other.data))
        if self.requires_grad or other.requires_grad:
            node = Node(Mul, [self, other])
            out.set_creator(node)
        return out

    def __sub__(self, other):
        return self + (other * -1)

    def __truediv__(self, other):
        return self * other ** -1

    def __pow__(self, power):
        from graph.node import Pow
        out = Tensor(Pow.forward(self, power))
        if self.requires_grad:
            node = Node(Pow, [self])
            out.set_creator(node)
        return out

    def backward(self, grad=None):
        if grad is None:
            grad = Array([1.0] * len(self.data))
        self.grad = grad

        visited = set()
        topo = []

        def build_topo(t):
            if t not in visited:
                visited.add(t)
                if t.creator:
                    for inp in t.creator.inputs:
                        build_topo(inp)
                    topo.append(t)

        build_topo(self)

        for t in reversed(topo):
            if t.creator:
                t.creator.backward(t.grad)

    def __str__(self):
        return f"Tensor(data={self.data}, requires_grad={self.requires_grad})"
