# 8c.numic.array - Minimalistyczna implementacja typu Array bez NumPy

class Array:
    """
    Klasa Array - reprezentuje uproszczony typ tablicowy podobny do NumPy array.

    Obsługuje operacje matematyczne, reshape, konwersje typów danych i podstawowe statystyki.
    Działa całkowicie niezależnie od bibliotek zewnętrznych takich jak NumPy.
    """

    def __init__(self, data, dtype=float):
        """
        Inicjalizuje nową tablicę.
        :param data: lista (1D lub 2D) danych wejściowych
        :param dtype: typ danych (np. float, int, bool)
        """
        self.data = self._flatten(data)
        self.shape = self._infer_shape(data)
        self.dtype = dtype
        self.data = [self.dtype(x) for x in self.data]

    def _flatten(self, data):
        if isinstance(data[0], list):
            return [item for sublist in data for item in sublist]
        return data[:]

    def _infer_shape(self, data):
        if isinstance(data[0], list):
            return (len(data), len(data[0]))
        return (len(data),)

    def __add__(self, other):
        return self._elementwise_op(other, lambda x, y: x + y)

    def __sub__(self, other):
        return self._elementwise_op(other, lambda x, y: x - y)

    def __mul__(self, other):
        return self._elementwise_op(other, lambda x, y: x * y)

    def __truediv__(self, other):
        return self._elementwise_op(other, lambda x, y: x / y)

    def __pow__(self, power):
        """Podnosi każdy element tablicy do potęgi 'power'."""
        return self._elementwise_op(power, lambda x, y: x ** y)

    def dot(self, other):
        """
        Wykonuje iloczyn skalarny dwóch wektorów 1D.
        """
        assert len(self.shape) == 1 and len(other.shape) == 1, "dot działa tylko na wektorach 1D"
        assert self.shape[0] == other.shape[0], "Wektory muszą mieć ten sam rozmiar"
        return sum(a * b for a, b in zip(self.data, other.data))

    def _elementwise_op(self, other, op):
        if isinstance(other, Array):
            assert self.shape == other.shape, "Shapes must match for elementwise operations"
            result_data = [op(a, b) for a, b in zip(self.data, other.data)]
        else:
            result_data = [op(a, other) for a in self.data]
        return Array(self._reshape(result_data, self.shape), dtype=self.dtype)

    def _reshape(self, flat_data, shape):
        if len(shape) == 1:
            return flat_data
        out = []
        rows, cols = shape
        for i in range(rows):
            out.append(flat_data[i * cols:(i + 1) * cols])
        return out

    def reshape(self, new_shape):
        assert self._size(self.shape) == self._size(new_shape), "Incompatible shape for reshape"
        reshaped_data = self._reshape(self.data, new_shape)
        return Array(reshaped_data, dtype=self.dtype)

    def _size(self, shape):
        from functools import reduce
        from operator import mul
        return reduce(mul, shape, 1)

    def sum(self):
        return sum(self.data)

    def max(self):
        return max(self.data)

    def mean(self):
        return sum(self.data) / len(self.data)

    def matmul(self, other):
        assert len(self.shape) == 2 and len(other.shape) == 2, "matmul wymaga dwóch macierzy 2D"
        assert self.shape[1] == other.shape[0], "Niekompatybilne kształty do mnożenia macierzy"

        result = []
        for i in range(self.shape[0]):
            row = []
            for j in range(other.shape[1]):
                val = 0
                for k in range(self.shape[1]):
                    val += self.data[i * self.shape[1] + k] * other.data[k * other.shape[1] + j]
                row.append(val)
            result.append(row)
        return Array(result, dtype=self.dtype)

    def __str__(self):
        return f"Array(shape={self.shape}, dtype={self.dtype.__name__}): {self._reshape(self.data, self.shape)}"

    def astype(self, new_dtype):
        return Array(self._reshape([new_dtype(x) for x in self.data], self.shape), dtype=new_dtype)
