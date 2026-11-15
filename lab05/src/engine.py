"""A tiny scalar automatic-differentiation engine used by Lab 05."""

from __future__ import annotations

import math
from typing import Callable, Iterable


class Value:
    """A scalar value and its reverse-mode autograd bookkeeping."""

    def __init__(
        self,
        data: float,
        _children: Iterable["Value"] = (),
        _op: str = "",
    ) -> None:
        self.data = float(data)
        self.grad = 0.0
        self._prev = tuple(_children)
        self._backward: Callable[[], None] = lambda: None
        self._op = _op

    @staticmethod
    def _coerce(other: "Value | float | int") -> "Value":
        return other if isinstance(other, Value) else Value(float(other))

    def __add__(self, other: "Value | float | int") -> "Value":
        other = self._coerce(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other: "Value | float | int") -> "Value":
        return self + other

    def __neg__(self) -> "Value":
        return self * -1.0

    def __sub__(self, other: "Value | float | int") -> "Value":
        return self + (-self._coerce(other))

    def __rsub__(self, other: "Value | float | int") -> "Value":
        return self._coerce(other) - self

    def __mul__(self, other: "Value | float | int") -> "Value":
        other = self._coerce(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other: "Value | float | int") -> "Value":
        return self * other

    def __pow__(self, exponent: int | float) -> "Value":
        if not isinstance(exponent, (int, float)):
            raise TypeError("Value powers require an int or float exponent")
        out = Value(self.data**exponent, (self,), f"**{exponent}")

        def _backward() -> None:
            self.grad += exponent * (self.data ** (exponent - 1)) * out.grad

        out._backward = _backward
        return out

    def __truediv__(self, other: "Value | float | int") -> "Value":
        return self * (self._coerce(other) ** -1)

    def __rtruediv__(self, other: "Value | float | int") -> "Value":
        return self._coerce(other) / self

    def relu(self) -> "Value":
        out = Value(max(0.0, self.data), (self,), "ReLU")

        def _backward() -> None:
            self.grad += (out.data > 0.0) * out.grad

        out._backward = _backward
        return out

    def exp(self) -> "Value":
        # Inputs in the lab's sigmoid stay in a moderate range.  Mapping
        # overflow to infinity preserves the mathematical operation and makes
        # failures visible instead of silently changing its derivative.
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward() -> None:
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def log(self) -> "Value":
        if self.data <= 0.0:
            raise ValueError("log is defined only for positive Value.data")
        out = Value(math.log(self.data), (self,), "log")

        def _backward() -> None:
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    def backward(self) -> None:
        """Backpropagate from this node without clearing existing gradients."""
        topo: list[Value] = []
        visited: set[int] = set()

        def build(node: Value) -> None:
            marker = id(node)
            if marker in visited:
                return
            visited.add(marker)
            for parent in node._prev:
                build(parent)
            topo.append(node)

        build(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    def __repr__(self) -> str:
        return f"Value(data={self.data:.6g}, grad={self.grad:.6g})"
