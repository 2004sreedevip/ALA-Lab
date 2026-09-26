import math
import random
from typing import Self


class Vec:

    def __init__(self, src=None) -> Self:
        if src is None:
            self.elements = ()
        else:
            elements = tuple(src)

            for x in elements:
                if not isinstance(x, (int, float)):
                    raise TypeError(
                        f"Scalar must be a number: {type(x)}"
                    )

            self.elements = elements

    def __add__(self, t: Self) -> Self:
        if not isinstance(t, Vec):
            raise TypeError(f"Expected Vec: {type(t)}")

        if len(self.elements) != len(t.elements):
            raise TypeError(
                "Vectors must be of same dimensions"
            )

        return Vec([
            round(x + y, 5)
            for x, y in zip(self.elements, t.elements)
        ])

    def __mul__(self, scalar: int | float) -> Self:
        if not isinstance(scalar, (int, float)):
            raise TypeError(
                f"Vector multiplication with invalid type: {type(scalar)}"
            )

        return Vec([
            round(x * scalar, 5)
            for x in self.elements
        ])

    def __rmul__(self, scalar: int | float) -> Self:
        return self.__mul__(scalar)

    def __imul__(self, scalar: int | float) -> Self:
        if not isinstance(scalar, (int, float)):
            raise TypeError(
                f"Vector multiplication with invalid type: {type(scalar)}"
            )

        self.elements = tuple(round(val * scalar, 5) for val in self.elements)
        return self

    def __repr__(self) -> str:
        return repr(self.elements)

    def __len__(self) -> int:
        return len(self.elements)

    def __sub__(self, t: Self) -> Self:
        if not isinstance(t, Vec):
            raise TypeError("Expected Vec")

        if len(self.elements) != len(t.elements):
            raise TypeError(
                "Vectors must be of same dimensions"
            )

        return Vec([
            round(x - y, 5)
            for x, y in zip(self.elements, t.elements)
        ])

    def __neg__(self) -> Self:
        return Vec([
            -x for x in self.elements
        ])

    def __radd__(self, other):
        if other == 0:
            return self

        raise TypeError(
            "Cannot add Vec with this type"
        )

    def __iadd__(self, other: Self) -> Self:
        if not isinstance(other, Vec):
            raise TypeError("Expected Vec")

        if len(self.elements) != len(other.elements):
            raise TypeError(
                "Vectors must be of same dimensions"
            )

        self.elements = tuple(
            x + y for x, y in zip(self.elements, other.elements)
        )
        return self

    @staticmethod
    def zeros(n: int) -> Self:
        return Vec([0] * n)

    @staticmethod
    def ones(n: int) -> Self:
        return Vec([1] * n)

    @staticmethod
    def uniform(n: int) -> Self:
        return Vec([
            random.uniform(0, 1)
            for _ in range(n)
        ])

    def mean(self) -> float:
        if not self.elements:
            return 0.0
        return sum(self.elements) / len(self.elements)

    def demean(self) -> Self:
        m = self.mean()
        return Vec([x - m for x in self.elements])

    def std(self) -> float:
        if len(self.elements) == 0:
            return 0.0
        if len(self.elements) == 1:
            return 0.0

        m = self.mean()
        variance = sum((x - m) ** 2 for x in self.elements) / len(self.elements)
        return math.sqrt(variance)

    def norm(self) -> float:
        return math.sqrt(
            sum(x * x for x in self.elements)
        )