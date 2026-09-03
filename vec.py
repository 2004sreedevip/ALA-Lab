import math
import random
from typing import Self


class Vec:

    def __init__(self, src=None) -> Self:
        if src is None:
            self.elements = []
        else:
            elements = list(src)

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

    def __rmul__(self, scalar: int | float) -> Self:
        if not isinstance(scalar, (int, float)):
            raise TypeError(
                f"Vector multiplication with invalid type: {type(scalar)}"
            )

        return Vec([
            round(x * scalar, 5)
            for x in self.elements
        ])

    def __imul__(self, scalar: int | float) -> Self:
        if not isinstance(scalar, (int, float)):
            raise TypeError(
                f"Vector multiplication with invalid type: {type(scalar)}"
            )

        for i, val in enumerate(self.elements):
            self.elements[i] = round(val * scalar, 5)

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

        for i in range(len(self.elements)):
            self.elements[i] += other.elements[i]

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

    def norm(self) -> float:
        return math.sqrt(
            sum(x * x for x in self.elements)
        )