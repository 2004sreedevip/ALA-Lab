
import math


class Vector:

    def __init__(self, values):
        self.values = tuple(float(x) for x in values)

    def __add__(self, other):
        if len(self.values) != len(other.values):
            raise ValueError("Vector dimensions must match")

        return Vector(
            a + b for a, b in zip(self.values, other.values)
        )

    def scale(self, scalar):
        return Vector(x * scalar for x in self.values)

    def dot(self, other):
        if len(self.values) != len(other.values):
            raise ValueError("Vector dimensions must match")

        return sum(
            a * b for a, b in zip(self.values, other.values)
        )

    def norm(self):
        return math.sqrt(self.dot(self))

    def cosine_similarity(self, other):
        denominator = self.norm() * other.norm()

        if denominator == 0:
            raise ValueError("Cosine similarity is undefined")

        return self.dot(other) / denominator
