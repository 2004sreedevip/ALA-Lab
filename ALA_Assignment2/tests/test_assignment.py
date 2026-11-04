
import unittest
import numpy as np

from src.vector import Vector

from src.matrices import (
    build_tag_matrix,
    build_text_matrix
)

from src.similarity import (
    similarity_matrix_vector_class,
    similarity_matrix_numpy
)


class FakeModel:

    def __init__(self):

        self.key_to_index = {
            "research": 0,
            "education": 1,
            "technology": 2,
            "science": 3,
            "university": 4,
            "learning": 5
        }

        self.vectors = {}

        for i, word in enumerate(self.key_to_index):
            vector = np.zeros(50)
            vector[i] = 1.0
            self.vectors[word] = vector

    def __getitem__(self, word):
        return self.vectors[word]


class TestAssignment(unittest.TestCase):

    def setUp(self):

        self.model = FakeModel()

        self.tags = [
            "research",
            "education",
            "technology",
            "science",
            "university",
            "learning"
        ] + ["research"] * 14

    def test_tag_matrix_shape(self):

        _, T = build_tag_matrix(
            self.model, self.tags
        )

        self.assertEqual(T.shape, (20, 50))

    def test_text_matrix_shape(self):

        _, _, W = build_text_matrix(
            self.model,
            ["research", "education"]
        )

        self.assertEqual(W.shape, (2, 50))

    def test_similarity_matrix_shape(self):

        _, T = build_tag_matrix(
            self.model, self.tags
        )

        _, _, W = build_text_matrix(
            self.model,
            ["research", "education"]
        )

        S = similarity_matrix_numpy(W, T)

        self.assertEqual(S.shape, (2, 20))

    def test_vector_similarity(self):

        v1 = Vector([1, 2, 3])
        v2 = Vector([1, 2, 3])

        self.assertAlmostEqual(
            v1.cosine_similarity(v2), 1.0
        )

    def test_both_methods_match(self):

        _, T = build_tag_matrix(
            self.model, self.tags
        )

        _, _, W = build_text_matrix(
            self.model,
            ["research", "education"]
        )

        S_vector = similarity_matrix_vector_class(W, T)

        S_numpy = similarity_matrix_numpy(W, T)

        self.assertTrue(
            np.allclose(S_vector, S_numpy)
        )

    def test_oov_token_handling(self):

        in_vocab, oov, W = build_text_matrix(
            self.model,
            ["research", "unknownword"]
        )

        self.assertEqual(in_vocab, ["research"])

        self.assertEqual(oov, ["unknownword"])

        self.assertEqual(W.shape, (1, 50))

    def test_invalid_tag_raises_error(self):

        invalid_tags = (
            ["research"] * 19 + ["unknownword"]
        )

        with self.assertRaises(ValueError):

            build_tag_matrix(
                self.model, invalid_tags
            )


if __name__ == "__main__":
    unittest.main()
