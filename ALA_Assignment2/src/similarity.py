
import numpy as np
from src.vector import Vector


def similarity_matrix_vector_class(W, T):
    """
    Calculate cosine similarity using Vector class.
    """

    n_words = W.shape[0]
    n_tags = T.shape[0]

    S = []

    for i in range(n_words):

        row = []

        for j in range(n_tags):

            word_vector = Vector(W[i])
            tag_vector = Vector(T[j])

            similarity = word_vector.cosine_similarity(
                tag_vector
            )

            row.append(similarity)

        S.append(row)

    return np.array(S)


def similarity_matrix_numpy(W, T):
    """
    Calculate cosine similarity using NumPy.
    """

    # Calculate row norms
    W_norm = np.linalg.norm(
        W, axis=1, keepdims=True
    )

    T_norm = np.linalg.norm(
        T, axis=1, keepdims=True
    )

    # Check for zero vectors
    if np.any(W_norm == 0) or np.any(T_norm == 0):
        raise ValueError("Zero vector found")

    # Normalize rows
    W_hat = W / W_norm
    T_hat = T / T_norm

    # Matrix multiplication
    S = W_hat @ T_hat.T

    return S
