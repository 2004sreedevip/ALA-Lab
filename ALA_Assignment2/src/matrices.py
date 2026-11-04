
import numpy as np


TAGS = [
    "research",
    "innovation",
    "education",
    "university",
    "students",
    "faculty",
    "campus",
    "engineering",
    "medicine",
    "technology",
    "learning",
    "academic",
    "collaboration",
    "publication",
    "laboratory",
    "scholarship",
    "internship",
    "entrepreneurship",
    "alumni",
    "science"
]


def build_tag_matrix(model, tags):
    """
    Construct the tag matrix T.

    Returns:
        tag_names: list of 20 tags
        T: NumPy array with shape (20, 50)
    """

    if len(tags) != 20:
        raise ValueError("Exactly 20 tags are required")

    tag_vectors = []

    for tag in tags:

        words = tag.lower().split()

        vectors = []

        for word in words:

            if word not in model.key_to_index:
                raise ValueError(
                    f"Tag word not in vocabulary: {word}"
                )

            vectors.append(model[word])

        # Average vectors for multi-word tags
        tag_vector = np.mean(vectors, axis=0)

        tag_vectors.append(tag_vector)

    T = np.array(tag_vectors)

    return tags, T


def build_text_matrix(model, tokens):
    """
    Construct the text matrix W.

    Returns:
        in_vocab_tokens
        oov_tokens
        W with shape (n, 50)
    """

    in_vocab_tokens = []
    oov_tokens = []

    for word in tokens:

        if word in model.key_to_index:
            in_vocab_tokens.append(word)

        else:
            oov_tokens.append(word)

    if len(in_vocab_tokens) > 0:

        W = np.array([
            model[word] for word in in_vocab_tokens
        ])

    else:
        W = np.empty((0, 50))

    return in_vocab_tokens, oov_tokens, W
