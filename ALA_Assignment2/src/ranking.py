
import numpy as np


def rank_tags(tag_names, S, in_vocab_tokens):
    """
    Rank tags using max pooling.

    Returns tags sorted in descending order
    of their maximum similarity score.
    """

    rankings = []

    for j, tag in enumerate(tag_names):

        # Find maximum similarity
        max_index = np.argmax(S[:, j])

        max_score = S[max_index, j]

        # Find the word responsible for maximum score
        best_word = in_vocab_tokens[max_index]

        rankings.append(
            (tag, max_score, best_word)
        )

    # Sort in descending order
    rankings.sort(
        key=lambda item: item[1],
        reverse=True
    )

    return rankings
