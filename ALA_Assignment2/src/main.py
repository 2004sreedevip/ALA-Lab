
import numpy as np
import gensim.downloader as api
from pathlib import Path
import re

from src.preprocessing import preprocess_text
from src.matrices import TAGS, build_tag_matrix, build_text_matrix

from src.similarity import (
    similarity_matrix_vector_class,
    similarity_matrix_numpy
)

from src.ranking import rank_tags


def main():

    # Find project directory
    BASE_DIR = Path(__file__).resolve().parent.parent

    # Input text file
    text_file = BASE_DIR / "data" / "linkedin_text.txt"

    if not text_file.exists():
        raise FileNotFoundError(
            "linkedin_text.txt not found"
        )

    # Read input text
    raw_text = text_file.read_text(encoding="utf-8")

    # Count original tokens
    original_tokens = re.findall(r"\b\w+\b", raw_text)

    print("\nLoading GloVe model...")

    # Load pretrained GloVe model
    model = api.load("glove-wiki-gigaword-50")

    print("GloVe model loaded successfully!")

    # Preprocess text
    tokens = preprocess_text(raw_text)

    print("\n========== TOKEN STATISTICS ==========")

    print("Tokens before preprocessing:",
          len(original_tokens))

    print("Tokens after preprocessing:",
          len(tokens))

    # Build tag matrix
    tag_names, T = build_tag_matrix(model, TAGS)

    # Build text matrix
    in_vocab_tokens, oov_tokens, W = build_text_matrix(
        model, tokens
    )

    print("In-vocabulary tokens:",
          len(in_vocab_tokens))

    print("Out-of-vocabulary tokens:",
          len(oov_tokens))

    print("Distinct OOV tokens:",
          sorted(set(oov_tokens)))

    # Check for usable words
    if len(in_vocab_tokens) == 0:
        raise ValueError(
            "No usable in-vocabulary words found"
        )

    print("\n========== MATRIX DIMENSIONS ==========")

    print("T.shape =", T.shape)
    print("W.shape =", W.shape)

    # Method 1: Vector class
    print("\nCalculating using Vector class...")

    S_vector = similarity_matrix_vector_class(W, T)

    # Method 2: NumPy
    print("Calculating using NumPy...")

    S_numpy = similarity_matrix_numpy(W, T)

    # Verification
    print("\n========== VERIFICATION ==========")

    result = np.allclose(
        S_vector,
        S_numpy,
        rtol=1e-5,
        atol=1e-8
    )

    print("Both implementations match:", result)

    max_difference = np.max(
        np.abs(S_vector - S_numpy)
    )

    print("Maximum absolute difference:",
          max_difference)

    # Rank tags
    rankings = rank_tags(
        tag_names,
        S_numpy,
        in_vocab_tokens
    )

    print("\n========== TOP 8 TAGS ==========")

    print(
        f"{'Rank':<8}{'Tag':<22}"
        f"{'Score':<15}{'Best matching word'}"
    )

    print("-" * 70)

    for rank, (tag, score, word) in enumerate(
        rankings[:8], start=1
    ):

        match_type = (
            "Identical" if tag == word
            else "Different"
        )

        print(
            f"{rank:<8}{tag:<22}"
            f"{score:<15.4f}{word} ({match_type})"
        )


if __name__ == "__main__":
    main()
