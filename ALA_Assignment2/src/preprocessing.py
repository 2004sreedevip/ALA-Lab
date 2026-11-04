
import re


STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to",
    "in", "on", "for", "is", "are", "was", "were",
    "with", "at", "by", "from", "it", "this",
    "that", "our", "we", "you", "they", "their",
    "has", "have", "had", "be", "been", "as",
    "will", "which", "who", "its", "into", "also",
    "about", "these", "those", "your", "us"
}


def preprocess_text(text):

    # Step 1: Convert to lowercase
    text = text.lower()

    # Step 2: Remove punctuation
    text = re.sub(r"[^\w\s]", " ", text)

    # Step 3: Split into tokens
    tokens = text.split()

    # Step 4: Remove stopwords
    tokens = [
        word for word in tokens
        if word not in STOPWORDS
    ]

    # Step 5: Remove words with length <= 2
    tokens = [
        word for word in tokens
        if len(word) > 2
    ]

    return tokens
