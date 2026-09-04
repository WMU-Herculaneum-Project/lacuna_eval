import math

# Imagine we are trying to fill in the lacuna in the following sentence:
# "The quick brown ___ jumps over the lazy dog."
# The system guesses, in descending order of fit:
# ["dog", "fox", "cat", "rabbit", "wolf"]
# And the correct answer is either {"fox", "cat"}
#
# For in_top_k, a correct answer is in the top 2 guesses, so the score is 1.0
# For top_1, a correct answer is NOT the top guess, so the score is 0.0
# For ndcg, the score is calculated as follows:
#   - The relevance of the guesses is [0, 1, 1, 0, 0]
#   - The discounted cumulative gain is 1/2 + 1/2 = 0.5
#   - The ideal discounted cumulative gain is 1/1 + 1/2


def relevance(guess, truth):
    """Returns 1 if the guess is in the truth set, 0 otherwise."""
    return float(guess in truth)


def discount(rank):
    """Discount function. Rank is 1-indexed."""
    return math.log(2, rank + 1)


def dcg(guesses, truth, k=None):
    """Discounted cumulative gain."""
    dcg_score = 0.0
    for i, guess in enumerate(guesses, 1):
        dcg_score += relevance(guess, truth) / discount(i)
        if k and i == k:
            break
    return dcg_score


def idcg(guesses, k=None):
    """Ideal discounted cumulative gain."""
    idcg_score = 0.0
    for i in range(len(guesses)):
        idcg_score += 1 / discount(i + 1)
        if k and i == k:
            break
    return idcg_score


def ndcg(guesses, truth, k=None):
    """Normalized discounted cumulative gain."""
    dcg_score = dcg(guesses, truth, k)
    idcg_score = idcg(guesses, k)

    return dcg_score / idcg_score if idcg_score > 0 else 0.0


def in_top_k(guesses, truth, k=None):
    """Returns 1 if the truth is in the top k guesses, 0 otherwise."""
    for i, guess in enumerate(guesses, 1):
        if guess in truth:
            return 1.0
        if k and i == k:
            break
    return 0.0


def top_1(guesses, truth):
    """Returns 1 if the truth is in the top guess, 0 otherwise."""
    return in_top_k(guesses, truth, 1)


def evaluate(guess_sets, truth_sets, k=None):
    """Aggregate ranking metrics over multiple guess and truth pairs.

    Returns the number of pairs and the mean DCG, NDCG, top-1 accuracy, and
    top-k accuracy. The two inputs must contain the same number of pairs.
    """
    if len(guess_sets) != len(truth_sets):
        raise ValueError("guess_sets and truth_sets must have the same length")

    pair_count = len(guess_sets)
    if pair_count == 0:
        return {
            "count": 0,
            "mean_dcg": 0.0,
            "mean_ndcg": 0.0,
            "top_1_accuracy": 0.0,
            "top_k_accuracy": 0.0,
        }

    dcg_scores = []
    ndcg_scores = []
    top_1_scores = []
    top_k_scores = []
    for guesses, truth in zip(guess_sets, truth_sets):
        dcg_scores.append(dcg(guesses, truth, k))
        ndcg_scores.append(ndcg(guesses, truth, k))
        top_1_scores.append(top_1(guesses, truth))
        top_k_scores.append(in_top_k(guesses, truth, k))

    return {
        "count": pair_count,
        "mean_dcg": sum(dcg_scores) / pair_count,
        "mean_ndcg": sum(ndcg_scores) / pair_count,
        "top_1_accuracy": sum(top_1_scores) / pair_count,
        "top_k_accuracy": sum(top_k_scores) / pair_count,
    }
