from dt_learner import DecisionTreeLearner
import random


def err_ratio(predict, dataset, examples=None):
    """Return the proportion of the examples that are NOT correctly predicted."""
    examples = examples or dataset.examples
    if len(examples) == 0:
        return 0.0
    right = 0
    for example in examples:
        desired = example[dataset.target]
        output = predict(dataset.sanitize(example))
        if output == desired:
            right += 1
    return 1 - (right / len(examples))


def train_test_split(dataset, start, end):
    """Reserve dataset.examples[start:end] for test; train on the remainder."""
    start = int(start)
    end = int(end)
    examples = dataset.examples
    train = examples[:start] + examples[end:]
    val = examples[start:end]
    return train, val


def cross_validation(dataset, k=10, trials=10):
    """Do k-fold cross_validate and return their mean.
    That is, keep out 1/k of the examples for testing on each of k runs.
    Shuffle the examples first; if trials>1, average over several shuffles.
    Returns Training error, Validation error"""
    k = k or len(dataset.examples)
    if trials > 1:
        trial_errT = 0
        trial_errV = 0
        for t in range(trials):
            errT, errV = cross_validation(dataset, k=10, trials=1)
            trial_errT += errT
            trial_errV += errV
        return trial_errT / trials, trial_errV / trials
    else:
        fold_errT = 0
        fold_errV = 0
        n = len(dataset.examples)
        examples = dataset.examples
        for fold in range(k):
            random.shuffle(dataset.examples)
            train_data, val_data = train_test_split(dataset, fold * (n / k), (fold + 1) * (n / k))
            dataset.examples = train_data
            h = DecisionTreeLearner(dataset)
            fold_errT += err_ratio(h, dataset, train_data)
            fold_errV += err_ratio(h, dataset, val_data)

            # Reverting back to original once test is completed
            dataset.examples = examples
        return fold_errT / k, fold_errV / k
