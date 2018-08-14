import copy
import random
from collections import Counter
from dataset import DataSet


def randomly_remove_values(dataset, p):
    """ Return a new dataset where each value from the original dataset is removed with probability p. """

    # dataset.examples is a list of lists, so we have to use deep copy
    # in order to copy by value and not reference.
    examples = copy.deepcopy(dataset.examples)

    for example in examples:
        for index in range(len(example)):

            # Return the next random floating point number in the range [0.0, 1.0).
            rng = random.random()

            # Remove value if rng <= p
            if index != dataset.target and rng <= p:
                example[index] = None

    return DataSet(name=dataset.name, examples=examples, attrnames=dataset.attrnames, target=dataset.target)


def handle_missing_values(dataset):
    """ Deal with missing values by assigning the most common value among training examples at that node. """

    # Collect every single value for each attribute from the dataset examples
    all_values = [[] for _ in range(len(dataset.values))]
    for example in dataset.examples:
        for i in range(len(example)):
            all_values[i].append(example[i])

    # Filter the most common value for each attribute
    most_common_values = []
    for value in all_values:
        c = Counter(value)
        del c[None]  # don't count missing values
        most_common_values.append(c.most_common(1)[0][0])

    # Where value is missing, assign the most common value for that attribute
    for example in dataset.examples:
        for i in range(len(example)):
            if example[i] is None:
                example[i] = most_common_values[i]
