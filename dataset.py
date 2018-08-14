from utils import *


class DataSet:
    """A data set for a machine learning problem. It has the following fields:

    d.examples   A list of examples. Each one is a list of attribute values.
    d.attrnames  Optional list of mnemonic names for corresponding attrs.
    d.target     The attribute that a learning algorithm will try to predict.
                 By default the final attribute.
    d.name       Name of the data set (for output display only)."""

    def __init__(self, examples=None, attrnames=None, target=-1, name=''):
        self.name = name
        self.examples = examples

        # Attrs are the indices of examples, unless otherwise stated.
        self.attrs = list(range(len(self.examples[0])))

        # Initialize attrnames
        self.attrnames = attrnames or self.attrs

        # Initialize targets, inputs and values
        self.target = self.attrnum(target)
        self.inputs = [a for a in self.attrs if a != self.target]
        self.values = list(map(unique, zip(*self.examples)))

    def attrnum(self, attr):
        """Returns the number used for attr, which can be a name, or -n .. n-1."""
        if isinstance(attr, str):
            return self.attrnames.index(attr)
        elif attr < 0:
            return len(self.attrs) + attr
        else:
            return attr

    def sanitize(self, example):
        """Return a copy of example, with non-input attributes replaced by None."""
        return [attr_i if i in self.inputs else None for i, attr_i in enumerate(example)]
