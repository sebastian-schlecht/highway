import abc

class Augmentation(object):
    """
    Apply a certain augmentation onto a set of data tensors.
    """

    @abc.abstractmethod
    def apply(self, values, deterministic=False):
        return