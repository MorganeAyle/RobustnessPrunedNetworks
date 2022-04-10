from models.criterions.SNIP import SNIP
from copy import deepcopy


class SNIPit(SNIP):

    """
    Original creation from our paper:  https://arxiv.org/abs/2006.00896
    Implements SNIP-it (before training)
    """

    def __init__(self, *args, limit=0.0, steps=5, lower_limit=0.5, **kwargs):
        self.limit = limit
        super(SNIPit, self).__init__(*args, **kwargs)
        if limit <= lower_limit:
            raise ValueError("lower_limit should be lower than limit.")
        # always smaller than limit, steps+1 elements (including limit)
        self.steps = [limit - (limit - lower_limit) * (0.5 ** i) for i in range(steps - 1)] + [limit]

    def get_prune_indices(self, *args, **kwargs):
        raise NotImplementedError

    def get_grow_indices(self, *args, **kwargs):
        raise NotImplementedError

    def prune(self, percentage=0.0, *args, **kwargs):
        self.grads_abs = None
        while len(self.steps) > 0:

            # determine k_i
            percentage = self.steps.pop(0)

            criterion = SNIP(model=self.model)

            # prune
            criterion.prune(percentage=percentage, *args, **kwargs)
