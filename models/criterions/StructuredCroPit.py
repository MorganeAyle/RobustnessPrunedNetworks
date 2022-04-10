from models.criterions.StructuredCroP import StructuredCroP


class StructuredCroPit(StructuredCroP):

    """
    Original creation from our paper:  https://arxiv.org/abs/2006.00896
    Implements SNAP-it (before training)
    SNAP-it (before training) provides computational benefits from the start of training
    """

    def __init__(self, *args, limit=0.0, lower_limit=0.5, steps=5, **kwargs):
        self.limit = limit
        super(StructuredCroPit, self).__init__(*args, **kwargs)
        if limit <= lower_limit:
            raise ValueError("lower_limit should be lower than limit.")
        self.steps = [limit - (limit - lower_limit) * (0.5 ** i) for i in range(steps + 1)] + [limit]
        self.left = 1.0
        self.pruned = 0.0

    def get_prune_indices(self, *args, **kwargs):
        raise NotImplementedError

    def get_grow_indices(self, *args, **kwargs):
        raise NotImplementedError

    def prune(self, percentage=0.0, *args, **kwargs):
        while len(self.steps) > 0:

            # get k_i
            percentage = self.steps.pop(0)
            prune_now = (percentage - self.pruned) / (self.left + 1e-8)

            # prune
            super().prune(percentage=prune_now, *args, **kwargs)

            # adjust
            self.pruned = self.model.structural_sparsity  # percentage
            self.left = 1.0 - self.pruned
