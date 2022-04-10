from utils.system_utils import *
import torch
import torch.nn as nn
from copy import deepcopy
import types
from utils.hydra_utils import linear_forward, conv_forward
from models.criterions.HYDRA import HYDRA


class Edgepop(HYDRA):
    def __init__(self, arguments, local_pruning, *args, **kwargs):
        super(Edgepop, self).__init__(arguments, local_pruning, *args, **kwargs)

    def foo(self):
        self.prune_model = deepcopy(self.model)
        for param in self.prune_model.parameters():
            param.requires_grad = False

        for name, layer in self.prune_model.named_modules():

            if name + ".weight" in self.prune_model.mask:
                gov = nn.Parameter(torch.ones_like(layer.weight), requires_grad=True)
                torch.nn.init.kaiming_normal_(
                    gov.data, mode='fan_in', nonlinearity='relu'
                )
                layer.gov = gov

                if self.local_pruning:
                    layer.prune_rate = self.percentage

                if isinstance(layer, nn.Linear):
                    layer.forward = types.MethodType(linear_forward, layer)

                elif isinstance(layer, nn.Conv2d):
                    layer.forward = types.MethodType(conv_forward, layer)