import numpy as np
import torch
import torch.nn as nn

from models.GeneralModel import GeneralModel
from models.networks.assisting_layers.ContainerLayers import ContainerLinear, ContainerConv2d
from utils.constants import ZERO_SIGMA


class Pruneable(GeneralModel):

    """

    Defines and manages a pruneable model and gathers statistics

    """

    def __init__(self,
                 is_maskable=True,
                 is_rewindable=True,
                 outer_layer_pruning=True,
                 device="cuda",
                 N=0,
                 maintain_outer_mask_anyway=False,
                 l2_reg=0.0,
                 **kwargs):
        self.hooks = {}
        self.l2_reg = l2_reg
        self.maintain_outer_mask_anyway = maintain_outer_mask_anyway
        self.N = N
        self._outer_layer_pruning = outer_layer_pruning
        self.device = device
        self.is_maskable = is_maskable
        self.is_rewindable = is_rewindable
        self.weight_count = 0
        self.deductable_weightcount = 0
        self._set_class_references()
        super(Pruneable, self).__init__(device=device, **kwargs)

    def add_hooks(self):
        def get_activation(name):
            def hook(model, input, output):
                # if self.hooks[name] is not None:
                #     self.hooks[name].append(output.detach())
                # else:
                if len(output.detach().shape) == 4:
                    temp = torch.mean(output.detach(), (2, 3)).T.detach().cpu().numpy()
                else:
                    temp = output.detach().T.detach().cpu().numpy()
                if name in self.hooks.keys():
                    self.hooks[name].append(temp)
                else:
                    self.hooks[name] = [temp]

            return hook

        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                module.register_forward_hook(get_activation(name))

    def get_num_nodes(self, init=False):
        counter = 0
        addition = 0
        for i, (name, module) in enumerate(self.named_modules()):
            if (hasattr(module, "weight") or hasattr(module, "weights")) and not ("Norm" in str(module.__class__)):
                addition = module.weight.shape[0]
                counter += addition
        else:
            return counter - addition  # not counting the last linear layer for classification?

    def _set_class_references(self):

        self.Linear = ContainerLinear
        self.Conv2d = ContainerConv2d

    def post_init_implementation(self):

        with torch.no_grad():
            self._num_nodes_start = self.get_num_nodes(init=True)  # get number of internal nodes (excluding last layer)
            self.weight_count = self._get_weight_count()  # number of weights (excluding bias i think)

            if self.is_maskable:
                self.mask = {name + ".weight": torch.ones_like(module.weight.data).to(self.device) for name, module in
                             self.named_modules() if isinstance(module, (nn.Linear, nn.Conv2d))
                             }

                if not self._outer_layer_pruning:
                    names = list(self.mask.keys())
                    self.first_layer_name = names[0]
                    self.last_layer_name = names[-1]
                    deductable = self.mask[names[0]].flatten().size()[0] + self.mask[names[-1]].flatten().size()[0]
                    self.percentage_fraction = self.weight_count / (1 + self.weight_count - deductable)

                    self.deductable_weightcount = deductable
                    if not self.maintain_outer_mask_anyway:
                        del self.mask[names[0]]
                        del self.mask[names[-1]]

            if self.is_rewindable:
                self.save_rewind_weights()

    def _get_weight_count(self):
        return sum([tens.flatten().size()[0] for name, tens in self.named_parameters() if
                    'weight' in name])

    def _clone_weights(self, weight_list_reference, bias=False):

        return {name: tens.data.detach().clone().to(self.device) for name, tens in
                weight_list_reference if
                ('weight' in name) or bias}

    def save_rewind_weights(self):
        """ Saves the weights used to rewind to"""

        if not self.is_rewindable:
            raise Exception("rewind weights is off")

        self.rewind_weights = self._clone_weights(self.named_parameters())

    def forward(self, x: torch.Tensor):
        raise NotImplementedError("please inherit child-class")

    def norm(self, p=2):

        regularisation = torch.zeros([1])
        for name, param in self.named_parameters():

            if "rho" in name:
                regularisation -= torch.norm((param == ZERO_SIGMA) * ZERO_SIGMA, p=p)
            regularisation += torch.norm(param, p=p)

        return regularisation.log()

    def apply_mask(self):
        self.apply_grad_mask()
        self.apply_weight_mask()

    def apply_weight_mask(self):

        if not self.is_maskable:
            raise Exception("mask is off")

        with torch.no_grad():
            for name, tensor in self.named_parameters():
                if name in self.mask:
                    tensor.data *= self.mask[name]
                    if "rho" in name:
                        tensor.data += (self.mask[name] == 0) * ZERO_SIGMA

    def apply_grad_mask(self):

        if not self.is_maskable:
            raise Exception("mask is off")

        for name, tensor in self.named_parameters():
            if name in self.mask:
                tensor.grad.data *= self.mask[name]

    def do_rewind(self):

        if not self.is_rewindable:
            raise Exception("rewind_weights is off")

        if not self.is_maskable:
            raise Exception("mask is off")

        with torch.no_grad():
            for name, tensor in self.named_parameters():
                if name in self.mask:
                    tensor.data = self.rewind_weights[name].detach().clone() * self.mask[name]
                    tensor.requires_grad = True

    def insert_noise_for_gradient(self, noise):
        if noise == 0:  return

        with torch.no_grad():
            for name, tensor in self.named_parameters():
                tensor.grad.data += noise * torch.randn_like(tensor.grad.data)

    @property
    def l2_norm(self):
        norm = self.norm(p=2).item()
        return norm

    @property
    def l1_norm(self):
        norm = self.norm(p=1).item()
        return norm

    @property
    def number_of_pruned_weights(self):

        if not self.is_maskable:
            return 0

        total = 0

        for name, tensor in self.named_parameters():
            if 'weight' in name:

                if 'rho' in name:
                    total += torch.sum(tensor == ZERO_SIGMA).item()
                else:
                    total += torch.sum(tensor == 0).item()
        return int(total)

    @property
    def pruned_percentage(self):
        # self.weight_count is the original number of weights
        # self.number_of_pruned_weights is the number of weights == 0 in the model
        return (self.number_of_pruned_weights + (self.weight_count - self._get_weight_count())) / (
                self.weight_count + 1e-6)

    @property
    def structural_sparsity(self):
        return 1.0 - ((self.get_num_nodes(init=False) + 1e-6) / (self._num_nodes_start + 1e-6))

    @property
    def get_params(self):
        params = self._clone_weights(self.named_parameters(), bias=True)
        return params

    @property
    def pruned_percentage_of_prunable(self):
        return self.number_of_pruned_weights / (self.weight_count - self.deductable_weightcount + 1e-6)

    @property
    def compressed_size(self):

        if not self.is_maskable and not self.l0:
            return np.nan
            # raise Exception("mask is off")

        size = 0
        with torch.no_grad():
            for name, tensor in self.named_parameters():
                if 'weight' in name:
                    nonzero = 0
                    if 'rho' in name:
                        nonzero = torch.sum(tensor != ZERO_SIGMA).item()
                    else:
                        nonzero = torch.sum(tensor != 0).item()
                    temp = tensor.view(tensor.shape[0], -1).detach()
                    m, n = temp.shape[0], temp.shape[1]
                    smallest = min(m, n)
                    size += nonzero * 34 + 2 * (smallest + 1)
        return size
