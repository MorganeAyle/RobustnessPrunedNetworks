from models.criterions.General import General
from models import GeneralModel
from utils.model_utils import *
from utils.system_utils import *
import torch
from utils.attacks_utils import construct_adversarial_examples
import torch.nn as nn
from copy import deepcopy
import torch.nn.functional as F
import types
from utils.hydra_utils import linear_forward, conv_forward, calculate_fan_in
import math
from torch.distributions import Categorical
import numpy as np
from utils.metrics import calculate_auroc

from torchvision import transforms


class HYDRA(General):
    def __init__(self, arguments, local_pruning, *args, **kwargs):
        self.arguments = arguments
        super(HYDRA, self).__init__(*args, **kwargs)
        self.post_init()
        self.loss = nn.CrossEntropyLoss()
        self.percentage = self.arguments["pruning_limit"]
        self.local_pruning = local_pruning

    def post_init(self):
        # get model
        arguments = self.arguments

        # load data
        self.train_loader, self.val_loader, self.test_loader = find_right_model(
            DATASETS, arguments['data_set'],
            arguments=arguments, val=True
        )

        # load unnormalized data
        (self.train_loader_unnormalized, self.val_loader_unnormalized, self.test_loader_unnormalized), mean, std = find_right_model(
            DATASETS, arguments['data_set'] + '_unnormalized',
            arguments=arguments, val=True, mean=arguments['mean'],
            std=arguments['std']
        )
        self.transform = transforms.Normalize(mean, std)

        # lood OOD data
        self.ood_loader_train, self.ood_loader_val, self.ood_loader_test = find_right_model(
            DATASETS, arguments['ood_prune_data_set'],
            arguments=arguments, val=True
        )

    def foo(self):
        self.prune_model = deepcopy(self.model)
        for param in self.prune_model.parameters():
            param.requires_grad = False

        for name, layer in self.prune_model.named_modules():

            if name + ".weight" in self.prune_model.mask:
                fan_in = calculate_fan_in(layer.weight)
                init_weights = layer.weight * math.sqrt(6/fan_in) / torch.max(torch.abs(layer.weight))
                gov = nn.Parameter(init_weights, requires_grad=True)
                layer.gov = gov

                if self.local_pruning:
                    layer.prune_rate = self.percentage

                if isinstance(layer, nn.Linear):
                    layer.forward = types.MethodType(linear_forward, layer)

                elif isinstance(layer, nn.Conv2d):
                    layer.forward = types.MethodType(conv_forward, layer)

    def handle_pruning(self, percentage):
        weight_masks = {}
        for name, layer in self.prune_model.named_modules():
            if name + ".weight" in self.prune_model.mask:
                weight_masks[name + ".weight"] = layer.gov.data
        all_scores = torch.cat([torch.flatten(x) for _, x in weight_masks.items()])

        if not self.local_pruning:
            num_params_to_keep = int(len(all_scores) * (1 - percentage))
            if num_params_to_keep < 1:
                num_params_to_keep += 1
            elif num_params_to_keep > len(all_scores):
                num_params_to_keep = len(all_scores)

            threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
            acceptable_score = threshold[-1]

        for name, weight_mask in weight_masks.items():
            if self.local_pruning:
                # don't prune more or less than possible
                num_params_to_keep = int(len(torch.flatten(weight_mask)) * (1 - percentage))
                if num_params_to_keep < 1:
                    num_params_to_keep += 1
                elif num_params_to_keep > len(torch.flatten(weight_mask)):
                    num_params_to_keep = len(torch.flatten(weight_mask))

                # threshold
                threshold, _ = torch.topk(torch.flatten(weight_mask), num_params_to_keep, sorted=True)
                acceptable_score = threshold[-1]

            self.model.mask[name] = (weight_mask > acceptable_score).__and__(
                self.model.mask[name].bool()).float().to(self.device)

            # how much we wanna prune
            length_nonzero = float(self.model.mask[name].flatten().shape[0])

            cutoff = (self.model.mask[name] == 0).sum().item()

            print("pruning", name, cutoff, "percentage", cutoff / length_nonzero, "length_nonzero", length_nonzero)

        self.model.apply_weight_mask()
        print("Final prune percentage ", self.model.pruned_percentage)

    def handle_global_pruning(self, percentage):
        # for global pruning
        weight_masks = {}
        for name, layer in self.prune_model.named_modules():
            if name + ".weight" in self.prune_model.mask:
                weight_masks[name + ".weight"] = layer.gov.data
        all_scores = torch.cat([torch.flatten(x) for _, x in weight_masks.items()])
        num_params_to_keep = int(len(all_scores) * (1 - percentage))
        threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
        acceptable_score = threshold[-1]
        for name, layer in self.prune_model.named_modules():
            if name + ".weight" in self.prune_model.mask:
                layer.prune_rate = (weight_masks[name + ".weight"] < acceptable_score).float().sum() / torch.numel(
                    layer.weight)

    def prune(self, percentage=0.0, **kwargs):
        self.foo()

        # get optimizer
        self.optimizer = find_right_model(
            OPTIMS, self.arguments['optimizer'],
            params=self.prune_model.parameters(),
            lr=self.arguments['learning_rate']
        )

        print("Training mask...")

        percentage = self.percentage
        self.prune_model.eval()

        acc = []
        for batch in self.val_loader:
            if not self.local_pruning:
                self.handle_global_pruning(percentage)
            x, y = batch
            x, y = x.to(self.device).float(), y.to(self.device)
            out = self.prune_model(x).squeeze()
            predictions = out.argmax(dim=-1, keepdim=True).view_as(y)
            correct = y.eq(predictions).sum().item()
            acc.append(correct / out.shape[0])
        print("Clean accuracy before training:", np.mean(acc))

        self.prune_model.train()
        best_model = None
        best_acc = 0

        best_count = 0
        early_stopping = 5

        for i in range(self.arguments["epochs"]):
            # Train
            for batch in self.train_loader:
                if not self.local_pruning:
                    self.handle_global_pruning(percentage)

                # train mask
                x, y = batch

                self.prune_model.train()

                x, y = x.to(self.device).float(), y.to(self.device)

                self.optimizer.zero_grad()
                out = self.prune_model(x)
                loss = self.loss.forward(out, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.prune_model.eval()
            acc = []

            # Validate on in-distribution data
            for batch in self.val_loader:
                if not self.local_pruning:
                    self.handle_global_pruning(percentage)
                x, y = batch
                x, y = x.to(self.device).float(), y.to(self.device)
                out = self.prune_model(x).squeeze()
                predictions = out.argmax(dim=-1, keepdim=True).view_as(y)
                correct = y.eq(predictions).sum().item()
                acc.append(correct / out.shape[0])
            print("Epoch", i, "clean accuracy:", np.mean(acc))

            if np.mean(acc) > best_acc:
                print('Best model')
                best_model = deepcopy(self.prune_model)
                best_acc = np.mean(acc)
                best_count = 0
                continue

            best_count += 1
            if best_count == early_stopping:
                break

            self.prune_model.train()

        self.prune_model = best_model
        self.handle_pruning(percentage)

        # Test on in-distribution data
        for batch in self.test_loader:
            x, y = batch
            x, y = x.to(self.device).float(), y.to(self.device)
            out = self.model(x).squeeze()
            predictions = out.argmax(dim=-1, keepdim=True).view_as(y)
            correct = y.eq(predictions).sum().item()
            acc.append(correct / out.shape[0])
        print("Final clean accuracy after pruning:", np.mean(acc))
