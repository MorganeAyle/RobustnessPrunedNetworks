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
from models.criterions.HYDRA import HYDRA


class HYDRADS(HYDRA):
    def __init__(self, arguments, local_pruning, *args, **kwargs):
        super(HYDRADS, self).__init__(arguments, local_pruning, *args, **kwargs)

    def prune(self, percentage=0.0, **kwargs):

        self.foo()

        # get optimizer
        self.optimizer = find_right_model(
            OPTIMS, self.arguments['optimizer'],
            params=self.prune_model.parameters(),
            lr=self.arguments['learning_rate']
        )

        print("Training mask DS...")

        percentage = self.percentage
        self.prune_model.eval()

        acc = []
        for batch in self.test_loader:
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
            for batch in self.train_loader:
                if not self.local_pruning:
                    self.handle_global_pruning(percentage)

                self.prune_model.train()

                x, y = batch
                x += (torch.rand(x.shape) > 0.5).float() * torch.normal(torch.zeros(x.shape), torch.ones(x.shape))

                x, y = x.to(self.device).float(), y.to(self.device)

                self.optimizer.zero_grad()
                out = self.prune_model(x)
                loss = self.loss.forward(out, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.prune_model.eval()
            acc = []
            ds_acc = []

            for batch in self.val_loader:
                if not self.local_pruning:
                    self.handle_global_pruning(percentage)
                x, y = batch
                x, y = x.to(self.device).float(), y.to(self.device)
                out = self.prune_model(x).squeeze()
                predictions = out.argmax(dim=-1, keepdim=True).view_as(y)
                correct = y.eq(predictions).sum().item()
                acc.append(correct / out.shape[0])

            for batch in self.val_loader:
                if not self.local_pruning:
                    self.handle_global_pruning(percentage)
                x, y = batch
                x += (torch.rand(x.shape) > 0.5).float() * torch.normal(torch.zeros(x.shape), torch.ones(x.shape))
                x, y = x.to(self.device).float(), y.to(self.device)
                out = self.prune_model(x).squeeze()
                predictions = out.argmax(dim=-1, keepdim=True).view_as(y)
                correct = y.eq(predictions).sum().item()
                ds_acc.append(correct / out.shape[0])

            print('Epoch', i)
            print('Accuracy', np.mean(acc))
            print('DS Accuracy', np.mean(ds_acc))

            if np.mean(ds_acc) > best_acc:
                print('Best model')
                best_model = deepcopy(self.prune_model)
                best_acc = np.mean(ds_acc)

                best_count = 0
                continue

            best_count += 1
            if best_count == early_stopping:
                break

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
