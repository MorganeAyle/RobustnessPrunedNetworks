
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


class HYDRAOOD(HYDRA):
    def __init__(self, arguments, local_pruning, *args, **kwargs):
        super(HYDRAOOD, self).__init__(arguments, local_pruning, *args, **kwargs)
        self.ood_loss = torch.nn.KLDivLoss(reduction='batchmean')

    def prune(self, percentage=0.0, **kwargs):

        self.foo()

        # get optimizer
        self.optimizer = find_right_model(
            OPTIMS, self.arguments['optimizer'],
            params=self.prune_model.parameters(),
            lr=self.arguments['learning_rate']
        )

        print("Training mask OOD...")

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
        best_auroc = 0
        best_model = None

        best_count = 0
        early_stopping = 5

        for i in range(self.arguments["epochs"]):
            for batch, ood_batch in zip(self.train_loader, self.ood_loader_train):
                if len(batch[0]) == len(ood_batch[0]):
                    if not self.local_pruning:
                        self.handle_global_pruning(percentage)

                    if i < int(self.arguments["epochs"] * 0.1):
                        # pre-train on normal loss
                        x, y = batch
                        x, y = x.to(self.device).float(), y.to(self.device)
                        out = self.prune_model(x)
                        loss = self.loss.forward(out, y)

                    else:
                        # OOD loss
                        data = torch.cat((batch[0], ood_batch[0]), 0)
                        target = batch[1]

                        data, target = data.to(self.device).float(), target.to(self.device)

                        # forward
                        out = self.prune_model(data)

                        loss = F.cross_entropy(out[:len(batch[0])], target)
                        # cross-entropy from softmax distribution to uniform distribution
                        loss += 0.5 * -(out[len(batch[0]):].mean(1) - torch.logsumexp(out[len(batch[0]):], dim=1)).mean()

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            self.prune_model.eval()
            acc = []
            ood_true = np.zeros(0)
            ood_preds = np.zeros(0)

            for batch in self.val_loader:
                if not self.local_pruning:
                    self.handle_global_pruning(percentage)
                x, y = batch
                x, y = x.to(self.device).float(), y.to(self.device)
                out = self.prune_model(x).squeeze()
                predictions = out.argmax(dim=-1, keepdim=True).view_as(y)
                correct = y.eq(predictions).sum().item()
                acc.append(correct / out.shape[0])

                probs = F.softmax(out, dim=-1)
                preds, _ = torch.max(probs, dim=-1)
                preds = preds.detach().cpu()
                ood_true = np.concatenate((ood_true, np.ones(len(preds))))
                ood_preds = np.concatenate((ood_preds, preds.reshape((-1))))

            for batch in self.ood_loader_val:
                if not self.local_pruning:
                    self.handle_global_pruning(percentage)
                x, y = batch
                x, y = x.to(self.device).float(), y.to(self.device)
                out = self.prune_model(x).squeeze()

                probs = F.softmax(out, dim=-1)
                preds, _ = torch.max(probs, dim=-1)
                preds = preds.detach().cpu()
                ood_true = np.concatenate((ood_true, np.zeros(len(preds))))
                ood_preds = np.concatenate((ood_preds, preds.reshape((-1))))

            print('Epoch', i)
            print('Accuracy', np.mean(acc))
            print('AUROC', calculate_auroc(ood_true, ood_preds))

            if calculate_auroc(ood_true, ood_preds) > best_auroc:
                print('Best model')
                best_model = deepcopy(self.prune_model)
                best_auroc = calculate_auroc(ood_true, ood_preds)

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
