from sacred import Experiment
import seml

import sys
import pickle

sys.path.append('.')

from experiments.common import load_checkpoint, log_start_run
from models import GeneralModel
from models.statistics.Metrics import Metrics
from utils.config_utils import *
from utils.model_utils import *
from utils.system_utils import *
from utils.constants import set_results_dir, TESTERS_DIR, NETWORKS_DIR, DATASET_PATH
from utils.data_loaders import CIFAR10C
from models.testers.AdversarialEvaluation import construct_adversarial_batch
from torchvision import transforms
from utils.metrics import calculate_auroc
from torch.autograd import Variable

import torch
import torch.nn.functional as F
import numpy as np
from copy import deepcopy


ex = Experiment()
seml.setup_logger(ex)


def iterate_data_gradnorm(data_loader, model, temperature, num_classes, true_labels, pred_scores, true_label, attack=None, device=None, epsilon=None, transform=None):
    logsoftmax = torch.nn.LogSoftmax(dim=-1).cuda()
    for b, (x, y) in enumerate(data_loader):
        if b % 100 == 0:
            print('{} batches processed'.format(b))
        if attack is not None:
            x = construct_adversarial_batch(x, y, attack, model, device, epsilon, False, False, transform)
        inputs = Variable(x.cuda(), requires_grad=True)

        model.zero_grad()
        outputs = model(inputs)
        targets = torch.ones((inputs.shape[0], num_classes)).cuda()
        outputs = outputs / temperature
        loss = torch.mean(torch.sum(-targets * logsoftmax(outputs), dim=-1))

        loss.backward()

        layer_grad = list(model.modules())[-1].weight.grad.data

        layer_grad_norm = torch.sum(torch.abs(layer_grad)).cpu().numpy()
        pred_scores.append(layer_grad_norm)
        true_labels.append(true_label)

    return true_labels, pred_scores


def main(
        arguments,
        metrics: Metrics
):

    global out
    out = metrics.log_line
    out(f"starting at {get_date_stamp()}")

    # hardware
    device = configure_device(arguments)

    if arguments['disable_cuda_benchmark']:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # get model
    model: GeneralModel = find_right_model(
        NETWORKS_DIR, arguments['model'],
        device=device,
        hidden_dim=arguments['hidden_dim'],
        input_dim=arguments['input_dim'],
        output_dim=arguments['output_dim'],
    ).to(device)

    # load pre-trained weights if specified
    path = os.path.join(arguments['results_dir'], arguments['checkpoint_name'], MODELS_DIR, arguments['checkpoint_model'] + '.pickle')
    with (open(path, "rb")) as openfile:
        state = pickle.load(openfile)
    model.load_state_dict(state)

    model.eval()
    results = {}

    # loss_fn = torch.nn.KLDivLoss(reduction="batchmean")
    # loss_fn = torch.nn.CrossEntropyLoss()

    # load data
    _, test_loader = find_right_model(
        DATASETS, arguments['data_set'],
        arguments=arguments,
        mean=arguments['mean'],
        std=arguments['std']
    )

    in_true_labels = []
    in_pred_scores = []
    # In-distribution evaluation
    in_true_labels, in_pred_scores = iterate_data_gradnorm(test_loader, model, 1, 10, in_true_labels, in_pred_scores, 1)

    # for batch in test_loader:
    #     x, _ = batch
    #     x = x.to(device)
    #
    #     model.zero_grad()
    #     output = model(x)
    #     loss = -(output.mean(1) - torch.logsumexp(output, dim=1)).mean()
    #     # preds = F.log_softmax(output, dim=-1)
    #     # loss = loss_fn(preds, torch.ones_like(preds) / preds.shape[-1])
    #     # loss = loss_fn(output, torch.full_like(output, 0.1).to(torch.float64))
    #     loss.backward()
    #     scores = [list(model.parameters())[-1].grad.norm().cpu().detach().numpy()]
    #     # scores = [p.grad.norm().cpu().detach().numpy() for p in model.parameters()]
    #     in_true_labels.append(0)
    #     in_pred_scores.append(sum(scores))

    if 'AdversarialEvaluation' in arguments['test_scheme']:
        # Adversarial evaluation
        for attack in arguments['eval_attacks']:
            for epsilon in arguments['eval_epsilons']:
                out("Attack {}".format(attack))

                # load data
                (_, un_test_loader), mean, std = find_right_model(
                    DATASETS, arguments['data_set'] + '_unnormalized',
                    arguments=arguments,
                    mean=arguments['mean'],
                    std=arguments['std']
                )
                transform = transforms.Normalize(mean, std)

                true_labels, pred_scores = iterate_data_gradnorm(un_test_loader, model, 1, 10, deepcopy(in_true_labels),
                                                                 deepcopy(in_pred_scores), 0,
                                                                 attack, device, epsilon, transform)

                name = 'AUROC_' + attack + '_' + str(epsilon)
                results[name] = calculate_auroc(np.array(true_labels), np.array(pred_scores))

    print(results)

    # OOD Evaluation
    if 'OODEvaluation' in arguments['test_scheme']:
        for ood_data_set in arguments['eval_ood_data_sets']:
            out("OOD Dataset: {}".format(ood_data_set))

            # load OOD data
            _, ood_loader = find_right_model(
                DATASETS, ood_data_set,
                arguments=arguments,
                mean=arguments['mean'],
                std=arguments['std']
            )

            true_labels, pred_scores = iterate_data_gradnorm(ood_loader, model, 1, 10, deepcopy(in_true_labels),
                                                             deepcopy(in_pred_scores), 0)

            name = 'AUROC_' + ood_data_set
            results[name] = calculate_auroc(np.array(true_labels), np.array(pred_scores))

    print(results)

    if 'DSEvaluation' in arguments['test_scheme']:
        # DS Evaluation
        if "CIFAR10" in arguments["data_set"]:
            ds_path = os.path.join(DATASET_PATH, "cifar10_corrupted")
            aurocs = []

            for ds_dataset_name in os.listdir(ds_path):
                if ds_dataset_name.endswith('5.npz'):
                    # Get corruption loader
                    npz_dataset = np.load(os.path.join(ds_path, ds_dataset_name))
                    ds_dataset = CIFAR10C(npz_dataset["images"], npz_dataset["labels"], arguments["mean"],
                                          arguments["std"])
                    ds_loader = torch.utils.data.DataLoader(
                        ds_dataset,
                        batch_size=arguments['batch_size'],
                        shuffle=False,
                        pin_memory=True,
                        num_workers=1
                    )

                    true_labels, pred_scores = iterate_data_gradnorm(ds_loader, model, 1, 10, deepcopy(in_true_labels),
                                                                     deepcopy(in_pred_scores), 0)

                    name = 'AUROC_' + ds_dataset_name
                    auroc = calculate_auroc(np.array(true_labels), np.array(pred_scores))
                    results[name] = auroc
                    aurocs.append(auroc)
            name = 'avg_AUROC'
            results[name] = np.mean(aurocs)

    return results


@ex.post_run_hook
def collect_stats(_run):
    seml.collect_exp_stats(_run)


@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(seml.create_mongodb_observer(db_collection, overwrite=overwrite))


@ex.automain
def run(arguments):
    metrics = Metrics()
    out = metrics.log_line
    print = out

    log_start_run(arguments, out)
    out("\n\n")
    metrics._batch_size = arguments['batch_size']
    return main(arguments, metrics)
