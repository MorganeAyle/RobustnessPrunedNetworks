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

import torch
import numpy as np
from copy import deepcopy


ex = Experiment()
seml.setup_logger(ex)


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

    # load data
    _, test_loader = find_right_model(
        DATASETS, arguments['data_set'],
        arguments=arguments,
        mean=arguments['mean'],
        std=arguments['std']
    )

    # In-distribution evaluation
    in_tester = find_right_model(
        TESTERS_DIR, 'InEvaluation',
        test_loader=test_loader,
        device=device,
        model=model,
        batch_results=arguments['batch_results'],
        group_batch_size=arguments['group_batch_size']
    )
    in_res, true_labels, all_preds, entropies = in_tester.evaluate()
    for key, value in in_res.items():
        results[key] = value

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
                # build tester
                tester = find_right_model(
                    TESTERS_DIR, 'AdversarialEvaluation',
                    attack=attack,
                    model=model,
                    device=device,
                    test_loader=un_test_loader,
                    mean=mean,
                    std=std,
                    batch_results=arguments['batch_results'],
                    group_batch_size=arguments['group_batch_size']
                )

                out("Epsilon {}".format(str(epsilon)))
                res = tester.evaluate(epsilon=epsilon, true_labels=deepcopy(true_labels), all_preds=deepcopy(all_preds),
                                      entropies=deepcopy(entropies))

                for key, value in res.items():
                    results[key] = value

    # OOD Evaluation
    if 'OODEvaluation' in arguments['test_scheme']:
        with torch.no_grad():
            for ood_data_set in arguments['eval_ood_data_sets']:
                out("OOD Dataset: {}".format(ood_data_set))

                # load OOD data
                _, ood_loader = find_right_model(
                    DATASETS, ood_data_set,
                    arguments=arguments,
                    mean=arguments['mean'],
                    std=arguments['std']
                )
                # build tester
                tester = find_right_model(
                    TESTERS_DIR, 'OODEvaluation',
                    model=model,
                    device=device,
                    ood_loader=ood_loader,
                    ood_dataset=ood_data_set,
                    batch_results=arguments['batch_results'],
                    group_batch_size=arguments['group_batch_size']
                )
                res = tester.evaluate(true_labels=deepcopy(true_labels), all_preds=deepcopy(all_preds),
                                      entropies=deepcopy(entropies))

                for key, value in res.items():
                    results[key] = value

    if 'DSEvaluation' in arguments['test_scheme']:
        # DS Evaluation
        with torch.no_grad():
            if "CIFAR10" in arguments["data_set"]:
                avg_acc = [[] for _ in range(5)]
                avg_entropy = [[] for _ in range(5)]
                avg_auroc = [[] for _ in range(5)]
                avg_aupr = [[] for _ in range(5)]
                avg_auroc_ent = [[] for _ in range(5)]
                avg_aupr_ent = [[] for _ in range(5)]

                ds_path = os.path.join(DATASET_PATH, "cifar10_corrupted")

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
                            num_workers=4
                        )

                        # build tester
                        tester = find_right_model(
                            TESTERS_DIR, 'DSEvaluation',
                            model=model,
                            device=device,
                            ds_loader=ds_loader,
                            ds_dataset=ds_dataset_name.split('.')[0],
                            batch_results=arguments['batch_results'],
                            group_batch_size=arguments['group_batch_size']
                        )
                        res = tester.evaluate(true_labels=deepcopy(true_labels), all_preds=deepcopy(all_preds),
                                              entropies=deepcopy(entropies))

                        severity = int(ds_dataset_name.split('.')[0].split('_')[-1]) - 1
                        for key, value in res.items():
                            if key.startswith('acc'):
                                avg_acc[severity].append(value)
                            elif key.startswith('auroc_entropy'):
                                avg_auroc_ent[severity].append(value)
                            elif key.startswith('aupr_entropy'):
                                avg_aupr_ent[severity].append(value)
                            elif key.startswith('auroc'):
                                avg_auroc[severity].append(value)
                            elif key.startswith('aupr'):
                                avg_aupr[severity].append(value)
                            elif key.startswith('entropy_'):
                                avg_entropy[severity].append(value)

                            results[key] = value

                avg_acc = [np.mean(acc) for acc in avg_acc]
                avg_auroc_ent = [np.mean(auroc_ent) for auroc_ent in avg_auroc_ent]
                avg_aupr_ent = [np.mean(aupr_ent) for aupr_ent in avg_aupr_ent]
                avg_auroc = [np.mean(auroc) for auroc in avg_auroc]
                avg_aupr = [np.mean(aupr) for aupr in avg_aupr]
                avg_entropy = [np.mean(entropy) for entropy in avg_entropy]

                for i in range(len(avg_acc)):
                    name = 'avg_acc_' + str(i + 1)
                    results[name] = avg_acc[i]
                for i in range(len(avg_acc)):
                    name = 'avg_auroc_ent_' + str(i + 1)
                    results[name] = avg_auroc_ent[i]
                for i in range(len(avg_acc)):
                    name = 'avg_aupr_ent_' + str(i + 1)
                    results[name] = avg_aupr_ent[i]
                for i in range(len(avg_acc)):
                    name = 'avg_auroc_' + str(i + 1)
                    results[name] = avg_auroc[i]
                for i in range(len(avg_acc)):
                    name = 'avg_aupr_' + str(i + 1)
                    results[name] = avg_aupr[i]
                for i in range(len(avg_acc)):
                    name = 'avg_entropy_' + str(i + 1)
                    results[name] = avg_entropy[i]

                results['avg_acc_cifar10c'] = np.mean(avg_acc)
                results['avg_auroc_ent_cifar10c'] = np.mean(avg_auroc_ent)
                results['avg_aupr_ent_cifar10c'] = np.mean(avg_aupr_ent)
                results['avg_auroc_cifar10c'] = np.mean(avg_auroc)
                results['avg_aupr_cifar10c'] = np.mean(avg_aupr)
                results['avg_entropy_cifar10c'] = np.mean(avg_entropy)

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
    # set_results_dir(arguments["results_dir"])
    metrics = Metrics()
    out = metrics.log_line
    print = out

    if arguments['batch_results']:
        assert arguments['batch_size'] % arguments['group_batch_size'] == 0

    log_start_run(arguments, out)
    out("\n\n")
    metrics._batch_size = arguments['batch_size']
    return main(arguments, metrics)
