from sacred import Experiment
import seml

import sys

sys.path.append('.')

from common import load_checkpoint, log_start_run
from models import GeneralModel
from models.statistics.Metrics import Metrics
from utils.config_utils import *
from utils.model_utils import *
from utils.system_utils import *

import torch
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
    load_checkpoint(arguments, model, out)

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
        model=model
    )
    in_res, true_labels, all_preds, entropies = in_tester.evaluate()
    for key, value in in_res.items():
        results[key] = value

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
                std=std
            )

            out("Epsilon {}".format(str(epsilon)))
            res = tester.evaluate(epsilon=epsilon, true_labels=deepcopy(true_labels), all_preds=deepcopy(all_preds),
                                  entropies=deepcopy(entropies))

            for key, value in res.items():
                results[key] = value

    # OOD Evaluation
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
                ood_dataset=ood_data_set
            )
            res = tester.evaluate(true_labels=deepcopy(true_labels), all_preds=deepcopy(all_preds),
                                  entropies=deepcopy(entropies))

            for key, value in res.items():
                results[key] = value

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
    set_results_dir(arguments["results_dir"])
    metrics = Metrics()
    out = metrics.log_line
    print = out

    log_start_run(arguments, out)
    out("\n\n")
    metrics._batch_size = arguments['batch_size']
    return main(arguments, metrics)
