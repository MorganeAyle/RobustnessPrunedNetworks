import sys
sys.path.append('.')

import os
from utils.constants import MODELS_DIR, DATA_MANAGER
import torch


def load_checkpoint(arguments, model, out):
    from utils.constants import RESULTS_DIR
    if (not (arguments['checkpoint_name'] is None)) and (not (arguments['checkpoint_model'] is None)):
        path = os.path.join(RESULTS_DIR, arguments['checkpoint_name'], MODELS_DIR, arguments['checkpoint_model'])
        state = DATA_MANAGER.load_python_obj(path)
        try:
            model.load_state_dict(state)
        except KeyError as e:
            print(list(state.keys()))
            raise e
        out(f"Loaded checkpoint {arguments['checkpoint_name']} from {arguments['checkpoint_model']}")


def log_start_run(arguments, out):
    arguments.PyTorch_version = torch.__version__
    arguments.PyThon_version = sys.version
    arguments.pwd = os.getcwd()
    out("PyTorch version:", torch.__version__, "Python version:", sys.version)
    out("Working directory: ", os.getcwd())
    out("CUDA avalability:", torch.cuda.is_available(), "CUDA version:", torch.version.cuda)
    out(arguments)

def assert_compatibilities(arguments):
    if arguments["prune_criterion"] in ["StructuredCroPRes", "StructuredCroPitRes"] and arguments["model"] not in [
        "ResNet18"]:
        raise ValueError(
            f"StructuredCroPRes and StructuredCroPitRes criterion only compatible with ResNet18 model, got {arguments['model']}")
    if arguments["prune_criterion"] in ["StructuredCroP", "StructuredCroPit"] and arguments["model"] in ["ResNet18"]:
        raise ValueError(
            f"ResNet18 model only compatible with StructuredCroPRes and StructuredCroPitRes criterion for structured pruning, got {arguments['prune_criterion']}")
