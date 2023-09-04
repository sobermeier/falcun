import argparse
import ast
import json
import tempfile, shutil, os

import numpy as np
import torch

from src.deepal.settings import RUN_CONFIG


def create_temporary_copy(path) -> tempfile.TemporaryDirectory:
    f = tempfile.TemporaryDirectory(dir=".")
    shutil.copytree(path, f.name, dirs_exist_ok=True)
    return f


def load_config(filename, path=RUN_CONFIG) -> dict:
    config_path = os.path.join(path, f'{filename}.json')
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    return config


def flatten_dict(d):
    """
    Function to transform a nested dictionary to a flattened dot notation dictionary.

    :param d: Dict
        The dictionary to flatten.

    :return: Dict
        The flattened dictionary.
    """

    def expand(key, value):
        if isinstance(value, dict):
            return [(key + '.' + k, v) for k, v in flatten_dict(value).items()]
        else:
            return [(key, value)]

    items = [item for k, v in d.items() for item in expand(k, v)]

    return dict(items)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.enabled = False


def str_bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def str_list(v):
    if isinstance(v, list):
        return v
    return ast.literal_eval(v)


def calc_unc(
    probs: torch.Tensor,
    unc: str = "entropy",
    normalized: bool = False,
    epsilon: float = 1e-8,
    inverse: bool = False
) -> torch.Tensor:
    """
    standard uncertainty functions -> larger values denote higher uncertainty
    """
    if unc == "entropy":
        probs += epsilon
        if inverse:
            _uncertainty = (probs * torch.log(probs)).sum(1)
            _uncertainty = _uncertainty + _uncertainty.min() * -1
        else:
            _uncertainty = - (probs * torch.log(probs)).sum(1)
    elif unc == "margin":
        probs_sorted, idxs = probs.sort(descending=True)
        if inverse:
            _uncertainty = (probs_sorted[:, 0] - probs_sorted[:, 1])
        else:
            _uncertainty = 1 - (probs_sorted[:, 0] - probs_sorted[:, 1])
        # sorted_margin, sm_idx = U.sort()
    elif unc == "lc":
        probs_sorted, idxs = probs.sort(descending=True)
        if inverse:
            _uncertainty = probs_sorted[:, 0]
        else:
            _uncertainty = 1 - probs_sorted[:, 0]
    elif unc == "anticonf":
        # --> is actually just 2 * lc
        p_conf = torch.zeros(probs.shape[-1])
        p_conf[0] = 1
        probs_sorted, idxs = probs.sort(descending=True)
        _uncertainty = torch.as_tensor(np.linalg.norm(p_conf - probs_sorted))

        if inverse:
            _uncertainty = -1 * _uncertainty
    else:
        raise NotImplementedError
    if normalized:
        _uncertainty = (_uncertainty - _uncertainty.min()) / _uncertainty.max()

    return _uncertainty
