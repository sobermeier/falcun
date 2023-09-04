import pathlib
import time
from typing import Dict, Any, List

import numpy as np
import pandas as pd

from src.deepal.dataset import get_handler, get_dataset, get_transform
from src.deepal.models import get_net
from src.deepal.query_strategies import get_strategy
from src.deepal.utils import set_seed
from src.deepal.database import MLFlowLogger


def active_loop(
        acq_strategy: str,
        model_architecture: str,
        ds: str,
        al_params: Dict[str, Any],
        m_params: Dict[str, Any],
        ds_params: Dict[str, Any],
        train_params: Dict[str, Any],
        database: MLFlowLogger,
        seeds: List,
        output_path: pathlib.Path,
        acq_strategy_params: Dict[str, Any]
):
    n_initial, n_query, budget = al_params["n_initials"], al_params["n_queries"], al_params["budget"]
    # load dataset
    ds_args = {}
    if "ds_args" in ds_params.keys():
        ds_args=ds_params["ds_args"]
    X_tr, Y_tr, X_te, Y_te, class_counts = get_dataset(ds, ds_args)
    n_pool = len(Y_tr)
    ds_params['transform'] = get_transform(ds)

    m_params["output_dim"] = int(max(Y_tr) + 1)
    if model_architecture in ["linear", "conv"]:
        m_params["input_dim"] = np.shape(X_tr)[1:]
    else:
        m_params["input_dim"] = ds_params["channels"]

    # evaluation loop
    n_rounds = int((budget - n_initial) / n_query)
    if (budget - n_initial) % n_query != 0:
        print(f"--> budget - initials) % queries = {(budget - n_initial) % n_query}")
        n_rounds += 1

    accuracies_per_seed = {iteration: [] for iteration in range(0, n_rounds + 1)}
    times_per_seed = {iteration: [] for iteration in range(0, n_rounds)}
    acq_times_per_seed = {iteration: [] for iteration in range(0, n_rounds)}
    train_times_per_seed = {iteration: [] for iteration in range(0, n_rounds)}
    for seed in seeds:
        set_seed(seed)
        remaining_budget = budget - n_initial
        print(f"-> Eval with seed={seed}")
        # generate initial labeled pool; dependent on seed
        idxs_lb = np.zeros(n_pool, dtype=bool)
        idxs_tmp = np.arange(n_pool)
        np.random.shuffle(idxs_tmp)
        idxs_lb[idxs_tmp[:n_initial]] = True
        # load network and data handler
        net_sampling = get_net(model_architecture=model_architecture)
        handler = get_handler(ds)

        # strategy requires data, currently labeled indices, the model, the data handler, and custom args
        strategy = get_strategy(
            name=acq_strategy, x=X_tr, y=Y_tr, labelled=idxs_lb,
            model={
                "net": net_sampling,
                "net_args": m_params,
                "weight_reset_type": al_params["weight_reset_type"]
            },
            d_handler=handler,
            d_args={"ds_params": ds_params},
            strat_args=acq_strategy_params
        )
        database.log_params({"ds.tr_size": n_pool})
        strategy.set_path(output_path / str(seed))
        strategy.set_total_rounds(n_rounds)
        # save initially labeled instances
        strategy.save_stats(pd.DataFrame([list(idxs_tmp[:n_initial])], index=["img_id"]).T)

        # round 0 accuracy
        tr_acc, tr_loss = strategy.train(**train_params)  # train sampling model for acquisition
        P = strategy.predict(X_te, Y_te)

        _accuracy = 1.0 * (Y_te == P).sum().item() / len(Y_te)
        accuracies_per_seed[0].append((_accuracy, np.sum(idxs_lb)))
        database.log_results(
            result={
                f"acc_{seed}": _accuracy,
                f"tr_loss_{seed}": tr_loss,
                f"tr_acc_{seed}": tr_acc
            }, step=np.sum(idxs_lb)
        )
        print('Round 0\ntesting accuracy {}'.format(_accuracy))

        rd = 1
        _query_size = n_query
        while remaining_budget > 0:
            if remaining_budget < _query_size:
                _query_size = remaining_budget

            print(f'Round {rd} - Budget {remaining_budget}')

            # query
            strategy.set_current_round(rd)
            _acq_start = time.time()
            q_idxs, _acq_end = strategy.query(_query_size)
            _acq_seconds_total = _acq_end - _acq_start
            acq_times_per_seed[rd-1].append((_acq_seconds_total, np.sum(idxs_lb)))
            idxs_lb[q_idxs] = True

            # update
            strategy.update(idxs_lb, q_idxs)
            # retrain
            _train_start = time.time()
            tr_acc, tr_loss = strategy.train(**train_params)
            _train_end = time.time()
            _train_seconds_total = _train_end - _train_start
            train_times_per_seed[rd-1].append((_train_seconds_total, np.sum(idxs_lb)))

            _sec_total = _train_seconds_total + _acq_seconds_total
            times_per_seed[rd-1].append((_sec_total, np.sum(idxs_lb)))

            P = strategy.predict(X_te, Y_te)
            _accuracy = 1.0 * (Y_te == P).sum().item() / len(Y_te)
            accuracies_per_seed[rd].append((_accuracy, np.sum(idxs_lb)))
            result = {
                f"acc_{seed}": _accuracy,
                f"tr_loss_{seed}": tr_loss,
                f"tr_acc_{seed}": tr_acc,
                f"query_time_{seed}": _acq_seconds_total,
                f"train_time_{seed}": _train_seconds_total,
                f"total_time_{seed}": _sec_total,
            }
            database.log_results(result=result, step=np.sum(idxs_lb))
            print('testing accuracy {}'.format(_accuracy))

            remaining_budget -= _query_size
            rd += 1

    print('--- accuracies per round & seed: --- ')
    print(f"  {accuracies_per_seed}")
    # save average over seeds to mlflow

    database.log_metric_statistics(name="acc", values=accuracies_per_seed)
    database.log_metric_statistics(name="total_time", values=times_per_seed)
    database.log_metric_statistics(name="query_time", values=acq_times_per_seed)
    database.log_metric_statistics(name="train_time", values=train_times_per_seed)
    # close experiment after all seeds have been visited
    database.finalise_experiment()

