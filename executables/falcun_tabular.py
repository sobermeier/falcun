"""
[46, 28, 40983, 40975, 50, 40966, 1462, 40979]  -> small (20,20,200)
[1590, 300, 182, 1461, 6, 155, 156, 184] -> large (500,500,5000) (-> BADGE setting)
"""

import pathlib

import click

from src.deepal.database import MLFlowLogger
from src.deepal.settings import RUN_CONFIG
from src.deepal.utils import str_list, load_config, create_temporary_copy
from src.deepal.active_loop import active_loop


@click.command()
@click.option('--seeds', default=[0, 1, 2, 3, 4], type=str_list)  # 5 seeds
@click.option('--tracking_uri', required=True, help='MLflow tracking uri.')
@click.option('--dataset', default="openml", type=str, help='Dataset.')
@click.option('--backbone', default="linear", required=False, help='Backbone.')
@click.option('--al', default="l", help='AL setting.')
def main(seeds, tracking_uri, dataset, backbone, al):
    # copy config directory to prevent unexpected overwriting during runs
    temp_dir = create_temporary_copy(path=RUN_CONFIG)

    # load config
    ds_params = load_config(filename=f"ds/{dataset.lower()}", path=temp_dir.name)
    openml_ids = ds_params.pop("ids", None)
    backbone_params = load_config(filename=f"backbone/{backbone.lower()}", path=temp_dir.name)
    m_params, train_params, m_architecture = backbone_params["model_params"], backbone_params["train_params"], \
        backbone_params["model_architecture"]
    al_params = load_config(filename=f"al/{al.lower()}", path=temp_dir.name)

    acq_strategies = [
        "Falcun",
        "RandomSampling",
        "AlphaMixSampling",
        "CdalCS",
        "BadgeSampling",
        "KCenterGreedy2",
        "CLUE",
        "EntropySampling",
    ]
    for oid in openml_ids:
        ds_id = f"{dataset}-{oid}"
        exp_name = f'Falcun-{dataset}-{backbone}'
        for acq_strategy in acq_strategies:
            strat_params = load_config(filename=f"strat/{acq_strategy.lower()}", path=temp_dir.name)["default"]
            db = MLFlowLogger(experiment_name=exp_name, tracking_uri=tracking_uri)

            settings = {
                "strategy": acq_strategy,
                "al": al_params,
                "dataset": ds_id,
                "backbone": m_architecture,
                "bb": m_params,
                "tr": train_params,
                "seed": seeds,  # save all seeds to mlflow for reproduction
                "ds": ds_params,
                "strat": strat_params
            }

            #  One mlflow run over all seeds
            run_id, output_path = db.init_experiment(hyper_parameters=settings)
            output_path = pathlib.Path(output_path)
            try:
                active_loop(
                    acq_strategy=acq_strategy,
                    model_architecture=m_architecture,
                    ds=ds_id,
                    al_params=al_params.copy(),
                    ds_params=ds_params.copy(),
                    m_params=m_params.copy(),
                    train_params=train_params.copy(),
                    database=db,
                    seeds=seeds,
                    output_path=output_path,
                    acq_strategy_params=strat_params
                )
            except Exception as e:
                print(e)
                db.finalise_experiment(status="FAILED")
    temp_dir.cleanup()


if __name__ == '__main__':
    main()
