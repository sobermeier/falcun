import pathlib

import click
from sklearn.model_selection import ParameterGrid

from src.deepal.database import MLFlowLogger
from src.deepal.settings import RUN_CONFIG
from src.deepal.utils import str_list, load_config, create_temporary_copy
from src.deepal.active_loop import active_loop

DATA_MAPPER = {
    "RepeatedMNIST": ("m", "LeNet"),
    "EMNIST": ("s", "LeNet"),
    "openml": ("l", "linear"),
    "bloodmnist": ("m_no", "resnet18"),
}


@click.command()
@click.option('--seeds', default=[0, 1, 2, 3, 4], type=str_list)  # 5 seeds
@click.option('--tracking_uri', required=True, help='MLflow tracking uri.')
@click.option('--datasets', default=["RepeatedMNIST", "EMNIST", "openml", "bloodmnist"], type=str_list, help='Dataset.')
@click.option('--strategy', default="Falcun")
def main(seeds, tracking_uri, datasets, strategy):
    # copy config directory to prevent unexpected overwriting during runs
    temp_dir = create_temporary_copy(path=RUN_CONFIG)
    if "openml" in datasets:
        ds_params_openml = load_config(filename=f"ds/openml", path=temp_dir.name)
        openml_ids = ds_params_openml.pop("ids", None)
        datasets = datasets + [f"openml-{_id}" for _id in openml_ids]
        datasets.remove("openml")
    for ds in datasets:
        # load config
        if "openml" in ds:
            ds_params = ds_params_openml
            al_setting, backbone = DATA_MAPPER["openml"]
        else:
            ds_params = load_config(filename=f"ds/{ds.lower()}", path=temp_dir.name)
            al_setting, backbone = DATA_MAPPER[ds]
        backbone_params = load_config(filename=f"backbone/{backbone.lower()}", path=temp_dir.name)
        m_params, train_params, m_architecture = backbone_params["model_params"], backbone_params["train_params"], \
                                                 backbone_params["model_architecture"]
        al_params = load_config(filename=f"al/{al_setting.lower()}", path=temp_dir.name)

        strat_params_grid = load_config(filename=f"strat/{strategy.lower()}", path=temp_dir.name)
        strat_params = list(ParameterGrid(strat_params_grid["grid"]))

        for strat_param_setting in strat_params:
            exp_name = f'Falcun-Ablation2'
            db = MLFlowLogger(experiment_name=exp_name, tracking_uri=tracking_uri)

            settings = {
                "strategy": strategy,
                "al": al_params,
                "dataset": ds,
                "backbone": m_architecture,
                "bb": m_params,
                "tr": train_params,
                "seed": seeds,  # save all seeds to mlflow for reproduction
                "ds": ds_params,
                "strat": strat_param_setting
            }

            #  One mlflow run over all seeds
            run_id, output_path = db.init_experiment(hyper_parameters=settings)
            output_path = pathlib.Path(output_path)
            active_loop(
                acq_strategy=strategy,
                model_architecture=m_architecture,
                ds=ds,
                al_params=al_params,
                ds_params=ds_params,
                m_params=m_params,
                train_params=train_params,
                database=db,
                seeds=seeds,
                output_path=output_path,
                acq_strategy_params=strat_param_setting
            )
    temp_dir.cleanup()


if __name__ == '__main__':
    main()
