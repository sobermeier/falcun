import logging
import pathlib
from typing import Dict, Any, Tuple, Sequence
import statistics

import mlflow
from mlflow.entities import ViewType
import pandas as pd

from src.deepal.settings import DATA_ROOT
from src.deepal.utils import flatten_dict


class MLFlowLogger:
    def __init__(
        self,
        tracking_uri: str,
        root: str = DATA_ROOT + '/experiments',
        experiment_name: str = 'IDEAL'
    ):
        """
        Constructor.

        Connects to a running MLFlow instance in which the runs of the experiments are stored.
        Also creates an output root directory. The directory will be created if it does not exist.

        :param tracking_uri: str
            The uri where the MLFlow instance is running.
        :param root: str
            The path of the output root.
        :param experiment_name: str
            The name of the experiment on the MLFlow instance server.
        """
        mlflow.set_tracking_uri(uri=tracking_uri)
        experiment = mlflow.get_experiment_by_name(name=experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(name=experiment_name, artifact_location=root)
        else:
            experiment_id = experiment.experiment_id
        self.experiment_id = experiment_id
        self.root = pathlib.Path(root)
        self.current_run = None

    def init_experiment(
        self,
        hyper_parameters: Dict[str, Any]
    ) -> Tuple[str, pathlib.Path]:
        """
        Initialise an experiment, i.e. a run in the specified experiment.
        Creates an entry describing the run's hyper-parameters, and associates an unique output directory.

        :param hyper_parameters: Dict
            The hyperparameters. Should be serialisable to JSON/BSON.
        :return: A tuple (id, path) where
            id:
                a unique ID of the experiment (equal to the ID in the MLFlow instance)
            path:
                the path to an existing directory for outputs of the experiment.
        """
        self.current_run = mlflow.start_run(experiment_id=self.experiment_id)

        # create output directory
        output_path = self.root / str(self.experiment_id) / str(self.current_run.info.run_id)
        if output_path.is_dir():
            logging.error('Output path already exists! {p}'.format(p=output_path))
        output_path.mkdir(exist_ok=True, parents=True)
        hyper_parameters["out_dir"] = output_path

        self.log_params(hyper_parameters=flatten_dict(hyper_parameters))
        # return id as experiment handle, and path of existing directory to store outputs to
        return self.current_run.info.run_id, output_path

    @staticmethod
    def log_params(hyper_parameters: Dict[str, Any]):
        mlflow.log_params(params=flatten_dict(hyper_parameters))

    @staticmethod
    def log_artifacts(file_path):
        mlflow.log_artifacts(file_path)

    @staticmethod
    def finalise_experiment(status="FINISHED") -> None:
        """
        Close the current run.
        :return: None.
        """
        mlflow.end_run(status=status)

    @staticmethod
    def log_results(result: Dict[str, Any], step=None):
        """
        :param result: Dict
            A flattened dictionary holding high-level results, e.g. a few numbers.
        :param step: int
        :return: None.
        """
        mlflow.log_metrics(metrics=flatten_dict(result), step=step)

    def log_metric_statistics(self, name, values):
        """
        Expected format: dictionary with key per active learning round and value list of accuracies per seed
        """
        for key, val in values.items():
            # key = iteration, val = [(accuracy, budget), ...]
            vals = [tup[0] for tup in val]
            steps = [tup[1] for tup in val]
            self.log_results(result={f"avg_{name}": statistics.mean(vals)}, step=steps[0])
            self.log_results(result={f"var_{name}": statistics.variance(vals)}, step=steps[0])
            self.log_results(result={f"std_{name}": statistics.stdev(vals)}, step=steps[0])

    def get_artifacts_path(self):
        return self.root / str(self.experiment_id) / str(self.current_run.info.run_id) / "artifacts"


def export_from_mlflow(mlflow_uri: str,
                       mlflow_experiment_name: str,
                       metrics: Tuple[str, ...],
                       ) -> pd.DataFrame:
    # Connect to MLflow
    mlflow.set_tracking_uri(mlflow_uri)
    client = mlflow.tracking.MlflowClient()

    # Get experiment by ID
    experiment = client.get_experiment_by_name(name=mlflow_experiment_name)
    experiment_id = experiment.experiment_id

    # Load parameters and metrics
    results_df = []
    for run in client.search_runs(experiment_ids=[experiment_id],
                                  max_results=50000,
                                  ):
        run_id = run.info.run_id

        data = run.data.params
        data.update({key: run.data.metrics[key] for key in run.data.metrics.keys() if key in metrics})
        data.update({"run_id": run_id})
        data.update({"run_start_time": run.info.start_time})
        data.update({"experiment_name": mlflow_experiment_name})
        run_df = pd.DataFrame(data=data, index=[run_id])

        results_df += [run_df]

    results_df = pd.concat(results_df, sort=True)
    return results_df
