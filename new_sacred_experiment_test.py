import tensorflow as tf
import sacred
from pprint import pprint
from pathlib import Path
import json
from scinets.utils.experiment import SacredExperiment, NetworkExperiment


def load_json(path):
    with path.open() as f:
        return json.load(f)


if __name__ == "__main__":
    data_path = Path("experiment")
    name = load_json(data_path / "experiment_params.json")["name"]

    ex = sacred.Experiment(name=name)
    ex.observers.append(
        sacred.observers.MongoObserver.create(
            db_name="sacred",
            port=27017,
            url="yngvem.no",
            username="sacredWriter",
            password="LetUsUseSacredForLogging",
        )
    )

    @ex.config
    def cfg():
        experiment_params = load_json(data_path / "experiment_params.json")
        dataset_params = load_json(data_path / "dataset_params.json")
        model_params = load_json(data_path / "model_params.json")
        trainer_params = load_json(data_path / "trainer_params.json")
        log_params = load_json(data_path / "log_params.json")

    @ex.automain
    def main(
        _run,
        experiment_params,
        model_params,
        dataset_params,
        trainer_params,
        log_params,
    ):
        experiment = SacredExperiment(
            _run=_run,
            experiment_params=experiment_params,
            model_params=model_params,
            dataset_params=dataset_params,
            trainer_params=trainer_params,
            log_params=log_params,
        )
        experiment.train(60)
        print(experiment.find_best_model("val", "dice"))
