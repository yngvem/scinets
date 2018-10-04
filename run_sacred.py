# Builtin imports
from pprint import pprint
import sys
from pathlib import Path
import argparse

# External imports
import json
import yaml
import sacred


def _load_file_using_module(path, module):
    if isinstance(path, str):
        path = Path(path)
    with path.open() as f:
        return module.load(f)


def load_json(path):
    return _load_file_using_module(path, json)


def load_yaml(path):
    return _load_file_using_module(path, yaml)


def parse_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=SmartFormatter
    )
    parser.add_argument(
        "database_credentials",
        help="Path YAML file with database credentials.",
        type=str
    )
    parser.add_argument(
        "experiment",
        help="R|Path to folder with the JSON files specifying the parameters \n"
             "used in the experiment. The folder should contain the \n"
             "following files\n"
             "   - 'experiment_params.json'\n"
             "   - 'dataset_params.json'\n"
             "   - 'model_params.json'\n"
             "   - 'trainer_params.json'\n"
             "   - 'log_params.json'",
        type=str
    )
    parser.add_argument(
        "num_steps",
        help="Number of training steps to perform.",
        type=int
    )
    parser.add_argument(
        "--name",
        help="The name of the experiment used for logging.",
        type=str
    )

    args = parser.parse_args()
    db_credentials = load_yaml(args.database_credentials)
    return db_credentials, Path(args.experiment), args.num_steps, args.name


class SmartFormatter(argparse.HelpFormatter):
    def _split_lines(self, text, width):
        if text.startswith('R|'):
            return text[2:].splitlines()
        return super()._split_lines(text, width)


def create_experiment(name, db_params):
    ex = sacred.Experiment(name=name)
    ex.observers.append(sacred.observers.MongoObserver.create(**db_params))
    return ex


if __name__ == '__main__':
    db_params, data_path, num_steps, name = parse_arguments()
    if name is None:
        name = load_json(data_path/'experiment_params.json')['name']

    ex = create_experiment(name, db_params)

    @ex.config
    def cfg():
        dataset_params = load_json(data_path/'dataset_params.json')
        model_params = load_json(data_path/'model_params.json')
        trainer_params = load_json(data_path/'trainer_params.json')
        log_params = load_json(data_path/'log_params.json')
        experiment_params = load_json(data_path/'experiment_params.json')
        experiment_params['name'] = name

    @ex.main
    def sacred_main(_run, experiment_params, model_params, dataset_params,
             trainer_params, log_params):
        from scinets.utils.experiment import SacredExperiment
        experiment = SacredExperiment(
            _run=_run,
            experiment_params=experiment_params,
            model_params=model_params,
            dataset_params=dataset_params,
            trainer_params=trainer_params,
            log_params=log_params,
        )
        experiment.train(num_steps)
    ex.run()
