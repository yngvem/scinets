__author__ = "Yngve Mardal Moe"
__email__ = "yngve.m.moe@gmail.com"


from pathlib import Path
import json
import argparse
import shutil
import os
import itertools


json_names = ["dataset_params", "log_params", "model_params", "trainer_params"]


def experiment_folders(path):
    for name in json_names:
        yield path / name


def dict_to_lists(d):
    """Generate all possible key-value pairs from a dict of lists

    Input dict:
    d = {'a': [1, 2, 3], 'b': [4, 5]}

    Output list:
    l = [[('a', 1), ('a', 2), ('a', 3)],
         [('b', 4), ('b', 5)]
    """

    def get_generator(k, l):
        return ((k, li) for li in l)

    for k, l in d.items():
        yield get_generator(k, l)


def dict_of_lists_to_list_of_dicts(d):
    """Generate a list of dictionaries from a dictionary of lists

    Input dict:
    d = {'a': [1, 2, 3], 'b': [4, 5]}

    Output list:
    l = [{'a': 1, 'b': 4},
         {'a': 2, 'b': 4},
         {'a': 3, 'b': 4},
         {'a': 1, 'b': 5},
         {'a': 2, 'b': 5},
         {'a': 3, 'b': 5}
        ]
    """
    key_value_pairs = dict_to_lists(d)
    return map(dict, itertools.product(*key_value_pairs))


def get_name_from_experiment(name, d):
    """Create an experiment name from the json names
    """
    name += "_"
    for v in d.values():
        fname = v.name
        if fname == "params.json":
            continue
        name += "".join(fname.split(".")[:-1])
        name += "_"
    return name[:-1]


def get_folders_content(path):
    """Returns a dict with folder name as key and content iterator as value
    """
    folder_contents = {}
    for folder_name, folder in zip(json_names, experiment_folders(path)):
        if not folder.exists():
            raise RuntimeError(
                f"The {folder_name} folder doesn' exist in the" " specified path"
            )

        folder_contents[folder_name] = folder.glob("*")
    return folder_contents


def get_all_experiments(name, path):
    """Create a dictionary with experiment name as key and experiment dict as value
    """
    folder_contents = get_folders_content(path)  # dict of lists of content
    experiment = dict_of_lists_to_list_of_dicts(folder_contents)

    return {(get_name_from_experiment(name, ex)): ex for ex in experiment}


def get_experiment_params(experiment_name, verbose, log_dir):
    return {
        "log_dir": log_dir,
        "name": experiment_name,
        "continue_old": False,
        "verbose": verbose,
    }


def make_experiment(
    path, experiment_name, experiment_info, verbosity, log_dir, output_path
):
    """Create all experiment directories.
    """
    experiment_dir = output_path / experiment_name
    if experiment_dir.is_dir():
        return
    experiment_dir.mkdir(parents=True)

    experiment_params = get_experiment_params(experiment_name, verbosity, log_dir)
    with (experiment_dir / "experiment_params.json").open("w") as f:
        json.dump(experiment_params, f)

    for filename, filepath in experiment_info.items():
        filename += ".json"
        new_path = experiment_dir / filename
        shutil.copy(filepath, new_path)


def create_experiment(name, path, verbosity, log_dir, output_path):
    experiments = get_all_experiments(name, path)  # dict of dicts
    for experiment_name, experiment_info in experiments.items():
        make_experiment(
            path, experiment_name, experiment_info, verbosity, log_dir, output_path
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create experiment directories for all possible permutations"
    )
    parser.add_argument("path", type=str)
    parser.add_argument("experiment_name", type=str)
    parser.add_argument(
        "--verbosity", type=int, default=1, help="Experiment verbosity level, default=1"
    )
    parser.add_argument(
        "--logdir", type=str, default="./logs/", help="Log directory, default=./logs/"
    )
    parser.add_argument(
        "--experimentdir",
        type=str,
        default=None,
        help="Directory to place the experiment files, default=PATH/experiments",
    )

    args = parser.parse_args()

    path = Path(args.path)
    if args.experimentdir is None:
        output_path = path / "experiments"
    else:
        output_path = Path(args.experimentdir)

    create_experiment(
        args.experiment_name, path, args.verbosity, args.logdir, output_path
    )
