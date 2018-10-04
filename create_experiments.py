__author__ = 'Yngve Mardal Moe'
__email__ = 'yngve.m.moe@gmail.com'


from pathlib import Path
import shutil
import os
import itertools


json_names = ['dataset_params', 'experiment_params', 'log_params', 
              'model_params', 'trainer_params']


def experiment_folders(path):
    for name in json_names:
        yield path/name


def dict_to_lists(d):
    def get_generator(k, v):
        return ((k, vi) for vi in v)

    for k, v in d.items():
        yield get_generator(k, v)


def dict_of_lists_to_list_of_dicts(d):
    return map(dict, itertools.product(*dict_to_lists(d)))


def get_name_from_experiment(name, d):
    name += '_'
    for v in d.values():
        fname = v.name
        if fname == 'params.json':
            continue
        name += ''.join(fname.split('.')[:-1])
        name += '_'
    return name[:-1]


def get_folders_content(path):
    """Returns a dict with folder name as key and content iterator as value
    """
    folder_contents = {}
    for folder_name, folder in zip(json_names, experiment_folders(path)):
        if not folder.exists():
            raise RuntimeError(f'The {folder_name} folder doesn\' exist in the'
                               ' specified path')

        folder_contents[folder_name] = folder.glob('*')
    return folder_contents


def get_all_experiments(name, path):
    folder_contents = get_folders_content(path) # dict of lists of content
    experiment = dict_of_lists_to_list_of_dicts(folder_contents)

    return {(get_name_from_experiment(name, ex)): ex for ex in experiment}


def make_experiment(path, experiment_name, experiment_info):
    experiment_dir = path / 'experiments' / experiment_name
    experiment_dir.mkdir(parents=True)
    for filename, filepath in experiment_info.items():
        filename += 'json'
        new_path = experiment_dir / filename
        shutil.copy(filepath, new_path)


def create_experiment(name, path):
    experiments = get_all_experiments(name, path)  # dict of dicts
    for experiment_name, experiment_info in experiments.items():
        make_experiment(path, experiment_name, experiment_info)
    

if __name__ == '__main__':
    import sys
    path = Path(sys.argv[1])
    name = sys.argv[2]
    create_experiment(name, path)