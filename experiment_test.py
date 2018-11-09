import tensorflow as tf
from scinets.utils.experiment import NetworkExperiment, MNISTExperiment


if __name__ == "__main__":
    experiment_params = {
        "log_dir": "./logs/",
        "name": "test_experiment",
        "continue_old": False,
        "num_steps": 1000,
    }
    dataset_params = {
        "data_path": "/home/yngve/dataset_extraction/val_split_2d.h5",
        "batch_size": [64, 64, 1],
        "val_group": "val",
    }
    architecture = [
        {
            "layer": "ResnetConv2D",
            "scope": "resnet1",
            "layer_params": {"out_size": 8, "k_size": 3},
            "normalization": {"operator": "batch_normalization"},
            "activation": {"operator": "relu"},
        },
        {
            "layer": "ResnetConv2D",
            "scope": "resnet2",
            "layer_params": {"out_size": 16, "k_size": 3, "strides": 2},
            "normalization": {"operator": "batch_normalization"},
            "activation": {"operator": "relu"},
        },
        {
            "layer": "ResnetConv2D",
            "scope": "resnet3",
            "layer_params": {"out_size": 16, "k_size": 3},
            "normalization": {"operator": "batch_normalization"},
            "activation": {"operator": "relu"},
        },
        {
            "layer": "LinearInterpolate",
            "scope": "upsample",
            "layer_params": {"rate": 2},
        },
        {
            "layer": "ResnetConv2D",
            "scope": "resnet4",
            "layer_params": {"out_size": 32, "k_size": 3, "strides": 2},
            "normalization": {"operator": "batch_normalization"},
            "activation": {"operator": "relu"},
        },
        {
            "layer": "ResnetConv2D",
            "scope": "resnet5",
            "layer_params": {"out_size": 10, "k_size": 3},
            "normalization": {"operator": "batch_normalization"},
            "activation": {"operator": "linear"},
        },
        {"layer": "GlobalAveragePool", "scope": "global_average"},
    ]
    model_params = {
        "type": "NeuralNet",
        "network_params": {
            "loss_function": "sigmoid_cross_entropy_with_logits",
            "loss_kwargs": {},
            "architecture": architecture,
            "verbose": True,
        },
    }

    trainer_params = {}

    log_params = {
        "val_log_frequency": 10,
        "evaluator": "BinaryClassificationEvaluator",
        "tb_params": {
            "log_dicts": [
                {"log_name": "Loss", "log_var": "loss", "log_type": "scalar"},
                {"log_name": "Accuracy", "log_var": "accuracy", "log_type": "scalar"},
                {"log_name": "Dice", "log_var": "dice", "log_type": "scalar"},
                {
                    "log_name": "Images",
                    "log_var": "input",
                    "log_type": "image",
                    "log_kwargs": {"max_outputs": 1, "channel": 0},
                },
            ]
        },
    }

    experiment = MNISTExperiment(
        experiment_params=experiment_params,
        model_params=model_params,
        dataset_params=dataset_params,
        trainer_params=trainer_params,
        log_params=log_params,
    )
