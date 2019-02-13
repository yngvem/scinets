import tensorflow as tf
from scinets.utils.experiment import NetworkExperiment


if __name__ == "__main__":
    print("Starting experiment")
    experiment_params = {
        "log_dir": "/home/yngve/logs/",
        "name": "test",
        "continue_old": False,
        "verbose": True,
    }
    dataset_params = {
        "operator": "HDFDataset",
        "arguments": {
            "data_path": "/home/yngve/dataset_extraction/val_split_2d.h5",
            # Husk: i Windows, dobbel backslash, ikke enkel. e.g. C:\\users\\yngve
            "batch_size": [8, 8, 1],
            "val_group": "val",  # default validation
            "test_group": "test",
            "train_group": "train",
        },
    }
    model_params = {
        "type": "UNet",
        "network_params": {
            "loss_function": {"operator": "BinaryFBeta", "arguments": {"beta": 2}},
            "architecture": [
                {
                    "layer": "Conv2D",
                    "scope": "conv1",
                    "layer_params": {"out_size": 8, "k_size": 3, "strides": 4},
                    "normalizer": {"operator": "BatchNormalization"},
                    "activation": {"operator": "RElU"},
                    "initializer": {"operator": "he_normal"},
                    "regularizer": {
                        "operator": "WeightDecay",
                        "arguments": {"amount": 1},
                    },
                },
                {
                    "layer": "ResnetConv2D",
                    "scope": "conv2",
                    "layer_params": {"out_size": 8, "k_size": 3, "strides": 4},
                    "normalizer": {"operator": "BatchNormalization"},
                    "activation": {"operator": "RElU"},
                    "initializer": {"operator": "he_normal"},
                },
                {
                    "layer": "ResnetConv2D",
                    "scope": "conv3",
                    "layer_params": {"out_size": 8, "k_size": 3},
                    "normalizer": {"operator": "BatchNormalization"},
                    "activation": {"operator": "RElU"},
                    "initializer": {"operator": "he_normal"},
                },
                {
                    "layer": "ResnetConv2D",
                    "scope": "conv4",
                    "layer_params": {"out_size": 16, "k_size": 3},
                    "normalizer": {"operator": "BatchNormalization"},
                    "activation": {"operator": "RElU"},
                    "initializer": {"operator": "he_normal"},
                },
                {
                    "layer": "LinearInterpolate",
                    "scope": "linear_upsample_1",
                    "layer_params": {"rate": 4},
                },
                {
                    "layer": "ResnetConv2D",
                    "scope": "conv5",
                    "layer_params": {"out_size": 16, "k_size": 3},
                    "normalizer": {"operator": "BatchNormalization"},
                    "activation": {"operator": "RElU"},
                    "initializer": {"operator": "he_normal"},
                },
                {
                    "layer": "LinearInterpolate",
                    "scope": "linear_upsample_2",
                    "layer_params": {"rate": 4},
                },
                {
                    "layer": "ResnetConv2D",
                    "scope": "conv6",
                    "layer_params": {"out_size": 32, "k_size": 3},
                    "normalizer": {"operator": "BatchNormalization"},
                    "activation": {"operator": "RElU"},
                    "initializer": {"operator": "he_normal"},
                },
                {
                    "layer": "ResnetConv2D",
                    "scope": "conv7",
                    "layer_params": {"out_size": 1, "k_size": 3},
                    "normalizer": {"operator": "BatchNormalization"},
                    "activation": {"operator": "Sigmoid"},
                    "initializer": {"operator": "he_normal"},
                },
            ],
            "skip_connections": [
                ["input", "linear_upsample_2"],
                ["conv1", "linear_upsample_1"],
            ],
        },
    }

    trainer_params = {
        "save_step": 10,
        "train_op": {
            "operator": "MomentumOptimizer",
            "arguments": {
                "momentum": 0.9,
                # Optional, either this or scheduler: "learning_rate": 0.001
            },
        },
        "learning_rate_scheduler": {
            "operator": "CosineDecayRestarts",
            "arguments": {
                "learning_rate": 0.05,
                "first_decay_steps": 1300,
                "t_mul": 10,
                "m_mul": 1,
                "alpha": 0.01,
            },
        },
    }

    log_params = {
        "val_log_frequency": 10,
        "evaluator": {"operator": "BinaryClassificationEvaluator"},
        "loggers": [
            {
                "operator": "TensorboardLogger",
                "arguments": {
                    "log_dicts": [
                        {"log_name": "Loss", "log_var": "loss", "log_type": "scalar"},
                        {
                            "log_name": "Probability_map",
                            "log_var": "probabilities",
                            "log_type": "image",
                            "log_kwargs": {"max_outputs": 1},
                        },
                        {
                            "log_name": "Accuracy",
                            "log_var": "accuracy",
                            "log_type": "scalar",
                        },
                        {"log_name": "Dice", "log_var": "dice", "log_type": "scalar"},
                        {
                            "log_name": "Mask",
                            "log_var": "true_out",
                            "log_type": "image",
                            "log_kwargs": {"max_outputs": 1},
                        },
                        {
                            "log_name": "CT",
                            "log_var": "input",
                            "log_type": "image",
                            "log_kwargs": {"max_outputs": 1, "channel": 0},
                        },
                        {
                            "log_name": "PET",
                            "log_var": "input",
                            "log_type": "image",
                            "log_kwargs": {"max_outputs": 1, "channel": 1},
                        },
                        {
                            "log_name": "Probability_map",
                            "log_var": "probabilities",
                            "log_type": "histogram",
                        },
                        {
                            "log_name": "Precision",
                            "log_var": "precision",
                            "log_type": "scalar",
                        },
                        {
                            "log_name": "Recall",
                            "log_var": "recall",
                            "log_type": "scalar",
                        },
                        {
                            "log_name": "true positives",
                            "log_var": "true_positives",
                            "log_type": "scalar",
                        },
                        {
                            "log_name": "true negatives",
                            "log_var": "true_negatives",
                            "log_type": "scalar",
                        },
                    ]
                },
            },
            {
                "operator": "HDF5Logger",
                "arguments": {
                    "log_dicts": [
                        {"log_name": "Loss", "log_var": "loss"},
                        {"log_name": "Accuracy", "log_var": "accuracy"},
                        {"log_name": "Dice", "log_var": "dice"},
                    ]
                },
            },
        ],
        "network_tester": {
            "metrics": [
                "dice",
                "precision",
                "sensitivity",
                "specificity",
                "true_positives",
                "true_negatives",
            ]
        },
    }

    experiment = NetworkExperiment(
        experiment_params=experiment_params,
        model_params=model_params,
        dataset_params=dataset_params,
        trainer_params=trainer_params,
        log_params=log_params,
    )
    experiment.train(100)
    best_it, result, result_std = experiment.find_best_model("val", "dice")
    print(f'{" Final score ":=^80s}')
    print(
        f" Achieved a {eval_metric:s} of {result:.3f}, with a standard "
        f"deviation of {result_std:.3f}"
    )
    print(f" This result was achieved at iteration {best_it}")
    print(80 * "=")
