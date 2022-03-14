"""Experiment that runs grid search over BERT parameters"""

from src.experiment.experiment import ExperimentRunner, make_dictionaries

if __name__ == "__main__":
    # Bert grid search here
    runner = ExperimentRunner("bert_models")
    runner.add_grid_experiments(
        modality="text",
        model="bert",
        model_name=[
            "bert_en_uncased_L-2_H-128_A-2",
            "bert_en_uncased_L-4_H-128_A-2",
            "bert_en_uncased_L-4_H-256_A-4",
            "bert_en_uncased_L-2_H-256_A-4",
            "bert_en_uncased_L-6_H-256_A-4",
            "bert_en_uncased_L-4_H-512_A-8",
        ],
    )
    runner.run_all()
    # Best model config parameters
    best_model = runner.experiments[runner.best_index].model_name
    train_parameters = make_dictionaries(
        init_lr=[1e-5, 3e-5], dropout_rate=[0.1, 0.2], dense_layer=[0, 512]
    )
    runner = ExperimentRunner("bert_parameters")
    runner.add_grid_experiments(
        modality="text",
        model="bert",
        model_name=best_model,
        train_parameters=train_parameters,
    )
    runner.run_all()

    # Distilbert grid search here
    train_parameters = make_dictionaries(
        init_lr=[1e-5, 3e-5], dropout_rate=[0.1, 0.2], dense_layer=[0, 512]
    )
    runner = ExperimentRunner("distilbert_parameters")
    runner.add_grid_experiments(
        modality="text", model="distilbert", train_parameters=train_parameters
    )
    runner.run_all()
