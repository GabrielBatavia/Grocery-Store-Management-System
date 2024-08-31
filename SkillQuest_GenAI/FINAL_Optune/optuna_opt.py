import optuna

def run_optuna(main_func):
    def objective(trial):
        production_params = {
            "batch_size": trial.suggest_int("batch_size", 64, 128),
            "num_train_epochs": trial.suggest_int("num_train_epochs", 1, 5),
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1e-3),
            "gradient_accumulation_steps": trial.suggest_int("gradient_accumulation_steps", 64, 128),
            "num_workers": trial.suggest_int("num_workers", 8, 20),
        }

        # Kirim trial_number ke fungsi main
        trial_number = trial.number
        avg_fbert_score = main_func(production_params, trial_number)
        return avg_fbert_score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)

    # Simpan hasil terbaik
    best_params = study.best_params
    print("Best Parameters:", best_params)