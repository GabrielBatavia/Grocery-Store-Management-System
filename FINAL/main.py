import neptune
from config.config import paths_config
from training.training import fine_tune_and_save_model
from visualization.visualization import (
    visualize_random_samples_attention,
    plot_learning_curve,
    plot_fbert_evolution_both,
    plot_token_length_distribution,
    plot_combined_learning_fbert
)
from evaluation.evaluation import evaluate_model_on_all_validation_data

# Inisialisasi Neptune
run = neptune.init_run(
    project="gabrielbatavia/Pintarpath",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2N2M2NTVmNi0yMWQwLTQzY2EtODY0Yy1kYzQ0Mjc2YTY3NzYifQ==",
    name="Production Fine-tuning Nusantara Chat Model",
    tags=["fine-tuning", "NLP", "Transformers", "production"],
    dependencies="infer",
    capture_hardware_metrics=True,
)

# Mengambil paths dari config.py
local_train_file = paths_config["train_file"]
local_eval_file = paths_config["eval_file"]
local_output_dir = paths_config["output_dir"]

# Paths untuk file tambahan vocabulary
local_root_file = paths_config["root_words_file"]
local_slang_file = paths_config["slang_words_file"]
local_stop_file = paths_config["stop_words_file"]

def main(production_params, trial_number=None):
    # Tentukan output directory berdasarkan trial_number jika ada
    trial_output_dir = f"{local_output_dir}/trial_{trial_number}" if trial_number is not None else local_output_dir

    # Jalankan pipeline
    model, eval_data, training_logs = fine_tune_and_save_model(
        model_name="kalisai/Nusantara-0.8b-Indo-Chat",
        train_file=local_train_file,
        eval_file=local_eval_file,
        output_dir=trial_output_dir,
        params=production_params,
        vocab_txt_file=vocab_txt_file,
        vocab_json_file=vocab_json_file
    )

    # Evaluasi model dan ambil skor FBERT
    avg_fbert_score, fbert_scores = evaluate_model_on_all_validation_data(model, tokenizer, eval_data)

    # Kirim hasil ke Neptune dengan menambahkan konteks trial jika ada
    if trial_number is not None:
        run[f"trial_{trial_number}/avg_fbert_score"].log(avg_fbert_score)
    else:
        run["avg_fbert_score"].log(avg_fbert_score)

    # Visualisasi learning curve dengan nama sesuai trial
    plot_learning_curve(training_logs, color='green', linestyle='--', marker='x')
    plt.title(f"Learning Curve - Trial {trial_number}" if trial_number is not None else "Learning Curve")
    plt.savefig(
        f"{trial_output_dir}/trial_{trial_number}_learning_curve.png" if trial_number is not None else f"{trial_output_dir}/learning_curve.png")
    plt.close()

    # Visualisasi evolusi FBERT score dengan nama sesuai trial
    plot_fbert_evolution_both(fbert_scores)
    plt.title(f"FBERT Evolution - Trial {trial_number}" if trial_number is not None else "FBERT Evolution")
    plt.savefig(
        f"{trial_output_dir}/trial_{trial_number}_fbert_evolution.png" if trial_number is not None else f"{trial_output_dir}/fbert_evolution.png")
    plt.close()

    # Visualisasi distribusi panjang token pada data evaluasi
    plot_token_length_distribution(eval_data,
                                   title=f'Evaluation Data Token Length Distribution - Trial {trial_number}' if trial_number is not None else 'Evaluation Data Token Length Distribution',
                                   color='lightgreen')
    plt.savefig(
        f"{trial_output_dir}/trial_{trial_number}_token_length_distribution.png" if trial_number is not None else f"{trial_output_dir}/token_length_distribution.png")
    plt.close()


    # Visualisasi gabungan dari learning curve dan FBERT score
    plot_combined_learning_fbert(training_logs, fbert_scores)
    plt.title(
        f"Combined Learning and FBERT - Trial {trial_number}" if trial_number is not None else "Combined Learning and FBERT")
    plt.savefig(
        f"{trial_output_dir}/trial_{trial_number}_combined_learning_fbert.png" if trial_number is not None else f"{trial_output_dir}/combined_learning_fbert.png")
    plt.close()

    return avg_fbert_score

    # Visualisasi heatmap untuk beberapa contoh acak dari data evaluasi
    visualize_random_samples_attention(model, tokenizer, eval_data, num_samples=3)

if __name__ == "__main__":
    use_optuna = True  # Ganti ini menjadi False jika tidak ingin menggunakan Optuna

    if use_optuna:
        from optuna_opt import run_optuna  # Import dari file optuna_opt.py

        run_optuna(main)
    else:
        production_params = {
            "batch_size": 8,
            "num_train_epochs": 2,
            "learning_rate": 0.0005,
            "gradient_accumulation_steps": 16,
        }
        main(production_params)

    run.stop()
