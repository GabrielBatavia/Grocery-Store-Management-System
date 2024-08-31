from dotenv import load_dotenv
import os
import neptune
import matplotlib.pyplot as plt
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

# Muat file .env
load_dotenv()

# Inisialisasi Neptune
run = neptune.init_run(
    project="SkillQuest/SkillQuest",
    api_token=os.getenv("NEPTUNE_API_TOKEN"),
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

def main(production_params, use_fixed_length_padding=False):
    # Jalankan pipeline
    model, tokenizer, eval_data, training_logs = fine_tune_and_save_model(
        model_name="kalisai/Nusantara-7b-Indo-Chat",
        train_file=local_train_file,
        eval_file=local_eval_file,
        output_dir=local_output_dir,
        params=production_params,
        vocab_txt_file=[local_root_file, local_stop_file],
        vocab_json_file=local_slang_file,
        num_workers=production_params["num_workers"]
    )

    # Evaluasi model dan ambil skor FBERT
    avg_fbert_score, fbert_scores = evaluate_model_on_all_validation_data(model, tokenizer, eval_data, use_fixed_length_padding=use_fixed_length_padding)
    run["avg_fbert_score"].log(avg_fbert_score)

    # Visualisasi learning curve
    learning_curve_filename = f"{local_output_dir}/learning_curve.png"
    plot_learning_curve(training_logs, color='green', linestyle='--', marker='x')
    plt.title("Learning Curve")
    plt.savefig(learning_curve_filename)
    plt.close()
    run["learning_curve"].upload(learning_curve_filename)

    # Visualisasi evolusi FBERT score
    fbert_evolution_filename = f"{local_output_dir}/fbert_evolution.png"
    plot_fbert_evolution_both(fbert_scores)
    plt.title("FBERT Evolution")
    plt.savefig(fbert_evolution_filename)
    plt.close()
    run["fbert_evolution"].upload(fbert_evolution_filename)

    # Visualisasi distribusi panjang token pada data evaluasi
    token_length_distribution_filename = f"{local_output_dir}/token_length_distribution.png"
    plot_token_length_distribution(eval_data, title='Evaluation Data Token Length Distribution', color='lightgreen')
    plt.savefig(token_length_distribution_filename)
    plt.close()
    run["token_length_distribution"].upload(token_length_distribution_filename)

    # Visualisasi gabungan dari learning curve dan FBERT score
    combined_learning_fbert_filename = f"{local_output_dir}/combined_learning_fbert.png"
    plot_combined_learning_fbert(training_logs, fbert_scores)
    plt.title("Combined Learning and FBERT")
    plt.savefig(combined_learning_fbert_filename)
    plt.close()
    run["combined_learning_fbert"].upload(combined_learning_fbert_filename)

    # Visualisasi heatmap untuk beberapa contoh acak dari data evaluasi
    heatmap_filenames = visualize_random_samples_attention(model, tokenizer, eval_data, num_samples=3, use_fixed_length_padding=use_fixed_length_padding)
    for heatmap_filename in heatmap_filenames:
        run[f"heatmaps/{heatmap_filename}"].upload(heatmap_filename)

    return avg_fbert_score

if __name__ == "__main__":
    production_params = {
        "batch_size": 8,
        "num_train_epochs": 2,
        "learning_rate": 0.0005,
        "gradient_accumulation_steps": 16,
        "num_workers": 8  # Add a default num_workers or get this from Optuna
    }
    use_fixed_length_padding = False  # Set to True jika ingin padding dengan 'max_length=128'
    main(production_params, use_fixed_length_padding=use_fixed_length_padding)

    run.stop()