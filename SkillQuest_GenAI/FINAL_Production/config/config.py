# Configuration file example (if used)

training_config = {
    "batch_size": 32,
    "num_train_epochs": 4,
    "learning_rate": 0.0005005632564246182,
    "gradient_accumulation_steps": 16,
}

paths_config = {
    "train_file": "data/train/train_data.csv",
    "eval_file": "data/test/test_data.csv",
    "output_dir": "models/fine-tuned-model-production",
    "root_words_file": "data/indonesian_word/combined_root_words.txt",
    "slang_words_file": "data/indonesian_word/combined_slang_words.txt",
    "stop_words_file": "data/indonesian_word/combined_stop_words.txt"
}
