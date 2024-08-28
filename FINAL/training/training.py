import torch
from transformers import Trainer, TrainingArguments, AutoTokenizer, DataCollatorForLanguageModeling
from utils.dataset_utils import CustomDataset
from utils.model_utils import load_model, load_dataset, load_additional_vocab
from evaluation.evaluation import evaluate_model_on_all_validation_data
def fine_tune_and_save_model(model_name, train_file, eval_file, output_dir, params, vocab_txt_file=None, vocab_json_file=None):
    """
    Fine-tunes a pre-trained model on a custom dataset, saves the fine-tuned model and tokenizer, and evaluates the model on validation data.

    Parameters
    ----------
    model_name : str
        The name or path of the pre-trained model to be fine-tuned.
    train_file : str
        The path to the training dataset file in CSV format.
    eval_file : str
        The path to the evaluation dataset file in CSV format.
    output_dir : str
        The directory where the fine-tuned model, tokenizer, and model state will be saved.
    params : dict
        A dictionary containing hyperparameters for training, including:
        - "batch_size" (int): The batch size for training.
        - "gradient_accumulation_steps" (int): The number of gradient accumulation steps.
        - "num_train_epochs" (int): The number of training epochs.
        - "learning_rate" (float): The learning rate for the optimizer.

    Returns
    -------
    model : transformers.PreTrainedModel
        The fine-tuned model.
    tokenizer : transformers.PreTrainedTokenizer
        The tokenizer used in the fine-tuning process, saved along with the model.
    eval_data : list of dict
        The evaluation data used for model evaluation.

    Notes
    -----
    The function performs the following steps:
    1. Loads the tokenizer and model specified by `model_name`.
    2. Loads and preprocesses the training and evaluation datasets.
    3. Fine-tunes the model on the training data using the specified hyperparameters.
    4. Saves the fine-tuned model, tokenizer, and the model's state dictionary to `output_dir`.
    5. Evaluates the model on the validation data and prints the average FBERT score.

    The function prints the average FBERT score for the model on the validation data at the end of the process.
    The fine-tuned model's state dictionary is saved as a `.h5` file in `output_dir`.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load and add additional vocabulary
    if vocab_txt_file and vocab_json_file:
        additional_vocab = load_additional_vocab(vocab_txt_file, vocab_json_file)
        num_added_toks = tokenizer.add_tokens(additional_vocab)
        print(f"Added {num_added_toks} tokens to the tokenizer")

    model = load_model(model_name).to(device)

    # Resize the model's embedding matrix to match the new tokenizer length
    if vocab_txt_file and vocab_json_file and num_added_toks > 0:
        model.resize_token_embeddings(len(tokenizer))

    train_data = load_dataset(train_file)
    eval_data = load_dataset(eval_file)

    train_inputs = tokenizer([conv['conversations'][0]['value'] for conv in train_data], return_tensors='pt',
                             truncation=True, padding='max_length', max_length=128).to(device)
    train_labels = tokenizer([conv['conversations'][1]['value'] for conv in train_data], return_tensors='pt',
                             truncation=True, padding='max_length', max_length=128).to(device)

    train_dataset = CustomDataset(train_inputs, train_labels)

    eval_inputs = tokenizer([conv['conversations'][0]['value'] for conv in eval_data], return_tensors='pt',
                            truncation=True, padding='max_length', max_length=128).to(device)
    eval_labels = tokenizer([conv['conversations'][1]['value'] for conv in eval_data], return_tensors='pt',
                            truncation=True, padding='max_length', max_length=128).to(device)
    eval_dataset = CustomDataset(eval_inputs, eval_labels)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=params["batch_size"],
        gradient_accumulation_steps=params["gradient_accumulation_steps"],
        num_train_epochs=params["num_train_epochs"],
        learning_rate=params["learning_rate"],
        save_steps=500,
        save_total_limit=2,
        logging_strategy="epoch",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir='./logs',
        report_to="none",
        fp16=torch.cuda.is_available() and device.type == 'cuda',
        load_best_model_at_end=True,
        disable_tqdm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    torch.save(model.state_dict(), f"{output_dir}/fine_tuned_model.h5")

    avg_fbert_score, fbert_scores = evaluate_model_on_all_validation_data(model.to(device), tokenizer, eval_data)
    print(f"Average FBERT Score on Validation Data: {avg_fbert_score}")

    return model, tokenizer, eval_data, trainer.state.log_history