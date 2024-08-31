import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments, AutoTokenizer, DataCollatorForLanguageModeling
from utils.dataset_utils import CustomDataset
from utils.model_utils import load_model, load_dataset, load_additional_vocab
from evaluation.evaluation import evaluate_model_on_all_validation_data

def fine_tune_and_save_model(model_name, train_file, eval_file, output_dir, params, vocab_txt_file=None, vocab_json_file=None, num_workers=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load and add additional vocabulary
    if vocab_txt_file and vocab_json_file:
        additional_vocab = load_additional_vocab(vocab_txt_file[0], vocab_txt_file[1], vocab_json_file)
        num_added_toks = tokenizer.add_tokens(additional_vocab)
        print(f"Added {num_added_toks} tokens to the tokenizer")

    model = load_model(model_name)

    # Resize the model's embedding matrix to match the new tokenizer length
    if vocab_txt_file and vocab_json_file and num_added_toks > 0:
        model.resize_token_embeddings(len(tokenizer))

    train_data = load_dataset(train_file)
    eval_data = load_dataset(eval_file)

    train_inputs = tokenizer([conv['conversations'][0]['value'] for conv in train_data], return_tensors='pt',
                             truncation=True, padding='max_length', max_length=128)
    train_labels = tokenizer([conv['conversations'][1]['value'] for conv in train_data], return_tensors='pt',
                             truncation=True, padding='max_length', max_length=128)

    eval_inputs = tokenizer([conv['conversations'][0]['value'] for conv in eval_data], return_tensors='pt',
                            truncation=True, padding='max_length', max_length=128)
    eval_labels = tokenizer([conv['conversations'][1]['value'] for conv in eval_data], return_tensors='pt',
                            truncation=True, padding='max_length', max_length=128)

    # Move data to the device
    try:
        train_inputs, train_labels = train_inputs.to(device), train_labels.to(device)
        eval_inputs, eval_labels = eval_inputs.to(device), eval_labels.to(device)
    except RuntimeError as e:
        print("GPU memory is full, continuing training on CPU...")
        device = torch.device("cpu")
        train_inputs, train_labels = train_inputs.to(device), train_labels.to(device)
        eval_inputs, eval_labels = eval_inputs.to(device), eval_labels.to(device)

    train_dataset = CustomDataset(train_inputs, train_labels)
    eval_dataset = CustomDataset(eval_inputs, eval_labels)

    if num_workers is None:
        num_workers = 8

    # Using DataLoader with multiple workers for asynchronous data loading
    train_dataloader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True, num_workers=num_workers)
    eval_dataloader = DataLoader(eval_dataset, batch_size=params["batch_size"], shuffle=False, num_workers=num_workers)

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

    # Mengatur parallelism antara CPU dan GPU
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model.to(device)  # Pindahkan model ke device yang tersedia

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save model in PyTorch state_dict format
    torch.save(model.state_dict(), f"{output_dir}/fine_tuned_model_state_dict.pth")

    # Save model in TorchScript format
    scripted_model = torch.jit.script(model)
    scripted_model.save(f"{output_dir}/fine_tuned_model_torchscript.pt")

    # Save model in ONNX format
    dummy_input = torch.randn(1, 128, device=device)  # Sesuaikan ukuran input sesuai modelmu
    torch.onnx.export(model, dummy_input, f"{output_dir}/fine_tuned_model.onnx",
                      input_names=['input'], output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

    print("Model has been saved in PyTorch state_dict, TorchScript, and ONNX formats.")

    avg_fbert_score, fbert_scores = evaluate_model_on_all_validation_data(model.to(device), tokenizer, eval_data)
    print(f"Average FBERT Score on Validation Data: {avg_fbert_score}")

    return model, tokenizer, eval_data, trainer.state.log_history
