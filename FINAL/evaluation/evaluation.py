import torch
from bert_score import score as bert_score
from visualization.visualization import plot_attention_heatmap


def evaluate_model_on_all_validation_data(model, tokenizer, eval_data):
    """
    Evaluates a pre-trained sequence-to-sequence model on a given validation dataset, computes FBERT scores, and visualizes attention mechanisms.

    Parameters
    ----------
    model : transformers.PreTrainedModel
        The pre-trained sequence-to-sequence model used for generating predictions and attention scores.
    tokenizer : transformers.PreTrainedTokenizer
        The tokenizer associated with the model, used for encoding input text and decoding output predictions.
    eval_data : list of dict
        A list of evaluation data samples. Each sample should contain 'conversations', which are dictionaries with human inputs and expected model outputs.

    Returns
    -------
    avg_fbert_score : float
        The average FBERT score across all validation examples.
    F1_scores : list of float
        A list of FBERT F1 scores for each evaluated example.

    Notes
    -----
    The function performs the following steps:
    1. Iterates over the evaluation data, generating model predictions for each human input.
    2. Decodes the model's predictions and the corresponding reference outputs (expected GPT outputs).
    3. Collects attention scores and visualizes them as heatmaps for each evaluated example.
    4. Computes FBERT scores (Precision, Recall, F1) for the predictions against the references.
    5. Returns the average FBERT F1 score and a list of F1 scores for individual examples.

    The attention heatmaps are saved as image files, and errors encountered during the heatmap generation are logged.
    """

    model.eval()  # Set model to evaluation mode
    decoded_preds = []
    decoded_labels = []
    attention_data = []

    for idx, entry in enumerate(eval_data):
        human_input = None
        gpt_output = None

        for conv in entry['conversations']:
            if conv['from'] == 'human':
                human_input = conv['value']
            elif conv['from'] == 'gpt':
                gpt_output = conv['value']

        # Pastikan bahwa human_input dan gpt_output telah didefinisikan
        if human_input is not None and gpt_output is not None:
            inputs = tokenizer(human_input, return_tensors='pt', truncation=True, padding='max_length', max_length=128)
            inputs = {key: val.to(model.device) for key, val in inputs.items()}  # Move to the same device as the model

            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id,
                                         output_attentions=True, return_dict_in_generate=True)

            decoded_pred = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True,
                                            clean_up_tokenization_spaces=True)
            decoded_preds.append(decoded_pred)
            decoded_labels.append(gpt_output)

            attentions = outputs.attentions

            attention_scores = attentions[-1][0].squeeze().detach().cpu().numpy()  # (seq_len, seq_len)
            input_tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'].squeeze())
            output_tokens = tokenizer.convert_ids_to_tokens(outputs.sequences[0].squeeze())
            attention_data.append((attention_scores, input_tokens, output_tokens, idx))
        else:
            print(f"Skipping example {idx + 1} due to missing human or gpt output.")

    P, R, F1 = bert_score(decoded_preds, decoded_labels, lang="id", rescale_with_baseline=True)
    avg_fbert_score = torch.mean(F1).item()

    for data in attention_data:
        attention_scores, input_tokens, output_tokens, idx = data
        try:
            #plot_attention_heatmap(attention_scores, input_tokens, output_tokens, idx)
        except Exception as e:
            print(f"Error in generating heatmap for example {idx + 1}: {e}")

    return avg_fbert_score, F1.tolist()
