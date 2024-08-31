import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import random

def plot_attention_heatmap(attention_scores, input_tokens, output_tokens, example_idx, cmap='viridis', annot=False, color_bar=True, vmin=0, vmax=1):
    # Filter out <|endoftext|> tokens from both input and output tokens
    filtered_input_tokens = [token for token in input_tokens if token != "<|endoftext|>"]
    filtered_output_tokens = [token for token in output_tokens if token != "<|endoftext|>"]

    truncated_output_tokens = filtered_output_tokens[:attention_scores.shape[0]]
    filtered_attention_scores = attention_scores[:len(truncated_output_tokens), :len(filtered_input_tokens)]

    df = pd.DataFrame(filtered_attention_scores, index=truncated_output_tokens, columns=filtered_input_tokens)

    plt.figure(figsize=(15, 10))
    sns.heatmap(df, annot=annot, cmap=cmap, cbar=color_bar, vmin=vmin, vmax=vmax)
    plt.xlabel("Input Tokens")
    plt.ylabel("Output Tokens")
    plt.title(f"Attention Heatmap Example {example_idx + 1}")
    plt.tight_layout()
    plt.savefig(f"attention_heatmap_example_{example_idx + 1}_{cmap}_annot_{annot}.png")
    plt.close()

    print(f"Attention heatmap for example {example_idx + 1} saved as attention_heatmap_example_{example_idx + 1}_{cmap}_annot_{annot}.png")

def visualize_random_samples_attention(model, tokenizer, eval_data, num_samples=3, use_fixed_length_padding=False):
    # Tentukan metode padding berdasarkan flag use_fixed_length_padding
    padding_strategy = 'max_length' if use_fixed_length_padding else True
    max_length = 128 if use_fixed_length_padding else None

    cmaps = ['viridis', 'coolwarm', 'RdBu']  # Colormaps to use
    annot_options = [True, False]  # Whether to annotate or not
    vmin_vmax_options = [(None, None), (0, 1)]  # vmin and vmax options

    heatmap_filenames = []

    for i in range(num_samples):
        sample = random.choice(eval_data)
        human_input = sample['conversations'][0]['value']
        gpt_output = sample['conversations'][1]['value']

        inputs = tokenizer(human_input, return_tensors='pt', truncation=True, padding=padding_strategy, max_length=max_length)
        inputs = {key: val.to(model.device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id, output_attentions=True, return_dict_in_generate=True)

        decoded_pred = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        decoded_label = gpt_output

        attentions = outputs.attentions
        attention_scores = attentions[-1][0][0].detach().cpu().numpy()  # Ambil head pertama
        input_tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'].squeeze())
        output_tokens = tokenizer.convert_ids_to_tokens(outputs.sequences[0].squeeze())

        for cmap in cmaps:
            for annot in annot_options:
                for vmin, vmax in vmin_vmax_options:
                    heatmap_filename = f"attention_heatmap_example_{i + 1}_{cmap}_annot_{annot}.png"
                    plot_attention_heatmap(attention_scores, input_tokens, output_tokens, i, cmap=cmap, annot=annot, color_bar=True, vmin=vmin, vmax=vmax)
                    heatmap_filenames.append(heatmap_filename)

    return heatmap_filenames

def plot_learning_curve(training_logs, color='blue', linestyle='-', marker='o'):
    epochs = range(1, len(training_logs) + 1)
    train_losses = [log['loss'] for log in training_logs]
    eval_losses = [log['eval_loss'] for log in training_logs]

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label='Training Loss', color=color, linestyle=linestyle, marker=marker)
    plt.plot(epochs, eval_losses, label='Validation Loss', color='red', linestyle=linestyle, marker=marker)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Learning Curve')
    plt.legend()
    plt.show()

def plot_fbert_evolution_both(fbert_scores):
    epochs = range(1, len(fbert_scores) + 1)
    norm = plt.Normalize(min(fbert_scores), max(fbert_scores))

    # First plot: Brighter = Higher Score (Green)
    colormap = plt.cm.Greens
    plt.figure(figsize=(10, 6))
    for i, epoch in enumerate(epochs):
        color = colormap(norm(fbert_scores[i]))
        plt.plot(epoch, fbert_scores[i], marker='o', color=color, markersize=8)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('FBERT Score', fontsize=12)
    plt.title("FBERT Score Evolution (Brighter = Higher Score)", fontsize=16, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=colormap), label='FBERT Score')
    plt.tight_layout()
    plt.show()

    # Second plot: Darker = Higher Score (Red)
    colormap = plt.cm.Reds
    plt.figure(figsize=(10, 6))
    for i, epoch in enumerate(epochs):
        color = colormap(norm(fbert_scores[i]))
        plt.plot(epoch, fbert_scores[i], marker='o', color=color, markersize=8)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('FBERT Score', fontsize=12)
    plt.title("FBERT Score Evolution (Darker = Higher Score)", fontsize=16, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=colormap), label='FBERT Score')
    plt.tight_layout()
    plt.show()

def plot_token_length_distribution(data, title='Token Length Distribution', color='skyblue'):
    input_lengths = [len(conv['conversations'][0]['value'].split()) for conv in data]
    output_lengths = [len(conv['conversations'][1]['value'].split()) for conv in data]

    plt.figure(figsize=(10, 5))
    plt.hist(input_lengths, bins=20, alpha=0.7, label='Input Lengths', color=color)
    plt.hist(output_lengths, bins=20, alpha=0.7, label='Output Lengths', color='salmon')
    plt.xlabel('Number of Tokens')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.legend()
    plt.show()

def plot_combined_learning_fbert(training_logs, fbert_scores):
    epochs = range(1, len(training_logs) + 1)
    train_losses = [log['loss'] for log in training_logs]
    eval_losses = [log['eval_loss'] for log in training_logs]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plotting Loss
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color='tab:blue')
    ax1.plot(epochs, train_losses, label='Training Loss', color='tab:blue', linestyle='-')
    ax1.plot(epochs, eval_losses, label='Validation Loss', color='tab:cyan', linestyle='--')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Adding a second y-axis for FBERT Scores
    ax2 = ax1.twinx()
    ax2.set_ylabel('FBERT Score', color='tab:red')
    ax2.plot(epochs, fbert_scores, label='FBERT Score', color='tab:red', linestyle=':')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    fig.tight_layout()
    plt.title('Combined Learning Curve and FBERT Score Evolution')
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
    plt.show()
