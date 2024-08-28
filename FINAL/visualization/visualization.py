import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import random

def plot_attention_heatmap(attention_scores, input_tokens, output_tokens, example_idx, cmap='viridis', annot=False, color_bar=True, vmin=None, vmax=None):
    """
    Generate and save a heatmap visualization of attention scores for a specific example in a sequence-to-sequence model.

    Parameters
    ----------
    attention_scores : numpy.ndarray
        A 2D array representing the attention scores. Each value indicates the level of attention an output token gives to an input token.
    input_tokens : list of str
        A list of tokens from the input sequence.
    output_tokens : list of str
        A list of tokens from the output sequence.
    example_idx : int
        The index of the example, used for labeling the saved heatmap file.
    cmap : str, optional
        The colormap used for the heatmap, by default 'viridis'.
    annot : bool, optional
        Whether to annotate the heatmap with the attention score values, by default False.
    color_bar : bool, optional
        Whether to include a color bar in the heatmap, by default True.
    vmin : float, optional
        Minimum value for the heatmap scale, by default None.
    vmax : float, optional
        Maximum value for the heatmap scale, by default None.

    Returns
    -------
    None
        The function saves the heatmap as an image file named "attention_heatmap_example_{example_idx + 1}.png".
    """

    truncated_output_tokens = output_tokens[:attention_scores.shape[0]]
    attention_scores = attention_scores[:len(truncated_output_tokens), :len(input_tokens)]
    df = pd.DataFrame(attention_scores, index=truncated_output_tokens, columns=input_tokens[:len(input_tokens)])

    plt.figure(figsize=(15, 10))
    sns.heatmap(df, annot=annot, cmap=cmap, cbar=color_bar, vmin=vmin, vmax=vmax)
    plt.xlabel("Input Tokens")
    plt.ylabel("Output Tokens")
    plt.title(f"Attention Heatmap Example {example_idx + 1}")
    plt.tight_layout()
    plt.savefig(f"attention_heatmap_example_{example_idx + 1}_{cmap}_annot_{annot}.png")
    plt.close()

    print(f"Attention heatmap for example {example_idx + 1} saved as attention_heatmap_example_{example_idx + 1}_{cmap}_annot_{annot}.png")


def visualize_random_samples_attention(model, tokenizer, eval_data, num_samples=3):
    """
    Select random samples from the evaluation data, generate model predictions, and visualize the attention mechanisms using heatmaps.

    Parameters
    ----------
    model : transformers.PreTrainedModel
        The pre-trained sequence-to-sequence model used for generating predictions and attention scores.
    tokenizer : transformers.PreTrainedTokenizer
        The tokenizer associated with the model, used for encoding input text and decoding output predictions.
    eval_data : list of dict
        A list of evaluation data samples. Each sample should contain 'conversations', which are dictionaries with human inputs and expected model outputs.
    num_samples : int, optional
        The number of random samples to visualize, by default 3.

    Returns
    -------
    None
        The function generates and saves heatmaps for each sample, but does not return any value.
    """

    cmaps = ['viridis', 'coolwarm', 'RdBu']  # Colormaps to use
    annot_options = [True, False]  # Whether to annotate or not
    color_bar_options = [True, False]  # Whether to include color bar or not
    vmin_vmax_options = [(None, None), (0, 1)]  # vmin and vmax options

    for i in range(num_samples):
        sample = random.choice(eval_data)
        human_input = sample['conversations'][0]['value']
        gpt_output = sample['conversations'][1]['value']

        inputs = tokenizer(human_input, return_tensors='pt', truncation=True, padding='max_length', max_length=128)
        inputs = {key: val.to(model.device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id,
                                     output_attentions=True, return_dict_in_generate=True)

        decoded_pred = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True,
                                        clean_up_tokenization_spaces=True)
        decoded_label = gpt_output

        print(f"--- Sample {i + 1} ---")
        print("Human Input:")
        print(human_input)
        print("\nModel Output (Prediction):")
        print(decoded_pred)
        print("\nExpected Output:")
        print(decoded_label)
        print("\n" + "-" * 50 + "\n")

        attentions = outputs.attentions

        attention_scores = attentions[-1][0][0].detach().cpu().numpy()  # Ambil head pertama
        input_tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'].squeeze())
        output_tokens = tokenizer.convert_ids_to_tokens(outputs.sequences[0].squeeze())

        for cmap in cmaps:
            for annot in annot_options:
                for vmin, vmax in vmin_vmax_options:
                    plot_attention_heatmap(attention_scores, input_tokens, output_tokens, i, cmap=cmap, annot=annot, color_bar=True, vmin=vmin,vmax=vmax)


def plot_learning_curve(training_logs, color='blue', linestyle='-', marker='o'):
    """
    Plots the learning curve of the model during training with custom color and style.

    Parameters
    ----------
    training_logs : list of dict
        List of logs from each epoch containing loss and other metrics.
    color : str, optional
        Color of the line, by default 'blue'.
    linestyle : str, optional
        Style of the line, by default '-'.
    marker : str, optional
        Marker style, by default 'o'.

    Returns
    -------
    None
        The function plots the learning curve and shows it.
    """
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
    """
    Plots the evolution of FBERT scores during training with two different gradient color maps.
    One where higher scores are represented by brighter colors (green),
    and another where higher scores are represented by darker colors (red).

    Parameters
    ----------
    fbert_scores : list of float
        List of FBERT scores for each epoch.

    Returns
    -------
    None
        The function plots the FBERT score evolution with two different color maps and shows both.
    """
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
    """
    Plots a histogram showing the distribution of token lengths in the dataset.

    Parameters
    ----------
    data : list of dict
        The dataset, where each entry is a dictionary containing input and output tokens.
    title : str, optional
        Title of the plot, by default 'Token Length Distribution'.
    color : str, optional
        Color of the histogram bars, by default 'skyblue'.

    Returns
    -------
    None
        The function plots the token length distribution and shows it.
    """
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
    """
    Plots the combined learning curve and FBERT score evolution in a single figure.

    Parameters
    ----------
    training_logs : list of dict
        List of logs from each epoch containing loss and other metrics.
    fbert_scores : list of float
        List of FBERT scores for each epoch.

    Returns
    -------
    None
        The function plots the combined graph and shows it.
    """
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



