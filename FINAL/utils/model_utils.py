import torch
from transformers import AutoModelForCausalLM
import pandas as pd

def load_model(model_name):
    """
    Loads a causal language model using the specified model name. The model is loaded onto the GPU if available;
    otherwise, it falls back to the CPU. In case of a GPU out-of-memory error, the model will be reloaded on the CPU.

    Args:
        model_name (str): The name or path of the pre-trained model to be loaded.

    Returns:
        model (AutoModelForCausalLM): The loaded causal language model, either on GPU or CPU.
    """

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name).to('cuda' if torch.cuda.is_available() else 'cpu')
        print("Model loaded on GPU" if torch.cuda.is_available() else "Model loaded on CPU")
    except RuntimeError as e:
        if 'out of memory' in str(e):
            print("Out of memory error on GPU. Loading model on CPU...")
            model = AutoModelForCausalLM.from_pretrained(model_name).to('cpu')
        else:
            raise e
    return model

def load_dataset(csv_file):
    """
    Loads a dataset from a CSV file and processes it into a list of dictionaries where each dictionary
    represents a conversation.

    Args:
        csv_file (str): The path to the CSV file containing the dataset. The file should have a column
                        named 'conversations' where each entry is a string representation of a list.

    Returns:
        list[dict]: A list of dictionaries where each dictionary corresponds to a row in the dataset
                    with the 'conversations' column converted to a list.
    """

    data = pd.read_csv(csv_file)
    data['conversations'] = data['conversations'].apply(lambda x: eval(x))
    return data.to_dict(orient='records')


def load_words(filename):
    """
    Loads words from a text file, where each line contains one word.

    Args:
        filename (str): Path to the text file.

    Returns:
        list: A list of words.
    """
    with open(filename, 'r', encoding='utf-8') as file:
        words = file.read().splitlines()
    return words


def load_slang_words(filename):
    """
    Loads slang words from a JSON file, where keys are slang words and values are their standard forms.

    Args:
        filename (str): Path to the JSON file.

    Returns:
        dict: A dictionary of slang words mapping to their standard forms.
    """
    with open(filename, 'r', encoding='utf-8') as file:
        slang_dict = json.loads(file.read())
    return slang_dict


def load_additional_vocab(root_words_file, stop_words_file, json_file):
    """
    Loads additional vocabulary from two text files and a JSON file.

    Args:
        root_words_file (str): Path to the text file containing root words.
        stop_words_file (str): Path to the text file containing stop words.
        json_file (str): Path to the JSON file containing slang words.

    Returns:
        list: A list of vocabulary tokens from all files.
    """
    root_words = load_words(root_words_file)
    stop_words = load_words(stop_words_file)
    slang_dict = load_slang_words(json_file)

    # Combine all words into a unique set of tokens
    all_words = list(set(root_words + stop_words + list(slang_dict.keys()) + list(slang_dict.values())))

    return all_words