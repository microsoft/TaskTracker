import logging
import os
import re
from datetime import datetime

import torch
from tqdm import tqdm

from task_tracker.models.model import Model
from task_tracker.utils.data import format_prompts

current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)


def get_last_token_activations_single(
    text, model, start_layer: int = 1, token: int = -1
):
    """
    Process a single text to extract the last token activations from all layers.

    Parameters:
    - text (str): The text to process.
    - model: The pre-trained model from Hugging Face's Transformers.
    - tokenizer: The tokenizer corresponding to the pre-trained model.
    - model_name: The name of the model

    Returns:
    - Tensor of shape (num_layers, hidden_size) containing the last token activations.
    """

    if "mistral" in model.name or "phi" in model.name:
        chat = [
            {
                "role": "user",
                "content": "you are a helpful assistant that will provide accurate answers to all questions. "
                + text,
            }
        ]
    else:
        chat = [
            {
                "role": "system",
                "content": "you are a helpful assistant that will provide accurate answers to all questions.",
            },
            {"role": "user", "content": text},
        ]

    inputs = model.tokenizer.apply_chat_template(
        chat, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )

    with torch.no_grad():
        try:
            inputs = inputs.cuda()
            outputs = model.model(inputs, output_hidden_states=True)

        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(
                    "CUDA out of memory. Printing memory status and attempting to clear cache."
                )
                for i in range(torch.cuda.device_count()):
                    print(f"Memory summary for GPU {i}:")
                    print(torch.cuda.memory_summary(device=i))
                torch.cuda.empty_cache()
            raise e

        end_layer = len(outputs["hidden_states"])
        last_tokens = []
        for i in range(start_layer, end_layer):
            # print(f'Extracting last token from layer {i}/{end_layer - 1}')
            last_tokens.append(outputs["hidden_states"][i][:, token].cpu())
        last_token_activations = torch.stack(last_tokens)

    return last_token_activations.squeeze(1)


def process_texts_in_batches(
    dataset_subset,
    model: Model,
    data_type: str,
    sub_dir_name: str,
    batch_size=1000,
    with_priming: bool = True,
):
    """
    Process texts in smaller batches and immediately write out each batch's activations.
    """
    if not os.path.exists(model.output_dir):
        os.makedirs(model.output_dir)

    output_subdir = os.path.join(model.output_dir, sub_dir_name)
    if not os.path.exists(output_subdir):
        os.makedirs(output_subdir)

    for i in tqdm(range(0, len(dataset_subset), batch_size)):

        batch_primary, batch_primary_clean, batch_primary_poisoned = format_prompts(
            dataset_subset[i : i + batch_size], with_priming
        )

        hidden_batch_primary = torch.stack(
            [get_last_token_activations_single(text, model) for text in batch_primary]
        )
        hidden_batch_primary_clean = torch.stack(
            [
                get_last_token_activations_single(text, model)
                for text in batch_primary_clean
            ]
        )
        hidden_batch_primary_poisoned = torch.stack(
            [
                get_last_token_activations_single(text, model)
                for text in batch_primary_poisoned
            ]
        )

        hidden_batch = torch.stack(
            [
                hidden_batch_primary,
                hidden_batch_primary_clean,
                hidden_batch_primary_poisoned,
            ]
        )
        # Construct file path for this batch
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filepath = os.path.join(
            output_subdir, f"{data_type}_hidden_states_{i}_{i+batch_size}_{time_str}.pt"
        )
        print(output_filepath)
        sanitized_output_filepath = re.sub(r"[^\x00-\x7F]+", "_", output_filepath)

        # Save this batch's activations to disk
        try:
            torch.save(hidden_batch, sanitized_output_filepath)
            print(f"File saved successfully to {sanitized_output_filepath}")
        except Exception as e:
            logging.error(f"Failed to save file to {sanitized_output_filepath}: {e}")
            print(f"An error occurred while saving the file: {e}")


def process_texts_in_batches_pairs(
    dataset_subset,
    model: Model,
    data_type: str,
    sub_dir_name: str,
    batch_size=1000,
    with_priming: bool = True,
):
    """
    Process texts in smaller batches and immediately write out each batch's activations.
    This is for validation data that contains primary + text
    """
    if not os.path.exists(model.output_dir):
        os.makedirs(model.output_dir)

    output_subdir = os.path.join(model.output_dir, sub_dir_name)
    if not os.path.exists(output_subdir):
        os.makedirs(output_subdir)

    for i in tqdm(range(0, len(dataset_subset), batch_size)):

        batch_primary, batch_primary_clean, batch_primary_poisoned = format_prompts(
            dataset_subset[i : i + batch_size], with_priming
        )

        hidden_batch_primary = torch.stack(
            [get_last_token_activations_single(text, model) for text in batch_primary]
        )
        if data_type == "clean":
            hidden_batch_primary_with_text = torch.stack(
                [
                    get_last_token_activations_single(text, model)
                    for text in batch_primary_clean
                ]
            )
        elif data_type == "poisoned":
            hidden_batch_primary_with_text = torch.stack(
                [
                    get_last_token_activations_single(text, model)
                    for text in batch_primary_poisoned
                ]
            )

        hidden_batch = torch.stack(
            [hidden_batch_primary, hidden_batch_primary_with_text]
        )
        # Construct file path for this batch
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filepath = os.path.join(
            output_subdir, f"{data_type}_hidden_states_{i}_{i+batch_size}_{time_str}.pt"
        )
        print(output_filepath)
        sanitized_output_filepath = re.sub(r"[^\x00-\x7F]+", "_", output_filepath)

        # Save this batch's activations to disk
        try:
            torch.save(hidden_batch, sanitized_output_filepath)
            print(f"File saved successfully to {sanitized_output_filepath}")
        except Exception as e:
            logging.error(f"Failed to save file to {sanitized_output_filepath}: {e}")
            print(f"An error occurred while saving the file: {e}")
