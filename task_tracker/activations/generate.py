import torch
import json
import logging
from task_tracker.utils.model import load_model
from task_tracker.config.models import models, cache_dir
from task_tracker.utils.activations import (
    process_texts_in_batches,
    process_texts_in_batches_pairs,
)

# NOTE: Configuration
# Update the model name to the model you want to generate activations for (from models in task_tracker.config.models)
# Update with_priming to False if you want to generate activations without priming
# Update paths in task_tracker.config.models of cache_dir (HF cache dir),
# activation_parent_dir (the output of activations), and text_dataset_parent_dir (dir of dataset text files)

model_name: str = "mistral"
with_priming: bool = True


def main():
    model = models[model_name]

    try:
        # Load the model and tokenizer
        loaded_model = load_model(
            model.name, cache_dir=cache_dir, torch_dtype=model.torch_dtype
        )
        model.tokenizer = loaded_model["tokenizer"]
        model.model = loaded_model["model"]

        # Check if multiple GPUs are available
        if torch.cuda.device_count() > 1:
            logging.info(f"Let's use {torch.cuda.device_count()} GPUs!")

        model.model.eval()

    except Exception as err:
        # Print memory summary for each GPU in case of an error
        for i in range(torch.cuda.device_count()):
            logging.info(f"Memory summary for GPU {i}:")
            logging.info(torch.cuda.memory_summary(device=i))
        raise err

    # Process data for activations
    for data_type, data in model.data.items():
        try:
            subset = json.load(open(data, "r"))

            # Determine directory and subset types based on data type
            if data_type == "train":
                directory_name = "training"
                process_texts_in_batches(
                    dataset_subset=subset[model.start_idx :]
                    model=model,
                    data_type=data_type,
                    sub_dir_name=directory_name,
                    with_priming=with_priming,
                )
            else:
                directory_name = "validation" if "val" in data_type else "test"
                subset_type = "clean" if "clean" in data_type else "poisoned"
                process_texts_in_batches_pairs(
                    dataset_subset=subset[model.start_idx :]
                    model=model,
                    data_type=subset_type,
                    sub_dir_name=directory_name,
                    with_priming=with_priming,
                )

        except json.JSONDecodeError as json_err:
            logging.error(f"Error decoding JSON for {data_type}: {json_err}")
        except Exception as data_err:
            logging.error(f"Error processing {data_type} data: {data_err}")


if __name__ == "__main__":
    main()
