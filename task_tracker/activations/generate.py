import torch
import json

from task_tracker.utils.model import load_model
from task_tracker.config.models import models
from task_tracker.utils.activations import process_texts_in_batches

# NOTE:
# Update the model name to the model you want to generate activations for
# Update with_priming to False if you want to generate activations without priming
model_name: str = "mistral_7B"
with_priming: bool = True
cache_dir: str = (
    "/disk1/"  # Update this to the cache directory where the model can be cached on your machine
)
model = models[model_name]

try:
    loaded_model = load_model(model.name, cache_dir=cache_dir)
    tokenizer = loaded_model["tokenizer"]
    model = loaded_model["model"]

    # Check if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")

    model.eval()

except Exception as err:
    for i in range(torch.cuda.device_count()):
        print(f"Memory summary for GPU {i}:")
        print(torch.cuda.memory_summary(device=i))
    raise err

# Gathers activations for training, test and validation data. Filter by subset if you want to
# get activations for a specific subset of the data (i.e., train, validation, or test).
for subset, data in model.data.items():
    subset = json.load(open(data, "r"))
    process_texts_in_batches(
        dataset_subset=subset[model.start_idx :],
        model_name=model.name,
        data_type=subset,
        output_dir=model.output_dir,
        with_priming=with_priming,
    )
