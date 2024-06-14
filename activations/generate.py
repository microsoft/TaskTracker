import torch
import json

from utils.model import load_model
from config.models import llama_3_70B_Instruct
from utils.activations import process_texts_in_batches

model = llama_3_70B_Instruct
with_priming: bool = True

try:
    loaded_model = load_model(model.name)
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

subset = json.load(open(model.data[model.subset], "r"))
process_texts_in_batches(
    dataset_subset=subset[model.start_idx :],
    model_name=model.name,
    data_type=model.subset,
    output_dir=model.output_dir,
    with_priming=with_priming,
)
