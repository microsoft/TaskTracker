import os
from typing import Dict, List
from task_tracker.models.model import Model

os.environ["TRANSFORMERS_CACHE"] = "/disk1/"
os.environ["HF_HOME"] = "/disk1/"

data = {
    "train": "./dataset_sampling/dataset_sampled_examples/train_subset.json",
    "val": "./dataset_sampling/dataset_sampled_examples/val_subset.json",
    "test": "./dataset_sampling/dataset_sampled_examples/test_subset.json",
}

llama_3_70B = Model(
    name="meta-llama/Meta-Llama-3-70B",
    output_dir="/disk1/activations/llama_3_70B",
    data=data,
    subset="train",
)

llama_3_8B = Model(
    name="meta-llama/Meta-Llama-3-8B",
    output_dir="/disk1/activations/llama_3_8B",
    data=data,
    subset="train",
)

mistral_7B = Model(
    name="mistralai/Mistral-7B-Instruct-v0.2",
    output_dir="/disk1/activations/mistral_7B",
    data=data,
    subset="train",
)

phi3 = Model(
    name="microsoft/Phi-3-mini-4k-instruct",
    output_dir="/disk1/activations/phi3",
    data=data,
    subset="train",
)

mixtral = Model(
    name="mistralai/Mixtral-8x7B-Instruct-v0.1",
    output_dir="/disk1/activations/mixtral",
    data=data,
    subset="train",
)

models: Dict[str, Model] = {
    "llama_3_70B": llama_3_70B,
    "llama_3_8B": llama_3_8B,
    "mistral_7B": mistral_7B,
    "phi3": phi3,
    "mixtral": mixtral,
}
