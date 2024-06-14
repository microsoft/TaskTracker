import os
from typing import Dict, List
from models.model import Model

os.environ["TRANSFORMERS_CACHE"] = "/disk1/"
os.environ["HF_HOME"] = "/disk1/"

llama_3_70B = Model(
    name="meta-llama/Meta-Llama-3-70B",
    output_dir="/disk1/activations/llama_3_70B",
    data={
        "train": "/share/projects/jailbreak-activations/dataset_sampling/dataset_sampled_examples/train_subset.json",
        "val": "/share/projects/jailbreak-activations/dataset_sampling/dataset_sampled_examples/val_subset.json",
        "test": "/share/projects/jailbreak-activations/dataset_sampling/dataset_sampled_examples/test_subset.json",
    },
    subset="train",
)

llama_3_8B = Model(
    name="meta-llama/Meta-Llama-3-8B",
    output_dir="/disk1/activations/llama_3_8B",
    data={
        "train": "/share/projects/jailbreak-activations/dataset_sampling/dataset_sampled_examples/train_subset.json",
        "val": "/share/projects/jailbreak-activations/dataset_sampling/dataset_sampled_examples/val_subset.json",
        "test": "/share/projects/jailbreak-activations/dataset_sampling/dataset_sampled_examples/test_subset.json",
    },
    subset="train",
)
