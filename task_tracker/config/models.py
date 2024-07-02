import os
from typing import Dict, List
from task_tracker.models.model import Model
import torch 

# update paths to HF cache dir 
cache_dir = "/disk1/"
os.environ["TRANSFORMERS_CACHE"] = cache_dir
os.environ["HF_HOME"] = cache_dir

# update paths to where the activations would be stored 
activation_parent_dir = "/disk1/activations"

# update paths to where the dataset text files are stored 
text_dataset_parent_dir = "/share/projects/jailbreak-activations/dataset_sampling/dataset_sampled"

data = {
    "train": text_dataset_parent_dir + "/train_subset.json",
    "val_clean": text_dataset_parent_dir + "/dataset_out_clean.json",
    "val_poisoned" : text_dataset_parent_dir + "/dataset_out_poisoned_v1.json",
    "test_clean": text_dataset_parent_dir + "/dataset_out_clean_v2.json",
    "test_poisoned": text_dataset_parent_dir + "/dataset_out_poisoned_v2.json",
}
llama_3_70B = Model(
    name="meta-llama/Meta-Llama-3-70B",
    output_dir= activation_parent_dir + "/llama3_70b",
    data=data,
    subset="train",
    torch_dtype = torch.bfloat16
)

llama_3_8B = Model(
    name="meta-llama/Meta-Llama-3-8B",
    output_dir= activation_parent_dir +  "/llama3_8b",
    data=data,
    subset="train",
    torch_dtype = torch.float32
)

mistral_7B = Model(
    name="mistralai/Mistral-7B-Instruct-v0.2",
    output_dir= activation_parent_dir + "/mistral_test",
    data=data,
    subset="train",
    torch_dtype = torch.float32
)

phi3 = Model(
    name="microsoft/Phi-3-mini-4k-instruct",
    output_dir= activation_parent_dir + "/phi3",
    data=data,
    subset="train",
    torch_dtype = torch.bfloat16
)

mixtral = Model(
    name="mistralai/Mixtral-8x7B-Instruct-v0.1",
    output_dir= activation_parent_dir + "/mixtral",
    data=data,
    subset="train",
    torch_dtype = torch.float16
)

models: Dict[str, Model] = {
    "llama3_70b": llama_3_70B,
    "llama3_8b": llama_3_8B,
    "mistral": mistral_7B,
    "phi3": phi3,
    "mixtral": mixtral,
}

