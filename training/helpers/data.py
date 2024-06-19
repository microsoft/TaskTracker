import os
from typing import List
import torch
import random
import numpy as np
import json

def get_file_paths(dir_path: str, prefix: str) -> List[str]:
    """
    Retrieves a list of file paths from the specified directory that begin with the given prefix.
    
    Args:
        dir_path (str): The path to the directory from which to retrieve file paths.
        prefix (str): The prefix used to filter files.
        
    Returns:
        List[str]: A list of file paths that match the given prefix.
    """
    # Initialize an empty list to store the paths of files that match the prefix
    file_paths = []
    
    # List all files in the given directory
    for file_name in os.listdir(dir_path):
        # Check if the file name starts with the given prefix
        if file_name.startswith(prefix):
            # Construct the full file path and add it to the list
            full_path = os.path.join(dir_path, file_name)
            file_paths.append(full_path)
            
    return file_paths

# Load file paths
def load_file_paths(file_path):
    with open(file_path, "r") as file:
        return [line.strip() for line in file]

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Exclude data items where no trigger sentences exist
def process_val_data(data_poisoned_file_path, distances_poisoned): 
    data_poisoned = json.load(open(data_poisoned_file_path))
    distances_poisoned_subset  = []
    
    for i,item in enumerate(data_poisoned):
        if item['embed_method'] == 'none': continue 
        distances_poisoned_subset.append(distances_poisoned[i])
    return distances_poisoned_subset 
