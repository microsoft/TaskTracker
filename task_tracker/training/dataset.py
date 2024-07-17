import torch
import os
import json
from tqdm import tqdm
from torch.utils.data import Dataset
import random
from typing import List, Union, Tuple

HIDDEN_STATES_DIR = '/share/data_instruct_sep/get_activations/mistral'

class ActivationsDatasetDynamic(Dataset):
    """
    A dataset class designed to dynamically load and return slices of neural network activations
    stored in files.

    The class is initialized with a list of file paths where activations are stored, a root directory
    for these files, and num_layers, which is either "int", in this case it would return the last n layers, 
    or a tuple, in this case it would return the range of layers from start to end (inclusive).
    
    Attributes:
        root_dir (str): The directory containing activation files.
        dataset_files (List[str]): A list of filenames (strings) within root_dir that contain the activations to be loaded.
        activations (Tensor): The loaded activations concatenated across all specified files.
        num_layers (int or Tuple): which activation layers to load 

    Methods:
        load_activations(num_layers): Loads the specified activation layers from each file in dataset_files.
        __len__(): Returns the number of samples in the dataset, determined by the second dimension of the activations tensor.
        __getitem__(idx): Returns the activations for a given index, supporting the retrieval of primary, clean, and poisoned data slices.
    """

    def __init__(self, dataset_files: List[str], root_dir: str, num_layers: Union[int, Tuple[int, int]]):
        """
        Initializes the dataset object with file paths, root directory, and the number of layers to load.

        Parameters:
            dataset_files (List[str]): List of paths to the saved activation files.
            root_dir (str): Directory containing the activation files.
            num_layers (int or Tuple): activation layers to load.
        """
        self.root_dir = root_dir
        self.dataset_files = dataset_files
        self.activations = self.load_activations(num_layers)

    def load_activations(self, num_layers: Union[int, Tuple[int, int]]):
        """
        Loads neural network activations from stored files, optionally selecting only a specified
        number of layers from each activation tensor.

        Parameters:
            num_layers (int or tuple): Specifies which layers to load from each file.

        Returns:
            torch.Tensor: A single tensor containing all requested activations concatenated along the
                        first dimension.

        Note:
            The method assumes that the activation files are stored in a consistent format, where the
            first dimension corresponds to different data types (e.g., primary, clean, poisoned), the
            second dimension indexes individual examples, the third dimension corresponds to the
            network's layers, and the fourth dimension contains the activation values from each layer.
            """
        activations = []

        for file_ in tqdm(self.dataset_files):
            file_name = os.path.join(self.root_dir, file_)

            # Load the activation tensor from the current file.
            activation_file = torch.load(file_name)

            # Slice the activation tensor to select only the requested layers.
            # The slice [0:3, :, num_layers:, :] means:
            # - Select all data types (0:3),
            # - Select all examples (:),
            # - Select the last `num_layers` layers from each example (-num_layers:),
            # - Include all activation values from the selected layers (:).

            if isinstance(num_layers, int):
                activation = activation_file[0:3, :, -num_layers:, :]

            elif isinstance(num_layers, tuple):
                activation = activation_file[0:3, :, num_layers[0] : num_layers[1] + 1, :]

            # Append the sliced activation tensor to the list.
            activations.append(activation)

        # Concatenate all activation tensors in the list along the first dimension.
        return torch.cat(activations, dim=1)

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            The size of the second dimension of the activations tensor, representing the number of samples.
        """
        return self.activations.size(1)

    def __getitem__(self, idx):
        """
        Retrieves the activations for a given index, returning primary, clean, and poisoned slices.

        Parameters:
            idx (int or tensor): Index of the sample to retrieve.

        Returns:
            A tuple containing the primary, clean, and poisoned activation tensors for the specified index.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        primary = self.activations[0, idx, :]
        clean = self.activations[1, idx, :]
        poisoned = self.activations[2, idx, :]

        return primary, clean, poisoned

class ActivationsDatasetDynamicReturnText(Dataset):
    """
    A dataset class designed to dynamically load and return slices of neural network activations
    stored in files. It also returns the text of the items. 

    The class is initialized with a list of file paths where activations are stored, a root directory
    for these files, and num_layers, which is either "int", in this case it would return the last n layers, 
    or a tuple, in this case it would return the range of layers from start to end (inclusive).
    
    Attributes:
        root_dir (str): The directory containing activation files.
        dataset_files (List[str]): A list of filenames (strings) within root_dir that contain the activations to be loaded.
        activations (Tensor): The loaded activations concatenated across all specified files.
        num_layers (int or Tuple): which activation layers to load 

    Methods:
        load_activations(num_layers): Loads the specified activation layers from each file in dataset_files.
        __len__(): Returns the number of samples in the dataset, determined by the second dimension of the activations tensor.
        __getitem__(idx): Returns the activations for a given index, supporting the retrieval of primary, clean, and poisoned data slices.
    """

    def __init__(self, dataset_files: List[str], root_dir: str, data_dir: str, num_layers: Union[int, Tuple[int, int]]):
        """
        Initializes the dataset object with file paths, root directory, and the number of layers to load.

        Parameters:
            dataset_files (List[str]): List of paths to the saved activation files.
            root_dir (str): Directory containing the activation files.
            data_dir (str): Directory containing the text files of the dataset (as json).
            num_layers (int or Tuple): activation layers to load.        
            
        """
        self.root_dir = root_dir
        self.dataset_files = dataset_files
        self.activations = self.load_activations(num_layers)
        self.data_dir = data_dir
        self.dataset_primary_clean_text = self.get_corresponding_primary_clean()

    def load_activations(self, num_layers: Union[int, Tuple[int, int]]):
        """
        Loads neural network activations from stored files, optionally selecting only a specified
        number of layers from each activation tensor.

        Parameters:
            num_layers (int or tuple): Specifies which layers to load from each file.

        Returns:
            torch.Tensor: A single tensor containing all requested activations concatenated along the
                        first dimension.

        Note:
            The method assumes that the activation files are stored in a consistent format, where the
            first dimension corresponds to different data types (e.g., primary, clean, poisoned), the
            second dimension indexes individual examples, the third dimension corresponds to the
            network's layers, and the fourth dimension contains the activation values from each layer.
        """
        activations = []

        for file_ in self.dataset_files:
            file_name = os.path.join(self.root_dir, file_)

            # Load the activation tensor from the current file.
            activation_file = torch.load(file_name)

            # Slice the activation tensor to select only the requested layers.
            # The slice [0:3, :, num_layers:, :] means:
            # - Select all data types (0:3),
            # - Select all examples (:),
            # - Select the last `num_layers` layers from each example (-num_layers:),
            # - Include all activation values from the selected layers (:).

            if isinstance(num_layers, int):
                activation = activation_file[0:3, :, -num_layers:, :]
            if isinstance(num_layers, tuple):
                activation = activation_file[0:3, :, num_layers[0] : num_layers[1] + 1, :]

            # Append the sliced activation tensor to the list.
            activations.append(activation)

        # Concatenate all activation tensors in the list along the first dimension.
        return torch.cat(activations, dim=1)

    def get_corresponding_primary_clean(self):
        """
        Loads the corresponding text files of the dataset as concatentation of primary_task_prompt and orig_text
        
        __getitem__ will now select the corresponding text items given the index.
        
        This is used to make filtering during training to remove duplicated primary + text samples in each mining batch.
        
        This is to remedy that there are duplicate examples in the dataset. 
        
        Can be safely ignored if there are no duplicates. 
        

        Returns:
            dataset_text: list of dataset text items. 

        """
        
        dataset_text = []
        dataset_files = {'train': json.load(open(os.path.join(self.data_dir,'train_subset.json')))
                        }
        for file in self.dataset_files:
            subset_name = file.split('_')[0]
            start_index = int(file.split('_')[3])
            end_index = int(file.split('_')[4])
            subset = dataset_files[subset_name][start_index:end_index]
            for item in subset:
                text_primary_clean = item['primary_task_prompt'] + ' ' + item['orig_text']
                dataset_text.append(text_primary_clean)
        return dataset_text

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            The size of the second dimension of the activations tensor, representing the number of samples.
        """
        return self.activations.size(1)

    def __getitem__(self, idx):
        """
        Retrieves the activations for a given index, returning primary, clean, and poisoned slices.

        Parameters:
            idx (int or tensor): Index of the sample to retrieve.

        Returns:
            A tuple containing the primary, clean, and poisoned activation tensors for the specified index.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        primary = self.activations[0, idx, :]
        clean = self.activations[1, idx, :]
        poisoned = self.activations[2, idx, :]
        primary_clean_text = self.dataset_primary_clean_text[idx]

        return primary, clean, poisoned, primary_clean_text

class ActivationsDatasetDynamicPrimaryText(Dataset):
    
    """
    This class is similar to previous ones. However, it assumes that the data is stored as pairs of (primary, text) instead of triplets of (primary, clean, poisoned)
    
    This is mainly for testing on validation and test data. 
    """

    def __init__(
        self,
        dataset_files: List[str],
        num_layers: Union[int, Tuple[int, int]],
        root_dir: str = HIDDEN_STATES_DIR,
    ):
        """
        Arguments:
            dataset_files (list): list of files of saved injection pt file
            root_dir (string): Directory with all the images.
        """
        self.root_dir = root_dir
        self.dataset_files = dataset_files
        self.num_layers = num_layers
        self.activations = self.load_activations()
        return

    def load_activations(self):
        activations = []
        for file_ in self.dataset_files:
            curr_file = torch.load(os.path.join(self.root_dir,file_))

            if isinstance(self.num_layers, int):
                activation = curr_file[:, :, -self.num_layers:, :]

            elif isinstance(self.num_layers, tuple):
                activation = curr_file[:, :, self.num_layers[0] : self.num_layers[1] + 1, :]

            activations.append(activation)

        return torch.cat(activations, dim=1)

    def __len__(self):
        return self.activations.size(1)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        activations_primary = self.activations[0,idx,:]
        activations_primary_with_text = self.activations[1,idx,:]

        return activations_primary, activations_primary_with_text


class LayerwiseDataLoader(Dataset):
    """
    A dataset class designed to dynamically load and return slices of neural network activations
    stored in files.

    The class is initialized with a list of file paths where activations are stored, a root directory
    for these files, and an option to select how many layers of activations to load, starting from
    the last layer and moving backwards. By default, all layers are loaded.

    Attributes:
        root_dir (str): The directory containing activation files.
        dataset_files (List[str]): A list of filenames (strings) within root_dir that contain the activations to be loaded.
        activations (Tensor): The loaded activations concatenated across all specified files.
        num_layers (int): The number of activation layers to load from each file. A value of 0 indicates that all layers should be loaded.
    """

    def __init__(self, dataset_files: List[str], root_dir: str = HIDDEN_STATES_DIR,  num_layers: int = 0, layers_to_mask : Union[int, List, Tuple[int, int]] = None):
        """
        Arguments:
            dataset_files (list): list of files of saved injection pt file
            root_dir (string): Directory with all the images.
        """
        self.root_dir = root_dir
        self.dataset_files = dataset_files
        self.layers_to_mask = layers_to_mask
        self.num_layers = num_layers
        self.activations = self.load_activations()
        return

    def load_activations(self):
        """
        Loads neural network activations from stored files, optionally selecting only a specified
        number of layers from each activation tensor.

        Returns:
            torch.Tensor: A single tensor containing all requested activations concatenated along the
                        first dimension.

        Note:
            The method assumes that the activation files are stored in a consistent format, where the
            first dimension corresponds to different data types (e.g., primary, clean, poisoned), the
            second dimension indexes individual examples, the third dimension corresponds to the
            network's layers, and the fourth dimension contains the activation values from each layer.
            """
        activations = []
        for file_ in self.dataset_files:
            file_name = os.path.join(self.root_dir, file_)
            activation_file = torch.load(file_name)
            activation_file = activation_file[:, :, -self.num_layers:, :]

            mask = torch.ones_like(activation_file, dtype=torch.bool)

            if self.layers_to_mask:

                if isinstance(self.layers_to_mask, int):
                    # Translate layer number to it's index (-1)
                    mask[:, :, self.layers_to_mask - 1, :] = 0
                elif isinstance(self.layers_to_mask, list):
                    for layer in self.layers_to_mask:
                        # Translate layer number to it's index (-1)
                        mask[:, :, layer - 1, :] = 0
                elif isinstance(self.layers_to_mask, tuple) and len(self.layers_to_mask) == 2:
                    start_layer_idx = self.layers_to_mask[0] - 1
                    end_layer_idx = self.layers_to_mask[1] # Don't -1 as we we want slice to be inclusive (the -1 and + 1 cancel out).
                    mask[:, :, start_layer_idx : end_layer_idx, :] = 0
                else:
                    raise ValueError("Invalid layers parameter. Must be an int, a list of ints, or a tuple of two ints.")

            activation = activation_file * mask
            activations.append(activation)

        return torch.cat(activations, dim=1)

    def __len__(self):
        return self.activations.size(1)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        activations_primary = self.activations[0,idx,:]
        activations_primary_with_text = self.activations[1,idx,:]

        return activations_primary, activations_primary_with_text
