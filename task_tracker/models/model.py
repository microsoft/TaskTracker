from typing import Any, Dict, Optional

import torch


class Model:
    """
    This class represents a Model from which activations will be obtained.

    Attributes:
        model_name (str): The name of the model to be retrived from Hugging Face.
        output_dir (str): The directory where the output will be stored.
        data (Dict[str, str]): A dictionary containing the paths to the training, validation, and testing data.
        subset (str): The subset of data to be used. It can be 'train', 'validation', or 'test'.
        start_idx (int, optional): The starting index for the data. Defaults to 0.
        start_layer (int, optional): The starting layer for the model. Defaults to 0.
        token (int, optional): The position of the token for extracting the activation value. Defaults to -1
        which represents the last token activation in the residual stream.
        torch_dtype (torch.dtype, optional): FP precision of loading the model. Defaults to torch.float32.
        tokenizer (Optional[Any]): The tokenizer for the model. Defaults to None.
        model (Optional[Any]): The model object. Defaults to None.

    Methods:
        __init__(self, name: str, output_dir: str, data: Dict[str, str], subset: str, start_idx: int = 0, start_layer: int = 0, token: int = -1, tokenizer: Optional[Any] = None, model: Optional[Any] = None): Initializes the Model object.
    """

    def __init__(
        self,
        name: str,
        output_dir: str,
        data: Dict[str, str],
        subset: str,
        start_idx: int = 0,
        start_layer: int = 0,
        token: int = -1,
        torch_dtype: torch.dtype = torch.float32,
        tokenizer: Optional[Any] = None,
        model: Optional[Any] = None,
    ):
        self.name = name
        self.output_dir = output_dir
        self.token = token
        self.data = data
        self.subset = subset
        self.start_idx = start_idx
        self.start_layer = start_layer
        self.torch_dtype = torch_dtype
        self.tokenizer = tokenizer
        self.model = model
