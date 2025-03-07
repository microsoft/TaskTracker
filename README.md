# TaskTracker (or, *Get my drift?*)
TaskTracker is a novel approach to detect task drift in large language models (LLMs) by analyzing their internal activations. It is based on the research described in this SaTML'25 paper [Get my drift? Catching LLM Task Drift with Activation Deltas](https://arxiv.org/abs/2406.00799). 

<p align="center">
<img src="https://github.com/microsoft/TaskTracker/blob/main/assets/teaser.png" width="700">
</p>
 
 **Key features:**

* Detects when an LLM deviates from a user's original instructions due to malicious prompts injected into external data sources
* Works across multiple state-of-the-art LLMs including Mistral 7B, Llama-3 8B,Llama-3 70B, Mixtral 8x7B, and Phi-3 3.8B
* Achieves over 0.99 ROC AUC on out-of-distribution test data spanning jailbreaks, malicious instructions, and unseen task domains
* Does not require model fine-tuning or output generation, maximizing deployability and efficiency
* Generalizes well to detect various types of task drift without being trained on specific attacks

**The repo includes:**

* A script to run the trained probes on new datasets/examples.
* Steps to recreate our exact large-scale dataset (500K+ examples), and generate a new one, for training and evaluating task drift detection
* Form to request access to the pre-computed activations
* Code to extract and analyze LLM activations
* Implementations of linear and metric learning probes for task drift classification
* Evaluation scripts and pre-trained models

TaskTracker enables more secure use of LLMs in retrieval-augmented applications by catching unwanted deviations from user instructions. It also opens up new directions for LLM interpretability and control.

## Table of Content 
- [Request Access to Activations Data](#request-data)
   - [Download Data](#download-data) 
- [Environment Setup](#env-setup)
- [Run on your data [NEW!]](#new-data)
- [Dataset construction](#dataset-construction)
- [Activation Generation](#activation-generation)
    - Supported models at the moment are: Phi-3 3.8B, Mistral 7B, Llama-3 8B, Mixtral 8x7B, Llama-3 70B
- [Training probes](#training)
    - [Linear Probes](#linear-probes)
    - [Metric Learning (Triplet) Probes](#metric-learning-probes)
- Code for reproducing the [Results and experiments](#evaluation) in the paper:
    - ROC AUC of probes on test set
    - Metric learning probes distances per different conditions in the test set
    - Metric learning probes temporal distances per tokens in the poisoned sequences
    - t-SNE visualization of raw activations and learned embeddings
    - GPT-4 verifier (judge) for attack execution
- [Citation](#citation)
----

## Request access to LLM activations
To request access to the activation data we generated for simulating/evaluating task drift, please fill out this [form](https://forms.microsoft.com/r/wXBfXQpuR2) and we will respond with a time-restricted download link (coming soon, we will send download links as soon as they are available).  

### Download Data
1. Install [azcopy](https://formulae.brew.sh/formula/azcopy). For Macbook, run `brew install azcopy`.
2. Run the following command:
     -  Replace {MODEL_NAME} and {DATA_DISTRIBUTION} with your target values. The available target values are listed below and must be transcribed exactly.
     -  Replace {SAS_TOKEN} with the provided SAS key.
     -  Update <LOCAL_PATH> with the local location you want to download the data.
```bash
azcopy copy 'https://tasktrackeropensource.blob.core.windows.net/activations/{MODEL_NAME}/{DATA_DISTRIBUTION}?{SAS_TOKEN}' <LOCAL_PATH> --recursive
```

### Target Values
#### Models
* phi__3__3.8
* mistral__7B
* mistral__7B__no_priming
* llama__3__8B
* llama__3__70B

#### Data Distributions
* training
* validation
* test

### Loading the Data 

- The shape of the activation tensor for training subset is the following: `[3, BATCH, LAYERS, DIM]`:
 - Dim 0 is stored as: Primary task, Clean, and Poisoned
 - Training data was constructed using the same text examples as pairs of clean vs poisoned ones 
   
- The shape of the activation tensor for validation subset is the following: `[2, BATCH, LAYERS, DIM]`:
  - Dim 0 is stored as: Primary task, the whole text.
  - Whether it is clean or poisoned depends on the activation files (which is included in the file names)
    
  ```python
  
  import torch
  
  # Load the activation data for validation/test 
  clean_activations = torch.load('activations/activations_0.pt')
  poisoned_activations = torch.load('activations/activations_1.pt')
  # Shape: (2, 1000, 32, 4096). For training files, this would be (3, 1000, 32, 4096)
  
  
  # Subtract the first dimension of the activations (to remove the instruction and only compare data blocks)
  clean_activations = clean_activations[1] - clean_activations[0]
  poisoned_activations = poisoned_activations[1] - poisoned_activations[0]
  # Shape: (1000, 32, 4096)
  ```
  
## Environment Setup 
1. Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate tasktracker
```

2. Install packages and setup a local instance of the TaskTracker package:

```bash
cd TaskTracker
pip install -e .
```
## New data

1- Check `quick_start` for a simple way to run on new data 

2- Edit `quick_start/config.yaml` for configurations of classifier path, which LLM, parameters of layers and thresholds, etc. 

3- Check the structure of data in `quick_start/mock_data.json`. You can prepare your data as 

```
[
 {
  "user_prompt": "primary task",
  "text": "paragraph, can be clean or poisoned",
  "label": 1 (poisoned), 0 (clean)
 }
]
```

4- run `quick_start/main_quick_test.py`. According to which LLM/Task Tracker you are using, change `torch_type` when loading the LLM (check `TaskTracker/task_tracker/config/models.py` for the precision we used for each LLM).

---

## Dataset Construction

We provide pre-sampled dataset examples for training and evaluation (see option 1 for regenerating our exact data which you probably may need to do if you are using our precomputed activations). 

### Option 1: Using Pre-sampled Dataset

1. We provide scripts to regenerate our dataset exactly (which you can verify with prompt hashes values).
2. Please run the notebooks in `task_tracker/dataset_creation/recreate_dataset` which will automatically download the relevant resources and build the dataset. No change is required. 
3. Update the dataset file paths in `task_tracker/config/models.py` to point to your created files.

### Option 2: Constructing Your Own Dataset

To create your own dataset:

1. Run the Jupyter notebooks in `task_tracker/dataset_creation/` to prepare training, validation, and test datasets.
2. Update dataset file paths in `task_tracker/config/models.py` to point to your newly generated files.

#### Dependencies

- This repository includes:
  - GPT-4 generated **triggers**
  - **Trivia** questions and answers pairs
  - **Translated** subset of trivia questions and answers
  - **Generic** NLP tasks
  - Attack prompts from [BIPIA](https://github.com/microsoft/BIPIA)
- Dataset construction scripts automatically download:
  - HotPotQA, SQuAD, Alpaca, Code Alpaca, WildChat, and other datasets from HuggingFace or their hosting websites
- Some jailbreak examples require manual download (URLs provided in corresponding notebooks)

#### Jupyter Notebooks for Dataset Creation
Note: Each notebook contains detailed instructions and customization options. Adjust parameters as needed for your specific use case.

1. `prepare_training_dataset.ipynb`: Samples training data from SQuAD training split
   - Customize with `args.orig_task`, `args.emb_task`, and `args.embed_loc`
   - See `training_dataset_combinations.ipynb` for combination examples

2. `prepare_datasets_clean_val.ipynb`: Samples clean validation data
   - Uses HotPotQA and SQuAD validation splits
   - Primary tasks: QA or Mix of QA and generic NLP prompts

3. `prepare_datasets_clean_test.ipynb`: Samples clean test data
   - Uses HotPotQA training split
   - Primary tasks: QA or Mix of QA and generic NLP prompts

4. `prepare_datasets_poisoned_val.ipynb`: Samples poisoned validation data

5. `prepare_datasets_poisoned_test.ipynb`: Samples poisoned test data

6. `prepare_datasets_poisoned_test_other_variations.ipynb`: Generates variations of poisoned injections (trigger variations)

7. `prepare_datasets_clean_test_spotlight.ipynb`: Constructs clean examples with spotlighting prompts

8. `prepare_datasets_poisoned_test_translation_WildChat.ipynb`: Constructs WildChat examples (clean examples with instructions) and poisoned examples with translated instructions

#### Post-Generation Steps

After generating or downloading the dataset:
- Update the dataset file paths in `task_tracker/config/models.py`

----

## Activation Generation

- We provide pre-computed activations for immediate use. To access them, please complete this [form](https://forms.microsoft.com/r/wXBfXQpuR2).
- Note: coming soon. We will reply with links to download once they are available. 

### Option 1: Using Pre-computed Activations

1. After receiving access, download the activation files.
2. Update the `DATA_LISTS` path in `task_tracker/training/utils/constants.py` to point to your downloaded files.

### Option 2: Generating Your Own Activations

To generate activations:

1. Configure paths in `task_tracker/config/models.py`:
```python
# HuggingFace cache directory
cache_dir = "/path/to/hf/cache/"
os.environ["TRANSFORMERS_CACHE"] = cache_dir
os.environ["HF_HOME"] = cache_dir

# Activations output directory
activation_parent_dir = "/path/to/store/activations/"

# Dataset text files directory
text_dataset_parent_dir = "/path/to/dataset/text/files/"
```

2. Customize activation generation in `task_tracker/activations/generate.py`:
```python
model_name: str = "mistral"  # Choose from models in task_tracker.config.models
with_priming: bool = True    # Set to False if no priming prompt is needed
```

3. (Optional) Modify the priming prompt in `task_tracker/utils/data.py` if needed.

4. Generate activations:
```bash
python task_tracker/activations/generate.py
```
  

#### Post-Generation Steps
After generating or downloading activations:

1. Organize activation files:
* Create lists of .pt files for training, validation (clean and test), and test (clean and poisoned) splits.
* See examples in task_tracker/data/.


2. Update the `DATA_LISTS` path in `task_tracker/training/utils/constants.py`:
```python
DATA_LISTS = "/path/to/activation/file/lists/"
```

Note: Ensure that dataset file paths in task_tracker/config/models.py are correct before generating activations.


----
## Training

We provide pre-trained probes in the repository. However, if you wish to train your own probes, follow these steps:

### Prerequisites

Ensure you have:
1. Dataset text files specified in `task_tracker/config/models.py` (from the dataset creation step)
2. Activation file lists (`DATA_LISTS`) in `task_tracker/training/utils/constants.py` (from the activation generation step)

### Configuration

1. Set the output directory for triplet probes in `task_tracker/training/utils/constants.py`:
   ```python
   MODEL_OUTPUT_DIR = '/path/to/output/directory'


### Linear Probes

1. Edit `task_tracker/training/linear_probe/train_linear_model.py`:


```python
MODEL = "llama3_70b"  # Choose from models in task_tracker.training.utils.constants
```

2. Run the training script

```bash
python task_tracker/training/linear_probe/train_linear_model.py
```

### Metric Learning Probes
1. Edit `task_tracker/training/triplet_probe/train_per_layer.py`:

```python
MODEL = 'mistral'  # Choose from models in task_tracker.training.utils.constants

config = {
    'model': MODEL,
    'activations': ACTIVATIONS_DIR,
    'activations_ood': ACTIVATIONS_VAL_DIR,
    'ood_poisoned_file': OOD_POISONED_FILE, 
    'exp_name': 'mistral_test',  # Update with your experiment name
    'margin': 0.3,
    'epochs': 6,
    'num_layers': (0,5),  # Start to end layer (both inclusive)
    'files_chunk': 10,
    'batch_size': 2500,  # Batch size for triplet mining
    'learning_rate': 0.0005,
    'restart': False,  # Set to True if restarting from a checkpoint
    'feature_dim': 275,
    'pool_first_layer': 5 if MODEL == 'llama3_70b' else 3,
    'dropout': 0.5,
    'check_each': 50,
    'conv': True,
    'layer_norm': False,
    'delay_lr_factor': 0.95,
    'delay_lr_step': 800
}
```
2. Run the training script

```bash
python task_tracker/training/triplet_probe/train_per_layer.py
```

### Post-Training Steps
After training or downloading pre-trained models:
1. Update the paths to trained model directories in `task_tracker/experiments_outputs.py`:

```python
linear_probe_out_parent_dir = "/path/to/linear/probes"
triplet_probe_out_parent_dir = "/path/to/triplet/probes"
```
Note: Adjust hyperparameters and configuration settings as needed for your specific use case.

----
## Evaluation

This section provides scripts to evaluate models and reproduce our experiments.

### Prerequisites

Ensure you have:
1. Dataset text files specified in `task_tracker/config/models.py`
2. Activation file lists (`DATA_LISTS`) in `task_tracker/training/utils/constants.py`
3. Paths to trained models in `task_tracker/experiments_outputs.py`

### Visualizing Activations

Use `task_tracker/evaluation/visualizations/tsne_raw_activations.ipynb` to visualize task activation residuals:

```python
from task_tracker.training.dataset import ActivationsDatasetDynamicPrimaryText
from task_tracker.training.utils.constants import TEST_ACTIVATIONS_DIR_PER_MODEL, TEST_CLEAN_FILES_PER_MODEL, TEST_POISONED_FILES_PER_MODEL

MODEL = 'mistral'
BATCH_SIZE = 256
TEST_ACTIVATIONS_DIR = TEST_ACTIVATIONS_DIR_PER_MODEL[MODEL]
FILES_CHUNCK = 10 
LAYERS = 80 if MODEL == 'llama3_70b' else 32
```

#### GPT-4 Verifier
To simulate attacks and verify model responses:

1. Get model responses:
```bash
python task_tracker/evaluation/verifier/get_model_responses.py
```

2. Configure in the script:
```python
from task_tracker.experiments_outputs import MODELS_RESPONSE_OUT_FILENAME_PER_MODEL
from task_tracker.config.models import data, models, cache_dir

POISONED_TEST_DATASET_FILENAME = data['test_poisoned'] 
CLEAN_TEST_DATASET_FILENAME = data['test_clean']
TOKEN = ''  # Add HF token
MODEL = 'mistral'  # Change as needed
```

2. Run the verifier:
```bash 
python task_tracker/evaluation/verifier/gpt4_judge_parallel_calls.py
```
3. Configure the script:

Note: This script uses parallel API calls. Be mindful of costs when processing large datasets.

```python
from task_tracker.experiments_outputs import MODELS_RESPONSE_OUT_FILENAME_PER_MODEL, VERIFIER_RESPONSE_OUT_FILENAME_PER_MODEL
from task_tracker.config.models import data

MODEL = 'llama3_8b'
MAX_THREADS = 60
JUDGE_PROMPT_FILE = 'judge_prompt.txt'
JUDGE_MODEL = 'gpt-4-no-filter'
AZURE_OPENAI_KEY = ''  # Add credentials
AZURE_OPENAI_ENDPOINT = ''
```


#### Evaluating Linear Probes
Use `task_tracker/evaluation/linear_probe/evaluate_linear_models.ipynb`:
```python
from task_tracker.training.utils.constants import TEST_ACTIVATIONS_DIR_PER_MODEL, TEST_CLEAN_FILES_PER_MODEL, TEST_POISONED_FILES_PER_MODEL
from task_tracker.training.dataset import ActivationsDatasetDynamicPrimaryText
from task_tracker.experiments_outputs import LINEAR_PROBES_PATHS_PER_MODEL

FILES = 'test'
MODEL = 'llama3_70b'

```


#### Evaluating Triplet Probes
1. Generate embeddings:

Use `task_tracker/evaluation/triplet_probe/evaluate_triplet_models_test_data.ipynb`:

```python
from task_tracker.training.utils.constants import TEST_ACTIVATIONS_DIR_PER_MODEL, TEST_CLEAN_FILES_PER_MODEL, TEST_POISONED_FILES_PER_MODEL
from task_tracker.experiments_outputs import TRIPLET_PROBES_PATHS_PER_MODEL

MODEL = 'llama3_70b'
```

2. Update embedding paths in `task_tracker/experiments_outputs.py`:

```python

TRIPLET_PROBES_PATHS_PER_MODEL = {
  'mistral': {
    'path': triplet_probe_out_parent_dir + '/mistral_best', 
    'num_layers': (17,31), 
    'saved_embs_clean': 'clean_embeddings_20240429-133151.json',
    'saved_embs_poisoned': 'poisoned_embeddings_20240429-134637.json'
  }
}
```
3. Analyze distances:

Use `task_tracker/evaluation/triplet_probe/distances_per_conditions.ipynb`:

```python
from task_tracker.config.models import data
from task_tracker.experiments_outputs import TRIPLET_PROBES_PATHS_PER_MODEL, VERIFIER_RESPONSE_OUT_FILENAME_PER_MODEL

POISONED_TEST_DATASET_FILENAME = data['test_poisoned'] 
CLEAN_TEST_DATASET_FILENAME = data['test_clean']
```

4. Analyze temporal distances:

Use `task_tracker/evaluation/triplet_probe/temporal_distances_per_tokens.ipynb`:

```python
from task_tracker.config.models import data, models, cache_dir
from task_tracker.experiments_outputs import TRIPLET_PROBES_PATHS_PER_MODEL

os.environ["TRANSFORMERS_CACHE"] = cache_dir
os.environ["HF_HOME"] = cache_dir

POISONED_TEST_DATASET_FILENAME = data['test_poisoned'] 
CLEAN_TEST_DATASET_FILENAME = data['test_clean']
MODEL = 'mistral'  # Change as needed
```
Note: Adjust model names and file paths as necessary for your specific setup and experiments.


## Citation 

If you find our paper, dataset, or this repo helpful, please cite our paper:

``` 
@inproceedings{abdelnabi2025getmydrift,
      title={Get my drift? Catching LLM Task Drift with Activation Deltas}, 
      author={Sahar Abdelnabi and Aideen Fay and Giovanni Cherubin and Ahmed Salem and Mario Fritz and Andrew Paverd},
      year={2025},
      booktitle={SaTML (To Appear)}
}
```
