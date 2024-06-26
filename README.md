# TaskTracker (or, *Are you still on track?!*)

This repo contains the code for the paper [Are you still on track!? Catching LLM Task Drift with Activations](https://arxiv.org/abs/2406.00799)

Authors: Sahar Abdelnabi* and Aideen Fay* (joint first-author), Giovanni Cherubin, Ahmed Salem, Mario Fritz, and Andrew Paverd. 

----

## Content 
This repo contains: 
- **Dataset construction** code
- Code for **collecting activations**
    - Supported models at the moment are: Phi-3 3.8B, Mistral 7B, Llama-3 8B, Mixtral 8x7B, Llama-3 70B
- Code for **training probes**
    - Linear probes
    - Metric learning probes
- Code for reproducing the **results and experiments** in the paper:
    - ROC AUC of probes on test set
    - Metric learning probes distances per different conditions in the test set
    - Metric learning probes temporal distances per tokens in the poisoned sequences
    - t-SNE visualization of raw activations and learned embeddings
    - GPT-4 verifier (judge) for attack execution
- **Trained probes**
  - Linear probes
  - Triplet probes 
  
----

## Setup 

- `conda env create -f environment.yml`
- `cd TaskTracker`
- `pip install -e .`

---- 

## Dataset Construction 

- To access the already sampled dataset examples that we used for training and evaluation, please check: **TODO**

### Dependencies
- This repo contains the GPT-4 generated **triggers**, **trivia** questions and answers pairs, a **translated** subset of trivia questions and answers, **generic** NLP tasks, and attack prompts sourced from [BIPIA](https://github.com/microsoft/BIPIA) 
- The dataset construction scripts downloads used datasets (HotPotQA, SQuAD, Alpaca, Code Alpaca, WildChat, and others) from HuggingFace and/or hosting websites
- Some jailbreaks examples can be downloaded manually (URLs are provided in the corresponding notebooks for these cases)

### How to (skip if our dataset is directly used)
- `task_tracker/dataset_creation/prepare_training_dataset.ipynb` code used to sample training data:
  - Based on training split of `SQuAD` 
  - The combinations of primary tasks and secondary tasks can be changed via changing `args.orig_task` and `args.emb_task`
  - The location of secondary tasks can be changed via `args.embed_loc`
  - `task_tracker/dataset_creation/training_dataset_combinations.ipynb` shows an example of the different combinations of these arguments used to construct the training data
    
- `task_tracker/dataset_creation/prepare_datasets_clean_val.ipynb` code used to sample clean val data:
  - Based on validation split of `HotPotQA` and `SQuAD`
  - Primary tasks are either `QA` or a `Mix` of QA and generic NLP prompts

- `task_tracker/dataset_creation/prepare_datasets_clean_val.ipynb` code used to sample clean test data:
  - Based on training split of `HotPotQA`
  - Primary tasks are either `QA` or a `Mix` of QA and generic NLP prompts

- `task_tracker/dataset_creation/prepare_datasets_poisoned_val.ipynb` code used to sample poisoned val data
    
- `task_tracker/dataset_creation/prepare_datasets_poisoned_test.ipynb` code used to sample poisoned test data

- `task_tracker/dataset_creation/prepare_datasets_poisoned_test_other_variations.ipynb` code used to generate other variations of poisoned injections (variations of the trigger)

- `task_tracker/dataset_creation/prepare_datasets_clean_test_spotlight.ipynb` code used to construct clean examples with spotlighting prompts 

- `task_tracker/dataset_creation/prepare_datasets_poisoned_test_translation_WildChat.ipynb` code used to construct examples from the WildChat dataset (clean examples containing instructions) and poisoned examples with translated instructions

### After generation/downloading 
- Edit the dataset files path in `task_tracker/config/models.py`
  
----
## Activation Generation 
- To access the already computed activations, please check: **TODO**

### Dependencies
- To generate activations for the dataset, the dataset files in `task_tracker/config/models.py` will be used (changed in dataset construction step) 
- Edit the activations directory path in `task_tracker/config/models.py` before generating the activations (or after downloading if the generation step is skipped) 
- Edit the HF cache directory path in `task_tracker/config/models.py`
  
### How to (skip if downloading)
- `task_tracker/utils/data.py` contains the priming prompt we used to generate the activations, change if required to test other prompting setups
- To generate activations for a particular model specify the model required in `task_tracker/activations/generate.py` (from the set of models in `task_tracker/config/models.py`)
- If no priming prompt is needed, set `with_priming` to `False`
- Run `python generate.py`

### After generation 
- **(skip if downloading)** The training and evaluation scripts expect a list of `.pt` files of training, val clean, val test, test clean, and test splits. Please refer to files in `task_tracker/data/` for examples
- For **both generating and downloading options**, change the path of activations files lists (`DATA_LISTS`) in `task_tracker/training/utils/constants.py` (these are currently in `task_tracker/data`)
  
----
## Training 
- The trained probes are provided in the repo
- To train your own probes please follow the following steps 

### Dependencies
- The training scripts need:
  - The dataset text files specified in `task_tracker/config/models.py` (in dataset creation step)
  - The activations files lists (`DATA_LISTS`) in `task_tracker/training/utils/constants.py` (in the activations step)
- **If training**: Change the path of the output directory (`MODEL_OUTPUT_DIR`) of triplet probes in `task_tracker/training/utils/constants.py`

### How to
- To train linear probes:
  - Specify the model required in `task_tracker/training/linear_probe/train_linear_model.py` from the set of models in (`task_tracker/training/utils/constants.py`)
  - run `python train_linear_model.py`
    
- To train metric learning (triplet) probes:
  - Specify the model required in `task_tracker/training/triplet_probe/train_per_layer.py` from the set of models in (`task_tracker/training/utils/constants.py`)
  - Change output sub-directory name of the experiment
  - Change other hyperparams if required 
  - run `python train_per_layer.py`
  
### After training/downloading models 
- Change file paths in of trained models directories in `task_tracker/experiments_outputs.py`

----
## Evaluation 
- Scripts to evaluate models and reproduce our experiments 

## Dependencies 
- Evaluation scripts need:
  - The dataset text files specified in `task_tracker/config/models.py` (in dataset creation step)
  - The activations files lists (`DATA_LISTS`) in `task_tracker/training/utils/constants.py` (in the activations step)
  - The paths of trained models in `task_tracker/experiments_outputs.py` (in the training step)
## How to
- **visualize** activations:
  - `task_tracker/evaluation/visualizations/tsne_raw_activations.ipynb` contains the script to visualize the task activations residual for raw activations. Change the model if required.
    
- GPT-4 **verifier**:
  - If required, we provide scripts to get the response of models to the attack prompts to simulate attacks.
    - run `task_tracker/evaluation/verifier/get_model_responses.py` to get models' responses. Change the model if required.
  - After getting responses:
    - run the verifier via `task_tracker/evaluation/verifier/gpt4_judge_parallel_calls.py`. Change the model and paths if required. **Important:** this script parallelize the calls to the API. Watch out for the cost before proceeding with large datasets.
    - You will need Azure OpenAI credentials.
  - The results of responses and verifier of our experiments are provided in this repo under `task_tracker/dataset_creation/dataset_sampled`      
- **Linear** probe:
  - `task_tracker/evaluation/linear_probe/evaluate_linear_models.ipynb` runs the linear probes on test data. Change the model if required.
- **Triplet** probe:
  - Prediction on test data:   
    - `task_tracker/evaluation/triplet_probe/evaluate_triplet_models_test_data.ipynb` runs the triplet probe for the test data. Change the model if required. This will save the embeddings of examples in the model output dirs. **Next**, update the saved embeddings paths in `task_tracker/experiments_outputs.py`
  - Distances per conditions:
      - `task_tracker/evaluation/triplet_probe/distances_per_conditions.ipynb` loads the saved embeddings generated in the previous step and calculates distances per different conditions in the dataset. Also, it uses the output of the verifier.
  - Distances per tokens:
    - `task_tracker/evaluation/triplet_probe/temporal_distances_per_tokens.ipynb` loads the test data text files and computes distances for a subset of examples per tokens to track distances before and after injections. Change the model if required     
    
----
  
## Citation 

- If you find our paper, dataset, or this repo helpful, please cite our paper:

``` 
@article{taskdrift2024,
  title={Are you still on track!? Catching LLM Task Drift with Activations},
  author={Sahar Abdelnabi and Aideen Fay and Giovanni Cherubin and Ahmed Salem and Mario Fritz and Andrew Paverd},
  journal={arXiv preprint arXiv:2406.00799},
  year={2024}
}
```
