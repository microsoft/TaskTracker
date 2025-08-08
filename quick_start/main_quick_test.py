import json

import torch
from sklearn.metrics import roc_auc_score
from utils import load_config, load_task_tracker, setup_hf_llm, task_tracker_main

# Load config file
"""
REPLACE with your path and configurations 
"""
config = load_config()
config["task_tracker_model"] = config["task_tracker_model"].format(
    config["task_tracker_layer"]
)
task_tracker_layer = config["task_tracker_layer"]
llm_name = config["open_llm_name"]
task_tracker_threshold = config["task_tracker_threshold"]
data_path = config["data_path"]


# Load models
"""
You may need to change torch_type according to how each probe was trained 
Check: TaskTracker/task_tracker/config/models.py for how we used each model 
"""
llm_model, llm_tokenizer = setup_hf_llm(
    config["open_llm_name"], cache_dir=config["hf_dir"], torch_type=torch.bfloat16
)
task_tracker_classifier = load_task_tracker(config["task_tracker_model"])


##Load data
"""
REPLACE with your own data 

Assumption about data:
    - "data_path" is a json file 
    - Each item has:
        - user_prompt: initial primary task (can be empty string "")
        - text: external text that can be clean or poisoned
        - label: clean (0) or poisoned (1)
"""
data = json.load(open(data_path))


# Run task tracker
labels = []
all_probs = []
correct = 0
for item in data:
    """
    If the "user_prompt" is empty. A generic user prompt ("Summarize the following text") is going to be used
    """
    probs = task_tracker_main(
        item["text"],
        llm_model,
        llm_name,
        llm_tokenizer,
        task_tracker_classifier,
        task_tracker_layer,
        specific_user_prompt=item["user_prompt"],
    )[0]
    labels.append(item["label"])
    all_probs.append(probs)
    predicted_label = 1 if (probs > task_tracker_threshold) else 0
    if predicted_label == item["label"]:
        correct += 1

# Accuracy
print(f"Accuracy on data: {correct/len(data)}")

# ROC AUC
roc_auc_score_value = roc_auc_score(labels, all_probs)
print(f"ROC AUC: {roc_auc_score_value}")
