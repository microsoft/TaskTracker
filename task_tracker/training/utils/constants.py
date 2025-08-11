import os

from task_tracker.config.models import activation_parent_dir, text_dataset_parent_dir

# == Experiment-wide constants ==


# Directory contain text files of activations lists
DATA_LISTS = "/home/saabdelnabi/TaskTracker/task_tracker/data"
# The output directory to store output models per epoch and the best running model
MODEL_OUTPUT_DIR = "/home/saabdelnabi/TaskTracker/task_tracker/output_dir"
# Directory of the text of sampled training examples
TRAINING_TEXT_DIR = text_dataset_parent_dir
# OOD poisoned file (validation)
OOD_POISONED_FILE = text_dataset_parent_dir + "/dataset_out_poisoned_v1.json"


# Constants for experiments for each model

CONSTANTS_ALL_MODELS = {
    "phi3": {
        # where the List of activations files to pull are stored
        "ACTIVATION_FILE_LIST_DIR": DATA_LISTS,
        # Activations training data
        "ACTIVATIONS_DIR": activation_parent_dir + "/phi3/training",
        # Activations validation data (OOD)
        "ACTIVATIONS_VAL_DIR": activation_parent_dir + "/phi3/validation",
    },
    "mixtral": {
        "ACTIVATION_FILE_LIST_DIR": DATA_LISTS,
        "ACTIVATIONS_DIR": activation_parent_dir + "mixtral_8x7B_instruct_float16/",
        "ACTIVATIONS_VAL_DIR": activation_parent_dir
        + "/mixtral_8x7B_instruct_float16/validation/",
    },
    "mistral": {
        # == With Priming ==
        "ACTIVATION_FILE_LIST_DIR": DATA_LISTS,
        "ACTIVATIONS_DIR": activation_parent_dir + "/mistral_7B/training",
        "ACTIVATIONS_VAL_DIR": activation_parent_dir + "/mistral_7B/validation",
    },
    "llama3_8b": {
        "ACTIVATION_FILE_LIST_DIR": DATA_LISTS,
        "ACTIVATIONS_DIR": activation_parent_dir + "/llama3_8b/training",
        "ACTIVATIONS_VAL_DIR": activation_parent_dir + "/llama3_8b/validation/",
    },
    "llama3_70b": {
        "ACTIVATION_FILE_LIST_DIR": DATA_LISTS,
        "ACTIVATIONS_DIR": activation_parent_dir + "/llama_3_70B_Instruct/training",
        "ACTIVATIONS_VAL_DIR": activation_parent_dir
        + "/llama_3_70B_Instruct/validation",
    },
    "mistral_no_priming": {
        # == Without priming ==
        "ACTIVATION_FILE_LIST_DIR": DATA_LISTS + "/no_priming",
        "ACTIVATIONS_DIR": activation_parent_dir + "/mistral_no_priming/training",
        "ACTIVATIONS_VAL_DIR": activation_parent_dir + "/mistral_no_priming/test",
    },
}

# Directory of activations of test data
TEST_ACTIVATIONS_DIR_PER_MODEL = {
    "mistral": activation_parent_dir + "/mistral_7B/test",
    "mixtral": activation_parent_dir + "/mixtral_8x7B_instruct_float16/test",
    "llama3_8b": activation_parent_dir + "/llama3_8b/test/",
    "phi3": activation_parent_dir + "/phi3/test",
    "mistral_no_priming": activation_parent_dir + "/mistral_no_priming/test",
    "llama3_70b": activation_parent_dir + "/llama_3_70B_Instruct/test",
}


# Activations text files of test data (clean)
TEST_CLEAN_FILES_PER_MODEL = {
    "mistral": [
        file.strip()
        for file in open(os.path.join(DATA_LISTS, "test_clean_files_mistral.txt"))
    ],
    "mixtral": [
        file.strip()
        for file in open(os.path.join(DATA_LISTS, "test_clean_files_mixtral.txt"))
    ],
    "llama3_8b": [
        file.strip()
        for file in open(os.path.join(DATA_LISTS, "test_clean_files_llama3_8b.txt"))
    ],
    "phi3": [
        file.strip()
        for file in open(os.path.join(DATA_LISTS, "test_clean_files_phi3.txt"))
    ],
    "mistral_no_priming": [
        file.strip()
        for file in open(
            os.path.join(DATA_LISTS, "no_priming", "test_clean_files_mistral.txt")
        )
    ],
    "llama3_70b": [
        file.strip()
        for file in open(os.path.join(DATA_LISTS, "test_clean_files_llama3_70b.txt"))
    ],
}

# Activations text files of test data (poisoned)
TEST_POISONED_FILES_PER_MODEL = {
    "mistral": [
        file.strip()
        for file in open(os.path.join(DATA_LISTS, "test_poisoned_files_mistral.txt"))
    ],
    "mixtral": [
        file.strip()
        for file in open(os.path.join(DATA_LISTS, "test_poisoned_files_mixtral.txt"))
    ],
    "llama3_8b": [
        file.strip()
        for file in open(os.path.join(DATA_LISTS, "test_poisoned_files_llama3_8b.txt"))
    ],
    "phi3": [
        file.strip()
        for file in open(os.path.join(DATA_LISTS, "test_poisoned_files_phi3.txt"))
    ],
    "mistral_no_priming": [
        file.strip()
        for file in open(
            os.path.join(DATA_LISTS, "no_priming", "test_poisoned_files_mistral.txt")
        )
    ],
    "llama3_70b": [
        file.strip()
        for file in open(os.path.join(DATA_LISTS, "test_poisoned_files_llama3_70b.txt"))
    ],
}
