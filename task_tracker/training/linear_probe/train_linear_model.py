import os
import sys
import json 
import numpy as np
from tqdm import tqdm
from pprint import pprint
from sklearn.linear_model import LogisticRegression
import pickle 

MODEL = "phi3"
OUTPUT_DIR = MODEL 
os.makedirs(OUTPUT_DIR,exist_ok=True)


from task_tracker.training.dataset import ActivationsDatasetDynamic, ActivationsDatasetDynamicPrimaryText
from task_tracker.training.helpers.data import load_file_paths
from task_tracker.training.utils.constants import CONSTANTS_ALL_MODELS, OOD_POISONED_FILE



#Which layers would be used for training probes 
LAYERS_PER_MODEL = {
    'llama3_70b': [0, 7, 15, 23, 31, 39, 47, 55, 63, 71, 79 ],
    'phi3': [0, 7, 15, 23, 31],
    'mixtral': [0, 7, 15, 23, 31],
    'mistral': [0, 7, 15, 23, 31],
    'llama3_8b': [0, 7, 15, 23, 31],
    'mistral_no_priming': [0, 7, 15, 23, 31],
}


ACTIVATION_FILE_LIST_DIR, ACTIVATIONS_DIR, ACTIVATIONS_VAL_DIR =\
CONSTANTS_ALL_MODELS[MODEL]['ACTIVATION_FILE_LIST_DIR'], CONSTANTS_ALL_MODELS[MODEL]['ACTIVATIONS_DIR'], CONSTANTS_ALL_MODELS[MODEL]['ACTIVATIONS_VAL_DIR']

# Configuration settings
config = {
    'activations': ACTIVATIONS_DIR,
    'activations_ood': ACTIVATIONS_VAL_DIR,
    'ood_poisoned_file': OOD_POISONED_FILE,
    'exp_name': 'logistic_regression_' + MODEL,
}


def train_model(train_files, num_layers):
    print("Loading dataset.")
    dataset = ActivationsDatasetDynamic(train_files, root_dir=config['activations'], num_layers=num_layers)

    print("Processing dataset.")
    clean_diff = []
    poisoned_diff = []
    for primary, clean, poisoned in tqdm(dataset):
        clean_diff.append((clean - primary).flatten().float().numpy())
        poisoned_diff.append((poisoned - primary).flatten().float().numpy())
    y = [0]*len(dataset) + [1]*len(dataset)
    X = clean_diff + poisoned_diff

    print("Training logistic regression classifier.")
    model = LogisticRegression()
    model.fit(X, y)

    return model

def load_evaluation_data(val_files_clean, val_files_poisoned, num_layers):
    print("Loading validation datasets.")
    clean_dataset = ActivationsDatasetDynamicPrimaryText(val_files_clean, num_layers=num_layers, root_dir=config.get('activations_ood'))
    poisoned_dataset = ActivationsDatasetDynamicPrimaryText(val_files_poisoned, num_layers=num_layers, root_dir=config.get('activations_ood'))

    print("Processing validation datasets.")
    clean_diff = []
    for primary, clean_with_text in tqdm(clean_dataset):
        clean_diff.append((clean_with_text - primary).flatten().float().numpy())
    poisoned_diff = []
    for primary, poisoned_with_text in tqdm(poisoned_dataset):
        poisoned_diff.append((poisoned_with_text - primary).flatten().float().numpy())
    X_validation = np.array(clean_diff + poisoned_diff)
    y_validation = [0]*len(clean_diff) + [1]*len(poisoned_diff)

    return X_validation, y_validation


if __name__ == "__main__":
    LAYERS = LAYERS_PER_MODEL[MODEL]

    for n_layer in LAYERS:
        print(f"[*] Training model for the {n_layer}-th activation layer.")
        os.makedirs(os.path.join(OUTPUT_DIR,str(n_layer)),exist_ok=True)
        layer_output_dir = os.path.join(OUTPUT_DIR,str(n_layer))
        

        _config = config.copy()
        _config["num_layers"] = n_layer
        _config["exp_name"] = f"{config['exp_name']}_{n_layer}"
        print(_config["exp_name"])
        with open(os.path.join(layer_output_dir,'config.json'), 'w') as f:
            json.dump(_config, f)

        # Train the model.
        train_files = load_file_paths(os.path.join(ACTIVATION_FILE_LIST_DIR, 'train_files_' + MODEL + '.txt'))
        val_files_clean = load_file_paths(os.path.join(ACTIVATION_FILE_LIST_DIR, 'val_clean_files_' + MODEL + '.txt'))
        val_files_poisoned = load_file_paths(os.path.join(ACTIVATION_FILE_LIST_DIR, 'val_poisoned_files_' + MODEL + '.txt'))

        print(f"Training model with {len(train_files)} files.")
        print(f"Evaluating model with {len(val_files_clean)} clean files and {len(val_files_poisoned)} poisoned files.")

        model = train_model(train_files, num_layers=(n_layer, n_layer))
        pickle.dump(model, open(os.path.join(layer_output_dir,'model.pickle'), "wb"))


        # Evaluate.
        X_eval, y_eval = load_evaluation_data(val_files_clean, val_files_poisoned, num_layers=(n_layer, n_layer))
        accuracy = model.score(X_eval, y_eval)
        print(accuracy)
        print("\n"*4)