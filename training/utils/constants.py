import os 
CURRENT_DIR = '/share/projects/TaskTracker/'

# == Experiment-wide constants ==
# Config file location for our AML workspace which tracks experiment run and metrics
AML_WORKSPACE_CONFIG = '/share/projects/jailbreak-activations/siamese/config/aml-config.json'
# The output directory to store output models per epoch and the best running model
MODEL_OUTPUT_DIR =  '/share/projects/jailbreak-activations/output_dir'
# Directory of the text of sampled training examples 
TRAINING_TEXT_DIR = '/share/projects/jailbreak-activations/dataset_sampling/dataset_sampled_examples'
# OOD poisoned file (Test)
OOD_POISONED_FILE = '/share/projects/jailbreak-activations/dataset_sampling/dataset_sampled_test/dataset_out_poisoned_v1.json'


# Constants for experiments for each model 

CONSTANTS_ALL_MODELS = {
    'phi3': {
        # where the List of activations files to pull are stored
        'ACTIVATION_FILE_LIST_DIR': '/share/projects/TaskTracker/data',
        # Activations data
        'ACTIVATIONS_DIR': '/home/asalem/disk1/data_instruct_sep/get_activations/phi3/',
        #Activations validation data (OOD)
        'ACTIVATIONS_VAL_DIR': '/home/asalem/disk1/data_instruct_sep/get_activations/phi3/validation'
    },
    'mixtral': {
        'ACTIVATION_FILE_LIST_DIR' : '/share/projects/TaskTracker/data',
        # Activations data
        'ACTIVATIONS_DIR': '/disk1/activations/mixtral_8x7B_instruct_float16/',
        # Activations validation data (OOD)
        'ACTIVATIONS_VAL_DIR' : '/disk1/activations/mixtral_8x7B_instruct_float16/validation/'
    },
    
    'mistral': {
        # == With Priming ==
        
        # where the List of activations files to pull are stored
        'ACTIVATION_FILE_LIST_DIR' : '/share/projects/TaskTracker/data',
        # Activations training data dir 
        'ACTIVATIONS_DIR' : '/disk1/activations/mistral_7B/training',
        # Activations validation data (OOD) dir
        'ACTIVATIONS_VAL_DIR' : '/disk1/activations/mistral_7B/validation'
    },
    'llama3_8b': {
        # where the List of activations files to pull are stored
        'ACTIVATION_FILE_LIST_DIR' : '/share/projects/TaskTracker/data',
        # Activations training data dir 
        'ACTIVATIONS_DIR' : '/disk1/activations/llama3_8b/training',
        # Activations validation data (OOD) dir
        'ACTIVATIONS_VAL_DIR' : '/disk1/activations/llama3_8b/validation/'
    },
    'llama3_70b': {
        # where the List of activations files to pull are stored
        'ACTIVATION_FILE_LIST_DIR' : '/share/projects/TaskTracker/data',
        # Activations training data dir 
        'ACTIVATIONS_DIR' : '/disk3/activations/llama_3_70B_Instruct/training',
        # Activations validation data (OOD) dir
        'ACTIVATIONS_VAL_DIR' : '/disk3/activations/llama_3_70B_Instruct/validation'
    },
    'mistral_no_priming': {
        # == Without priming ==
        
        # where the List of activations files to pull are stored
        'ACTIVATION_FILE_LIST_DIR' : '/share/projects/TaskTracker/data/no_priming',
        # Activations training data dir 
        'ACTIVATIONS_DIR' : '/disk1/activations/mistral_no_priming/training', 
        # Activations validation data (OOD) dir
        'ACTIVATIONS_VAL_DIR' : '/disk1/activations/mistral_no_priming/test' 
    }
}


TEST_ACTIVATIONS_DIR_PER_MODEL = {
    'mistral': '/disk1/activations/mistral_7B/test',
    'mixtral': '/disk1/activations/mixtral_8x7B_instruct_float16/test',
    'llama3_8b': '/disk1/activations/llama3_8b/test/',
    'phi3': '/home/asalem/disk1/data_instruct_sep/get_activations/phi3/test',
    'mistral_no_priming': '/disk1/activations/mistral_no_priming/test',
    'llama3_70b': '/disk3/activations/llama_3_70B_Instruct/test'
    
}


DATA_LISTS = os.path.join(CURRENT_DIR, 'data')
TEST_CLEAN_FILES_PER_MODEL = {
    'mistral': [file.strip() for file in open(os.path.join(DATA_LISTS, 'test_clean_files_mistral.txt'))],
    'mixtral': [file.strip() for file in open(os.path.join(DATA_LISTS, 'test_clean_files_mixtral.txt'))],
    'llama3_8b' : [file.strip() for file in open(os.path.join(DATA_LISTS, 'test_clean_files_llama3_8b.txt'))],
    'phi3' : [file.strip() for file in open(os.path.join(DATA_LISTS, 'test_clean_files_phi3.txt'))],
    'mistral_no_priming' : [file.strip() for file in open(os.path.join(DATA_LISTS, 'no_priming', 'test_clean_files_mistral.txt'))],
    'llama3_70b' : [file.strip() for file in open(os.path.join(DATA_LISTS, 'test_clean_files_llama3_70b.txt'))]
}

TEST_POISONED_FILES_PER_MODEL = {
    'mistral': [file.strip() for file in open(os.path.join(DATA_LISTS, 'test_poisoned_files_mistral.txt'))],
    'mixtral': [file.strip() for file in open(os.path.join(DATA_LISTS, 'test_poisoned_files_mixtral.txt'))],
    'llama3_8b' : [file.strip() for file in open(os.path.join(DATA_LISTS, 'test_poisoned_files_llama3_8b.txt'))],
    'phi3' : [file.strip() for file in open(os.path.join(DATA_LISTS, 'test_poisoned_files_phi3.txt'))],
    'phi3' : [file.strip() for file in open(os.path.join(DATA_LISTS,  'test_poisoned_files_phi3.txt'))],
    'mistral_no_priming' : [file.strip() for file in open(os.path.join(DATA_LISTS, 'no_priming', 'test_poisoned_files_mistral.txt'))],
    'llama3_70b' : [file.strip() for file in open(os.path.join(DATA_LISTS, 'test_poisoned_files_llama3_70b.txt'))]
}


