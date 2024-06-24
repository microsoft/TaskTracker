import os 

## directory of dataset (as text)
from task_tracker.config.models import text_dataset_parent_dir


#Constants for parent dir of experiments outputs 
linear_probe_out_parent_dir = "/share/projects/jailbreak-activations/linear_probing/training"
triplet_probe_out_parent_dir = "/share/projects/jailbreak-activations/output_dir"

MODELS_RESPONSE_OUT_FILENAME_PER_MODEL = {
    
    'mistral': text_dataset_parent_dir + '/dataset_sampled_test/dataset_out_poisoned_test_mistral_response_test.json',
    'mixtral': text_dataset_parent_dir + '/dataset_sampled_test/dataset_out_poisoned_test_mixtral_response.json',
    'llama3_8b' : text_dataset_parent_dir + '/dataset_sampled_test/dataset_out_poisoned_test_llama3_8b_response.json'
}

VERIFIER_RESPONSE_OUT_FILENAME_PER_MODEL = {
    'mistral': text_dataset_parent_dir + '/dataset_sampled_test/mistral_responses/dataset_out_poisoned_v2_mistral_response_verifier_full_data_cleaned.json',
    'mixtral': text_dataset_parent_dir + '/dataset_sampled_test/mixtral_responses/dataset_out_poisoned_v2_mixtral_response_verifier_full_data_cleaned.json',
    'llama3_8b' : text_dataset_parent_dir + '/dataset_sampled_test/llama3_8b_responses/dataset_out_poisoned_v2_llama3_8b_response_verifier_full_data_cleaned.json'
}


## Path to trained logistic regression models 
LINEAR_PROBES_PATHS_PER_MODEL = {
    'mistral' : {
         linear_probe_out_parent_dir +'/mlruns/0/599aacf342f2439ca3580f5a63907c68/artifacts/model': 0,
         linear_probe_out_parent_dir + '/mlruns/0/c780daad0f324d0496fd63553553718f/artifacts/model': 15,
         linear_probe_out_parent_dir + '/training/mistral/mistral_uksouth-2_layer_23_model.pickle': 23,
         linear_probe_out_parent_dir + '/mlruns/0/eab2fe2e6cf24ff08f7e2b8788b77d59/artifacts/model': 31,
         linear_probe_out_parent_dir + '/mlruns/0/f75afb2b80be4541b4f65ad1558ba023/artifacts/model': 7
    },
    'mixtral' : {
        linear_probe_out_parent_dir + '/mixtral/mlruns/0/5dc3a3ae54184b8f824b77517d4495b5/artifacts/model': 7,
        linear_probe_out_parent_dir + '/mixtral/mlruns/0/9e79a77d73ff4207b2a637125e532e9e/artifacts/model': 23,
        linear_probe_out_parent_dir + '/mixtral/mlruns/0/8cae6f94286149569a008d1a6a73a5fb/artifacts/model': 15,
        linear_probe_out_parent_dir + '/mixtral/mlruns/0/c122ca3e1cef404a8eef584400f4e17f/artifacts/model': 0,
        linear_probe_out_parent_dir + '/mixtral/mlruns/0/79f8480830814cb18433e340845cf7b5/artifacts/model': 31
    },
    
    'llama3_8b' : {
        linear_probe_out_parent_dir + '/llama3_8b/0/model.pickle' : 0,
        linear_probe_out_parent_dir + '/llama3_8b/7/model.pickle' : 7,
        linear_probe_out_parent_dir + '/llama3_8b/15/model.pickle' : 15,
        linear_probe_out_parent_dir + '/llama3_8b/23/model.pickle' : 23,
        linear_probe_out_parent_dir + '/llama3_8b/31/model.pickle' : 31
    },
    'phi3' : { 
        linear_probe_out_parent_dir + '/phi3/0/model.pickle': 0,
        linear_probe_out_parent_dir + '/phi3/7/model.pickle': 7,
        linear_probe_out_parent_dir + '/phi3/15/model.pickle': 15, 
        linear_probe_out_parent_dir + '/phi3/23/model.pickle': 23,
        linear_probe_out_parent_dir + '/phi3/31/model.pickle': 31 
    },
    'mistral_no_priming' : {
        linear_probe_out_parent_dir + '/mistral_no_priming/0/model.pickle': 0,
        linear_probe_out_parent_dir + '/mistral_no_priming/7/model.pickle': 7,
        linear_probe_out_parent_dir + '/mistral_no_priming/15/model.pickle': 15, 
        linear_probe_out_parent_dir + '/mistral_no_priming/23/model.pickle': 23,
        linear_probe_out_parent_dir + '/mistral_no_priming/31/model.pickle': 31 
        
    },
    'llama3_70b' : {
        linear_probe_out_parent_dir + '/llama3_70b/0/model.pickle' : 0,
        linear_probe_out_parent_dir + '/llama3_70b/7/model.pickle' : 7,
        linear_probe_out_parent_dir + '/llama3_70b/15/model.pickle' : 15,
        linear_probe_out_parent_dir + '/llama3_70b/23/model.pickle' : 23,
        linear_probe_out_parent_dir + '/llama3_70b/31/model.pickle' : 31,
        linear_probe_out_parent_dir + '/llama3_70b/39/model.pickle' : 39,
        linear_probe_out_parent_dir + '/llama3_70b/47/model.pickle' : 47, 
        linear_probe_out_parent_dir + '/llama3_70b/55/model.pickle' : 55,
        linear_probe_out_parent_dir + '/llama3_70b/63/model.pickle' : 63,
        linear_probe_out_parent_dir + '/llama3_70b/71/model.pickle' : 71, 
        linear_probe_out_parent_dir + '/llama3_70b/79/model.pickle' : 79 
    },
}



TRIPLET_PROBES_PATHS_PER_MODEL = {
    'mistral' : {'path': triplet_probe_out_parent_dir + '/mistral_best', 
                 'num_layers': (17,31), 
                 'saved_embs_clean': 'clean_embeddings_20240429-133151.json' ,
                 'saved_embs_poisoned':  'poisoned_embeddings_20240429-134637.json'},
    
    'mixtral' : {'path': triplet_probe_out_parent_dir + '/mixtral_best', 
                 'num_layers': (0,5),
                 'saved_embs_clean': 'clean_embeddings_20240512-184429.json' ,
                 'saved_embs_poisoned':  'poisoned_embeddings_20240512-184534.json'},
    
    'llama3_70b' : {'path': triplet_probe_out_parent_dir + '/llama3_70b_best', 
                    'num_layers': (1,15),
                    'saved_embs_clean': 'clean_embeddings_20240614-160133.json',
                    'saved_embs_poisoned':  'poisoned_embeddings_20240614-160957.json'},
    
    'llama3_8b': {'path': triplet_probe_out_parent_dir + '/llama3_8b_best', 
                  'num_layers': (0,5),
                  'saved_embs_clean': 'clean_embeddings_20240512-205233.json',
                  'saved_embs_poisoned': 'poisoned_embeddings_20240512-205937.json'}
}