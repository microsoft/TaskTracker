## Path to trained logistic regression models 
LINEAR_PROBES_PATHS_PER_MODEL = {
    'mistral' : {
         '/share/projects/jailbreak-activations/linear_probing/mlruns/0/599aacf342f2439ca3580f5a63907c68/artifacts/model': 0,
         '/share/projects/jailbreak-activations/linear_probing/mlruns/0/c780daad0f324d0496fd63553553718f/artifacts/model': 15,
         '/share/projects/jailbreak-activations/linear_probing/training/mistral/mistral_uksouth-2_layer_23_model.pickle': 23,
         '/share/projects/jailbreak-activations/linear_probing/mlruns/0/eab2fe2e6cf24ff08f7e2b8788b77d59/artifacts/model': 31,
         '/share/projects/jailbreak-activations/linear_probing/mlruns/0/f75afb2b80be4541b4f65ad1558ba023/artifacts/model': 7
    },
    'mixtral' : {
        '/share/projects/jailbreak-activations/linear_probing/mixtral/mlruns/0/5dc3a3ae54184b8f824b77517d4495b5/artifacts/model': 7,
        '/share/projects/jailbreak-activations/linear_probing/mixtral/mlruns/0/9e79a77d73ff4207b2a637125e532e9e/artifacts/model': 23,
        '/share/projects/jailbreak-activations/linear_probing/mixtral/mlruns/0/8cae6f94286149569a008d1a6a73a5fb/artifacts/model': 15,
        '/share/projects/jailbreak-activations/linear_probing/mixtral/mlruns/0/c122ca3e1cef404a8eef584400f4e17f/artifacts/model': 0,
        '/share/projects/jailbreak-activations/linear_probing/mixtral/mlruns/0/79f8480830814cb18433e340845cf7b5/artifacts/model': 31
    },
    
    'llama3_8b' : {
        '/share/projects/jailbreak-activations/linear_probing/training/llama/mlruns/0/f677c7865f2a49fea34a736580a158a4/artifacts/model' : 0,
        '/share/projects/jailbreak-activations/linear_probing/training/llama/mlruns/0/d3bd5f9deb104c91b3f3900204b37634/artifacts/model' : 7,
        '/share/projects/jailbreak-activations/linear_probing/training/llama/mlruns/0/479ce4baf7d7435da837ef5205c24ce8/artifacts/model' : 15,
        '/share/projects/jailbreak-activations/linear_probing/training/llama/mlruns/0/df7a153505294694bdecbc9c45d07ef8/artifacts/model' : 23,
        '/share/projects/jailbreak-activations/linear_probing/training/llama/mlruns/0/0c8adf3a91424132b2ea1737e670c65e/artifacts/model' : 31
    },
    'phi3' : { 
        '/share/projects/jailbreak-activations/linear_probing/training/phi3/0/model.pickle': 0,
        '/share/projects/jailbreak-activations/linear_probing/training/phi3/7/model.pickle': 7,
        '/share/projects/jailbreak-activations/linear_probing/training/phi3/15/model.pickle': 15, 
        '/share/projects/jailbreak-activations/linear_probing/training/phi3/23/model.pickle': 23,
        '/share/projects/jailbreak-activations/linear_probing/training/phi3/31/model.pickle': 31 
    },
    'mistral_no_priming' : {
        '/share/projects/jailbreak-activations/linear_probing/training/mistral_nopriming/manual_saves/mistral_layer_0_model.pickle': 0,
        '/share/projects/jailbreak-activations/linear_probing/training/mistral_nopriming/manual_saves/mistral_layer_7_model.pickle': 7,
        '/share/projects/jailbreak-activations/linear_probing/training/mistral_nopriming/manual_saves/mistral_layer_15_model.pickle': 15, 
        '/share/projects/jailbreak-activations/linear_probing/training/mistral_nopriming/manual_saves/mistral_layer_23_model.pickle': 23,
        '/share/projects/jailbreak-activations/linear_probing/training/mistral_nopriming/manual_saves/mistral_layer_31_model.pickle': 31 
        
    },
    'llama3_70b' : {
        '/share/projects/jailbreak-activations/linear_probing/training/llama3_70b/0/model.pickle' : 0,
        '/share/projects/jailbreak-activations/linear_probing/training/llama3_70b/7/model.pickle' : 7,
        '/share/projects/jailbreak-activations/linear_probing/training/llama3_70b/15/model.pickle' : 15,
        '/share/projects/jailbreak-activations/linear_probing/training/llama3_70b/23/model.pickle' : 23,
        '/share/projects/jailbreak-activations/linear_probing/training/llama3_70b/31/model.pickle' : 31,
        '/share/projects/jailbreak-activations/linear_probing/training/llama3_70b/39/model.pickle' : 39,
        '/share/projects/jailbreak-activations/linear_probing/training/llama3_70b/47/model.pickle' : 47, 
        '/share/projects/jailbreak-activations/linear_probing/training/llama3_70b/55/model.pickle' : 55,
        '/share/projects/jailbreak-activations/linear_probing/training/llama3_70b/63/model.pickle' : 63,
        '/share/projects/jailbreak-activations/linear_probing/training/llama3_70b/71/model.pickle' : 71, 
        '/share/projects/jailbreak-activations/linear_probing/training/llama3_70b/79/model.pickle' : 79 
    },
}



TRIPLET_PROBES_PATHS_PER_MODEL = {
    'mistral' : {'path': '/share/projects/jailbreak-activations/output_dir/mistral_best', 
                 'num_layers': (17,31), 
                 'saved_embs_clean': 'clean_embeddings_20240429-133151.json' ,
                 'saved_embs_poisoned':  'poisoned_embeddings_20240429-134637.json'},
    'mixtral' : {'path': '/share/projects/jailbreak-activations/output_dir/mixtral_best', 
                 'num_layers': (0,5),
                 'saved_embs_clean': 'clean_embeddings_20240512-184429.json' ,
                 'saved_embs_poisoned':  'poisoned_embeddings_20240512-184534.json'},
    'llama3_70b' : {'path': '/share/projects/jailbreak-activations/output_dir/llama3_70b_best', 
                    'num_layers': (1,15),
                    'saved_embs_clean': 'clean_embeddings_20240614-160133.json',
                    'saved_embs_poisoned':  'poisoned_embeddings_20240614-160957.json'},
    'llama3_8b': {'path': '/share/projects/jailbreak-activations/output_dir/llama3_8b_best', 
                  'num_layers': (0,5),
                  'saved_embs_clean': 'clean_embeddings_20240512-205233.json',
                  'saved_embs_poisoned': 'poisoned_embeddings_20240512-205937.json'}
}