import torch
import numpy as np
import json 
import os 
import sys 
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig 
from transformers import AutoTokenizer, pipeline
current_dir = '/share/projects/TaskTracker'
sys.path.append(current_dir)
from text_dataset_files_constants import POISONED_TEST_DATASET_FILENAME, MODELS_RESPONSE_OUT_FILENAME_PER_MODEL, VERIFIER_RESPONSE_OUT_FILENAME_PER_MODEL

TOKEN = '' #add HF token 
CACHE_DIR = '/disk1/'

START_IDX = 0 
END_IDX = -1 # -1 will save for everything 
MODEL = 'mistral' ##change this to something else when needed 

MODEL_TO_HF_NAME = {'mistral': 'mistralai/Mistral-7B-Instruct-v0.2',
                    'mixtral': 'mistralai/Mixtral-8x7B-Instruct-v0.1',
                    'llama3_8b': 'meta-llama/Meta-Llama-3-8B-Instruct',
                    'llama3_70b': 'meta-llama/Meta-Llama-3-70B-Instruct'}

MODEL_NAME = MODEL_TO_HF_NAME[MODEL]

RESPONSE_OUT_FILENAME = MODELS_RESPONSE_OUT_FILENAME_PER_MODEL[MODEL] ##output file 



def load_model(model,cache_dir,hf_token):
    config = AutoConfig.from_pretrained(model)
    model = {
        "tokenizer": AutoTokenizer.from_pretrained(model),
        "model": AutoModelForCausalLM.from_pretrained(model, config=config,cache_dir=f'{cache_dir}/{model}', device_map="balanced_low_0",token=hf_token)
    }
    return model 


model = load_model(MODEL_NAME,CACHE_DIR,TOKEN)
tokenizer = model["tokenizer"]
model = model["model"]
model.eval()
gen_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=4096, return_full_text=False)


def get_answers(prompt, tokenizer,pipeline):
    chat = [{"role": "user", "content": prompt}]
    model_input = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    model_output = pipeline(model_input)[0]['generated_text']
    return model_output
    
data = json.load(open(POISONED_TEST_DATASET_FILENAME))
print(f'Start index: {START_IDX}')
if START_IDX != 0:
    responses = json.load(open(RESPONSE_OUT_FILENAME))
else:
    responses = {}

END_IDX = len(data) if END_IDX == -1 else END_IDX
print(f'End index: {END_IDX}')

for i in range(START_IDX,END_IDX): 
    prompt = data[i]['final_aggregated_prompt']

    model_output = get_answers(prompt, tokenizer,gen_pipeline)

    responses[i] = model_output
    if (i+1)%100 == 0:
        with open(RESPONSE_OUT_FILENAME, 'w', encoding ='utf8') as json_file: 
            json.dump(responses, json_file)
        print(f'Saved until index: {i}')

with open(RESPONSE_OUT_FILENAME, 'w', encoding ='utf8') as json_file: 
    json.dump(responses, json_file)
    print(f'Saved until index: {END_IDX}')
        
print(f"Inference complete. Results saved to {RESPONSE_OUT_FILENAME}")