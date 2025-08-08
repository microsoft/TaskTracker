import logging
import os
import pickle
from typing import Dict

import torch
import yaml
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, pipeline

# Task Tracker prompts
SEP_PROMPT = (
    "Consider the following request that you must answer based on the given text: "
)

GLOBAL_USER_PROMPT = "Summarize the following text"

# Anything you want to add here as part of a meta-prompt
ADDITIONAL_PROMPT = ""


def setup_hf_llm(model_name, cache_dir, torch_type=torch.bfloat16):
    """
    Sets up a Hugging Face model and tokenizer, caching it for future use.
    """
    config = AutoConfig.from_pretrained(
        model_name,
        use_cache=True,
        cache_dir=os.path.join(cache_dir, model_name),
        device_map="auto",
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        cache_dir=os.path.join(cache_dir, model_name),
        device_map="auto",
        torch_dtype=torch_type,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_cache=True)
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def load_config(config_path: str = "config.yaml") -> Dict:
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)  # Load the YAML file contents into a dictionary
        return config
    except FileNotFoundError:
        logging.error("Configuration file not found.", exc_info=True)
        raise
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML configuration: {e}", exc_info=True)
        raise
    except Exception as e:
        logging.error(f"Unexpected error loading configuration: {e}", exc_info=True)
        raise


def load_task_tracker(model_path: str):
    try:
        with open(model_path, "rb") as f:
            task_tracker_model = pickle.load(f)
        return task_tracker_model
    except FileNotFoundError:
        logging.error(f"Model file not found at path: {model_path}", exc_info=True)
        raise
    except pickle.UnpicklingError:
        logging.error(f"Error unpickling model from file: {model_path}", exc_info=True)
        raise
    except Exception as e:
        logging.error(
            f"Unexpected error loading task tracker model: {e}", exc_info=True
        )
        raise


def get_last_token_activations(text, model_name, layer, model, tokenizer):
    if "mistral" in model_name.lower() or "phi" in model_name.lower():
        chat = [
            {
                "role": "user",
                "content": "you are a helpful assistant that will provide accurate answers to all questions. "
                + text,
            }
        ]
    else:
        chat = [
            {
                "role": "system",
                "content": "you are a helpful assistant that will provide accurate answers to all questions.",
            },
            {"role": "user", "content": text},
        ]
    inputs = tokenizer.apply_chat_template(
        chat, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(inputs.cuda(), output_hidden_states=True)

    last_token_activations = outputs["hidden_states"][layer][:, -1].cpu()
    return last_token_activations


def format_prompts(documents, specific_user_prompt=""):

    user_prompt = (
        specific_user_prompt if specific_user_prompt != "" else GLOBAL_USER_PROMPT
    ) + ADDITIONAL_PROMPT

    orig_task = SEP_PROMPT + " " + user_prompt

    eliciting_instructions = """ but first before you answer, please complete the following sentence by briefly writing each request(s) you received and you are going to execute next:  
        "All requests that I am going to execute now are:" """

    primary = (
        "here are your main requests: <MAIN> "
        + orig_task
        + " </MAIN>"
        + eliciting_instructions
    )
    primary_text = (
        "here are your main requests: <MAIN> "
        + orig_task
        + " "
        + documents
        + " </MAIN>"
        + eliciting_instructions
    )
    return primary, primary_text


def task_tracker_main(
    documents,
    llm,
    llm_name,
    tokenizer,
    task_tracker_model,
    layer,
    specific_user_prompt="",
):

    primary, primary_text = format_prompts(documents, specific_user_prompt)
    primary_activations = get_last_token_activations(
        primary, llm_name, layer, llm, tokenizer
    )
    primary_text_activations = get_last_token_activations(
        primary_text, llm_name, layer, llm, tokenizer
    )

    deltas = (primary_text_activations - primary_activations).float().numpy()

    y_pred_prob = task_tracker_model.predict_proba(deltas)[:, 1]

    return y_pred_prob
