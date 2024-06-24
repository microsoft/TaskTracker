from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import os
import torch


def load_model(model_name: str, cache_dir: str, torch_dtype: torch.dtype):
    config = AutoConfig.from_pretrained(model_name, use_cache=True)
    model = {
        "tokenizer": AutoTokenizer.from_pretrained(
            model_name, use_cache=True, cache_dir=os.path.join(cache_dir, model_name)
        ),
        "model": AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            cache_dir=os.path.join(cache_dir, model_name),
            device_map="balanced_low_0",
            torch_dtype=torch_dtype,
        ),
    }
    return model
