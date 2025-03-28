import os
import glob
import json
import torch
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm, trange
from transformers import AutoModelForCausalLM
import hydra
from omegaconf import DictConfig
from utils.data import get_data_for_inference, get_tokenizer, Datamodule

def load_model_and_tokenizer(config):
    print(f"Loading model from {config.inference.notebook_modelpath}...")
    
    # Load tokenizer using the same function from your original code
    tokenizer = get_tokenizer(config.tok_data)
    
    # Load model
    model_dir = Path(config.inference.notebook_modelpath)
    
    # Check if we have a state dict saved separately
    state_dict_path = model_dir / "model.pth"
    state_dict = None
    if state_dict_path.exists():
        print("Loading state dict from model.pth...")
        state_dict = torch.load(state_dict_path)
    
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        state_dict=state_dict,
        # Use flash attention if available
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
    )
    
    # Move model to device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    return model, tokenizer

def load_single_example(config, tokenizer, sample_id):
    datapath = config.tok_data.val_file
    print(f"Loading example from {datapath}...")
    
    # Load the example from JSON
    with open(datapath, 'r') as f:
        data = json.load(f)

    # Extract the text directly from the first item
    example_text = data[sample_id]["text"]
    
    # Find if there's a "Command:" delimiter in the prompt
    delimiter = config.data.split_str

    parts = example_text.split(delimiter, 1)
    prompt = parts[0] + delimiter
    
    # Store the ground truth (what comes after "Command:") for comparison
    ground_truth = parts[1].strip()
    print(f"Ground truth answer: '{ground_truth}'")
    
    print(f"Loaded example prompt: {prompt[:100]}...")  # Print first 100 chars
    
    if tokenizer.bos_token and not prompt.startswith(tokenizer.bos_token):
        full_prompt = tokenizer.bos_token + " " + prompt
    else:
        full_prompt = prompt
    
    return full_prompt, {"text": example_text, "ground_truth": ground_truth}

def run_inference(model, tokenizer, prompt, config):
    print("Tokenizing prompt and running inference...")
    
    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_length = inputs['input_ids'].shape[1]
    
    # Generate outputs
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask", None),
            max_length=config.model.block_size,
            pad_token_id=tokenizer.pad_token_id,
            num_beams=1,  # Greedy decoding
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Get the generated text (everything after the prompt)
    generated_text = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=False)
    
    # Get token-level output
    generated_ids = outputs[0][prompt_length:].tolist()
    
    result = generated_text.strip()
    
    return result, generated_text, generated_ids

# Run the main function if executed as a script
@hydra.main(
    config_path="config",
    config_name="config_base",
    version_base=None,
)
def main(cfg: DictConfig):
    # This will trigger Hydra configuration loading

    model, tokenizer = load_model_and_tokenizer(cfg)

    # Load the example
    prompt, example = load_single_example(cfg, tokenizer, sample_id=42)

    # Run inference
    result, full_generated_text, generated_ids = run_inference(model, tokenizer, prompt, cfg)
    print("\nexample:", example, "\n")
    print("result:", result, "\n",
          "full_generated_text:", 
          full_generated_text, "\n", 
          "generated_ids:", 
          generated_ids)

if __name__ == "__main__":
    main()