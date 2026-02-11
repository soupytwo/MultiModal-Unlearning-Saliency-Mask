from process_dataset import CLEARDatasetProcessor
from datasets import load_dataset
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, AutoTokenizer

# --- Setup ---
MODEL_NAME = "llava-hf/llava-1.5-7b-hf"
model = LlavaForConditionalGeneration.from_pretrained(MODEL_NAME, torch_dtype=torch.float32, device_map="auto")
processor = AutoProcessor.from_pretrained(MODEL_NAME, use_fast=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

dataset = load_dataset("therem/CLEAR", "full")["train"]
data_handler = CLEARDatasetProcessor(model, processor, tokenizer)


"""
Processes a range of data and saves the resulting tensors. 
This is just for testing and will eventually move to more intentful datapoints.
"""
def generate_tensor_batch(start_idx, end_idx, output_path):
    processed_samples = []
    
    for i in range(start_idx, end_idx):
        processed_row = data_handler.process_row(dataset[i])
        processed_samples.append(processed_row)
        print(f"Index {i}: Processing complete.")
    
    torch.save(processed_samples, output_path)


generate_tensor_batch(0, 5, "forget.pt")
generate_tensor_batch(5, 10, "retain.pt")