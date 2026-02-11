import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor, AutoTokenizer
from generate_mask import SaliencyMaskGenerator

# --- Initialization ---
MODEL_NAME = "llava-hf/llava-1.5-7b-hf"
model = LlavaForConditionalGeneration.from_pretrained(MODEL_NAME, torch_dtype=torch.float32, device_map="cpu")

forget_tensors = torch.load("forget.pt", weights_only=False)
retain_tensors = torch.load("retain.pt", weights_only=False)

target_layers = [
    "model.vision_tower.vision_model",
    "model.multi_modal_projector",
    "model.language_model"
]

# --- Execution ---
mask_engine = SaliencyMaskGenerator(model, forget_tensors, retain_tensors, target_layers)
mask_engine.compute_gradient_importance()
final_mask = mask_engine.generate_mask(threshold_percentile=0.95)

print("Saliency mask successfully generated.")