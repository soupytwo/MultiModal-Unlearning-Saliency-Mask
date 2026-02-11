from PIL import Image

class CLEARDatasetProcessor:
    """Handles the preprocessing of the CLEAR dataset for Llava multimodal inference."""
    
    def __init__(self, model, processor, tokenizer):
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer

    def process_row(self, dataset_row):
        """
        Converts a raw dataset row into model-ready tensors.
        
        Args:
            dataset_row (dict): A single entry from the CLEAR dataset containing 'image', 'caption', and 'name'.
        """
        image = dataset_row["image"].convert("RGB")

        # Format prompt according to Llava 1.5 requirements
        prompt_text = (
            "<image>\n"
            f"Caption: {dataset_row['caption']}\n"
            f"Name: {dataset_row['name']}"
        )

        model_inputs = self.processor(images=image, text=prompt_text, return_tensors="pt")

        # Set up Labels for Training/Inference
        # We clone input_ids and set pad tokens to -100 so they are ignored by CrossEntropyLoss
        target_labels = model_inputs["input_ids"].clone()  
        target_labels[target_labels == self.tokenizer.pad_token_id] = -100
        model_inputs["labels"] = target_labels

        return model_inputs