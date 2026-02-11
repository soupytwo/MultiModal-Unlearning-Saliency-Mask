import torch

class SaliencyMaskGenerator:
    """
    Calculates parameter importance by comparing gradients of 'forget' vs 'retain' data.
    This utility is used to identify specific weights in a multimodal model
     that contribute to certain concepts while leaving others untouched.
    """
    
    def __init__(self, model, forget_data, retain_data, target_modules):
        """
        Initializes the generator with model and data slices.
        
        Args:
            model: The pre-trained Llava model, the intended model is: "llava-hf/llava-1.5-7b-hf"
            forget_data (list): Tensors containing information to be unlearned.
            retain_data (list): Tensors containing information to be preserved.
            target_modules (list): String prefixes of the model layers to analyze 
                                   (e.g., 'model.language_model').
        """
        self.model = model
        self.forget_data = forget_data
        self.retain_data = retain_data
        self.target_modules = target_modules

        # Dictionaries to store the cumulative absolute gradients (importance scores)
        self.forget_importance_map = {}
        self.retain_importance_map = {}
        self.saliency_mask = {}

    def compute_importance_scores(self):
        """
        Computes the average absolute gradient for both forget and retain sets.
        This identifies which weights are most 'active' or sensitive for specific inputs.
        """
        self.model.train()

        # Initialize tracking for only the requested modules to save memory
        for name, param in self.model.named_parameters():
            if any(name.startswith(module_prefix) for module_prefix in self.target_modules):
                self.forget_importance_map[name] = torch.zeros_like(param.data)
                self.retain_importance_map[name] = torch.zeros_like(param.data)

        # 1. ANALYZE FORGET DATA
        print("Analyzing gradients for the 'Forget' dataset...")
        for batch in self.forget_data:
            batch = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = self.model(**batch)
            
            loss = -outputs.loss
            self.model.zero_grad()
            loss.backward()

            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if name in self.forget_importance_map and param.grad is not None:
                        self.forget_importance_map[name] += param.grad.data.abs().cpu() / len(self.forget_data)
        
        # 2. ANALYZE RETAIN DATA
        print("Analyzing gradients for the 'Retain' dataset...")
        for batch in self.retain_data:
            batch = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = self.model(**batch)
            loss = outputs.loss

            self.model.zero_grad()
            loss.backward()

            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if name in self.retain_importance_map and param.grad is not None:
                        self.retain_importance_map[name] += param.grad.data.abs().cpu() / len(self.retain_data)
        
        print("Importance scoring complete.")

    def generate_binary_mask(self, threshold_percentile=0.9):
        """
        Creates a 0/1 mask where 1 marks a parameter that is high-impact for 
        forgetting but low-impact for retaining.
        
        Formula: Saliency = Forget_Importance / (Retain_Importance + epsilon)
        """
        print(f"Calculating binary mask at the {threshold_percentile*100}th percentile...")
        
        for name in self.forget_importance_map.keys():
            # Epsilon prevents division by zero
            epsilon = 1e-8
            saliency_scores = self.forget_importance_map[name] / (self.retain_importance_map[name] + epsilon)
            
            flattened_scores = saliency_scores.view(-1)
            if flattened_scores.numel() == 0:
                continue
                
            # Calculate the top-K threshold based on the desired percentile
            k_elements = int((1 - threshold_percentile) * flattened_scores.numel())
            k_elements = max(1, k_elements) # Ensure at least one parameter is picked
            
            threshold_value = torch.topk(flattened_scores, k_elements).values[-1]
            
            self.saliency_mask[name] = (saliency_scores >= threshold_value).float().to(self.model.device)

        print("Binary mask generation complete.")
        return self.saliency_mask