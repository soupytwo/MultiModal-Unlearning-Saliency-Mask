# MultiModal Unlearning: Saliency Mask Generation

## Description

This project implements a Machine Unlearning Pipeline, particularly for Multimodal Large Language Models (MLLMs) such as Llava-1.5B-7B. The core logic uses a gradient-based saliency mask that identifies specific parameters to forget/suppress while retaining semantic truth by remembering a retain dataset. 

The saliency mask is calculated by calculating a cumulative ratio of the absolute gradients of both the retain and forget datasets, enabling the model to selectively freeze weights that are impactful to the data we want to forget (i.e. names, social security numbers) without actually retraining the model. Of course, the latter can be very expensive which is why we need efficient and effective saliency mask generation.

## Getting Started

### Dependencies
The following libraries are used for preprocessing and mask generation scripts:
* PyTorch: For tensor operations, backpropagation, and gradients.
* Transformers Hugging Face: For LlavaConditionalGeneration.
* Datasets Hugging Face: For streamlining and loading the CLEAR dataset.
* Pillow (PIL): For image preprocessing.

### Core Components:
1. CLEARDatasetProcessor: This handles the image-text data and creates label masks to be processed in the future.
2. SaliencyMaskGenerator: The function that processes all the label masks and generates the saliency mask.

### Project Outline:
1. generate_mask.py: Contains the logic, or SaliencyMaskGenerator, for generating the mask.
2. process_dataset.py: Tests generate mask to ensure the functions work properly
3. tensor_generation.py: Utitlizes CLEARDatasetProcessor to generate forget and retain tensors.
4. testing_generate_mask.py: Saves the tensors to .pt, and tests tensor_generation.

## Author

Eric Wang 

UCLA email: soupytwo@g.ucla.edu

Personal email: wangericwork@gmail.com

## Version History

* 0.1
    * Initial Release
